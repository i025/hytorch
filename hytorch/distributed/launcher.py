"""Windows-safe spawn launcher. Only paths/configuration cross process boundaries."""
import json
import multiprocessing as mp
import os
import secrets
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from .config import TrainingConfig
from .monitor import EventWriter, HardwareMonitor, environment_info, write_json
from .server import server_main
from .task import create_task
from .worker import single_main, worker_main
from .ring_worker import ring_worker_main


def _wait_ready(connection, process, timeout):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if connection.poll(0.1):
            result = connection.recv()
            if not result.get("ok"):
                raise RuntimeError(f"PS initialization failed: {result.get('error')}")
            return tuple(result["address"])
        if process.exitcode is not None:
            raise RuntimeError(f"PS exited during startup (exit code {process.exitcode})")
    raise TimeoutError("PS startup exceeded startup_timeout_s")


def _wait_workers(workers, ps, deadline, running_events, startup_deadline):
    while True:
        for process in workers:
            if process.exitcode not in (None, 0):
                raise RuntimeError(f"{process.name} exited with code {process.exitcode}; see workers/*.jsonl")
        if ps is not None and ps.exitcode is not None:
            raise RuntimeError(f"PS exited unexpectedly with code {ps.exitcode}; see ps_events.jsonl")
        if all(process.exitcode == 0 for process in workers):
            return
        if time.perf_counter() > startup_deadline and not all(event.is_set() for event in running_events):
            missing = [p.name for p, event in zip(workers, running_events) if not event.is_set()]
            raise TimeoutError(f"Workers exceeded startup_timeout_s: {missing}")
        if time.perf_counter() > deadline:
            raise TimeoutError("Training exceeded run_timeout_s")
        for process in workers:
            process.join(timeout=0.05)


def run_training(config):
    """Run one experiment and return summary.json as a dict; raise on any failure.

    Call from an ``if __name__ == '__main__'`` guard on Windows. External agents
    should launch scripts/train_ps.py as a subprocess or use the same guard.
    """
    if isinstance(config, dict):
        config = TrainingConfig.from_dict(config)
    else:
        config = TrainingConfig.from_dict(config.to_dict())
    config.validate(require_data=True)
    # Also cover Python API callers: fresh spawn workers read the inherited
    # environment before importing CuPy. The parent only prepares CPU metadata.
    if sys.platform == "win32" and os.environ.get("CUPY_ACCELERATORS") in (None, ""):
        os.environ["CUPY_ACCELERATORS"] = ","
    config.data_dir = str(Path(config.data_dir).resolve())
    config.output_dir = str(Path(config.output_dir).resolve())
    started = time.perf_counter()
    run_id = f"{config.run_name}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "workers").mkdir()
    write_json(run_dir / "config.json", config.to_dict())
    events = EventWriter(run_dir / "launcher.jsonl", run_id, "launcher")
    context = mp.get_context("spawn")
    stop_event = context.Event()
    processes, workers, running_events = {}, [], []
    monitor, ps, ready_reader, ready_writer = None, None, None, None
    hardware = {}
    ring_connections = []
    status = {"schema_version": 1, "run_id": run_id, "run_dir": str(run_dir), "status": "starting"}
    write_json(run_dir / "status.json", status)
    print(f"[Launcher] run_dir={run_dir}", flush=True)
    try:
        manifest = create_task(config).prepare_manifest()
        if manifest["train_size"] < 1 or manifest["val_size"] < 1:
            raise ValueError("Task manifest must contain nonempty train and validation sets")
        write_json(run_dir / "dataset_manifest.json", manifest)
        write_json(run_dir / "environment.json", environment_info(config))
        events.log("starting", mode=config.mode, worker_count=config.num_workers)
        if config.mode == "ps":
            token = secrets.token_hex(32)
            ready_reader, ready_writer = context.Pipe(duplex=False)
            ps = context.Process(target=server_main, name="parameter-server", args=(
                config.to_dict(), manifest, str(run_dir), token, ready_writer, stop_event))
            ps.start()
            processes[ps.name] = ps
            ready_writer.close()
            address = _wait_ready(ready_reader, ps, config.startup_timeout_s)
            ready_reader.close()
            events.log("ps_ready", host=address[0], port=address[1], pid=ps.pid)
            worker_start = time.perf_counter()
            for rank in range(config.num_workers):
                running_event = context.Event()
                process = context.Process(target=worker_main, name=f"worker-{rank}", args=(
                    config.to_dict(), manifest, str(run_dir), token, rank, address, running_event))
                process.start()
                processes[process.name] = process
                workers.append(process)
                running_events.append(running_event)
        elif config.mode == 'ring':
            token = secrets.token_hex(32)
            worker_start = time.perf_counter()
            for rank in range(config.num_workers):
                parent_pipe, child_pipe = context.Pipe()
                running_event = context.Event()
                process = context.Process(target=ring_worker_main, name=f'ring-worker-{rank}', args=(
                    config.to_dict(), manifest, str(run_dir), token, rank, child_pipe, running_event, stop_event))
                process.start()
                child_pipe.close()
                processes[process.name] = process
                workers.append(process)
                running_events.append(running_event)
                ring_connections.append(parent_pipe)
            addresses = [_wait_ready(conn, proc, config.startup_timeout_s)
                         for conn, proc in zip(ring_connections, workers)]
            for conn in ring_connections:
                conn.send(addresses)
            events.log('ring_ready', addresses=addresses, topology='neighbor_only_ring', parameter_server=False)
        else:
            worker_start = time.perf_counter()
            running_event = context.Event()
            process = context.Process(target=single_main, name="single-worker", args=(
                config.to_dict(), manifest, str(run_dir), running_event))
            process.start()
            processes[process.name] = process
            workers.append(process)
            running_events.append(running_event)
        monitor = HardwareMonitor(config, run_dir, processes)
        monitor.start()
        status["status"] = "running"
        status["processes"] = {name: p.pid for name, p in processes.items()}
        write_json(run_dir / "status.json", status)
        _wait_workers(workers, ps, started + config.run_timeout_s, running_events,
                      worker_start + config.startup_timeout_s)
        stop_event.set()
        if ps is not None:
            ps.join(timeout=10)
            if ps.exitcode != 0:
                raise RuntimeError(f"PS did not stop cleanly (exit code {ps.exitcode})")
        hardware = monitor.stop()
        monitor = None
        result = json.loads((run_dir / "training_result.json").read_text(encoding="utf-8"))
        if len(result["epochs"]) != config.epochs:
            raise RuntimeError("Training result has missing epochs")
        summary = dict(result, schema_version=1, run_id=run_id, run_dir=str(run_dir),
                       status="completed", mode=config.mode, transport=config.transport if config.mode != "single" else "none",
                       num_workers=config.num_workers, global_batch_size=config.global_batch_size,
                       dataset_fingerprint=manifest["fingerprint"], hardware=hardware,
                       total_wall_s=time.perf_counter() - started,
                       metrics_path=str(run_dir / "metrics.jsonl"),
                       checkpoint_path=str(run_dir / "checkpoints" / "last.npz"))
        write_json(run_dir / "summary.json", summary)
        status["status"] = "completed"
        write_json(run_dir / "status.json", status)
        events.log("completed", total_wall_s=summary["total_wall_s"])
        return summary
    except BaseException as exc:
        status.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        write_json(run_dir / "status.json", status)
        write_json(run_dir / "summary.json", dict(status, total_wall_s=time.perf_counter() - started))
        events.log("failed", error=str(exc))
        raise
    finally:
        stop_event.set()
        for process in processes.values():
            if process.is_alive():
                process.terminate()
        for process in processes.values():
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        if monitor is not None:
            hardware = monitor.stop()
        if hardware:
            write_json(run_dir / "hardware_summary.json", hardware)
        for connection in (ready_reader, ready_writer):
            if connection is not None:
                connection.close()
        for connection in ring_connections:
            connection.close()
