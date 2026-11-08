"""Independent training workers and a matching single-process baseline."""
import math
import time
import traceback
from pathlib import Path

import numpy as np
from hytorch import cuda
from hytorch.core import Config, no_grad, test_mode
from hytorch.recorder import MetricRecorder

from .config import TrainingConfig
from .monitor import EventWriter, write_json
from .task import (create_task, epoch_batches, export_parameters, load_batch,
                   load_parameters, make_optimizer, named_parameters,
                   normalize_gradients, seed_process, select_device, synchronize)
from .transport import Client


def add_numbers(total, values):
    for key, value in values.items():
        total[key] = total.get(key, 0) + value


def make_state(config, manifest):
    seed_process(config.seed)
    task = create_task(config)
    model = task.build_model()
    xp = select_device(config)
    if config.device == "gpu":
        model.to_gpu()
    params = named_parameters(model)
    return task, model, params, xp, task.load_split(manifest, "train")


def accelerator_info(xp):
    if xp is np:
        return {"device": "cpu"}
    return {"device": "gpu", "cupy_version": xp.__version__,
            "reduction_accelerators": xp._core.get_reduction_accelerators(),
            "routine_accelerators": xp._core.get_routine_accelerators()}


def warmup(task, model, dataset, xp, config, rank):
    x, target = load_batch(dataset, [0], xp)
    Config.train = True
    prediction = model(x)
    loss = task.loss(prediction, target)
    model.cleargrads()
    loss.backward()
    normalize_gradients(model, xp)
    synchronize(xp)
    model.cleargrads()
    seed_process((config.seed + rank + 1000) % (2**32))
    if xp is not np:
        xp.random.seed((config.seed + rank + 1000) % (2**32))
    return task.batch_statistics(prediction, target)


def compute_batch(task, model, dataset, indices, xp):
    times = {"data_loading_s": 0.0, "forward_s": 0.0, "backward_s": 0.0,
             "metric_s": 0.0}
    if not len(indices):
        return {"sample_count": 0}, times
    Config.train = True
    synchronize(xp)
    started = time.perf_counter()
    x, target = load_batch(dataset, indices, xp)
    synchronize(xp)
    times["data_loading_s"] = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model(x)
    loss = task.loss(prediction, target)
    synchronize(xp)
    times["forward_s"] = time.perf_counter() - started
    started = time.perf_counter()
    model.cleargrads()
    loss.backward()
    normalize_gradients(model, xp)
    synchronize(xp)
    times["backward_s"] = time.perf_counter() - started
    started = time.perf_counter()
    statistics = task.batch_statistics(prediction, target)
    synchronize(xp)
    if any(not math.isfinite(value) for value in statistics.values()):
        raise FloatingPointError("Non-finite training statistics")
    times["metric_s"] = time.perf_counter() - started
    return statistics, times


def evaluate(task, model, dataset, xp, batch_size):
    started = time.perf_counter()
    totals = {}
    with no_grad(), test_mode():
        for offset in range(0, len(dataset), batch_size):
            x, target = load_batch(dataset, range(offset, min(offset + batch_size, len(dataset))), xp)
            prediction = model(x)
            add_numbers(totals, task.batch_statistics(prediction, target))
    synchronize(xp)
    if any(not math.isfinite(value) for value in totals.values()):
        raise FloatingPointError("Non-finite validation statistics")
    return task.quality(totals), time.perf_counter() - started


def worker_main(config_dict, manifest, run_dir, token, rank, address, running_event):
    config = TrainingConfig.from_dict(config_dict)
    path = Path(run_dir)
    events = EventWriter(path / "workers" / f"worker-{rank}.jsonl", path.name, "worker", rank)
    client = None
    try:
        events.log("starting", device=config.device)
        task, model, params, xp, dataset = make_state(config, manifest)
        events.log("device_ready", **accelerator_info(xp))
        val_data = task.load_split(manifest, "val") if rank == 0 else None
        client = Client(config, address, token, rank)
        initial, weights, _ = client.call("init")
        load_parameters(params, weights, xp)
        del weights
        version = initial["version"]
        warmup(task, model, dataset, xp, config, rank)
        client.call("ready")
        events.log("running", version=version)
        running_event.set()
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.perf_counter()
            totals, epoch_times = {}, {}
            for batch_index, indices in enumerate(epoch_batches(len(dataset), config.global_batch_size,
                                                                 config.seed, epoch), start=1):
                step_started = time.perf_counter()
                shard = np.array_split(indices, config.num_workers)[rank]
                statistics, timing = compute_batch(task, model, dataset, shard, xp)
                started = time.perf_counter()
                gradients = {}
                if len(shard):
                    for name, param in params.items():
                        if param.grad is None:
                            raise ValueError(f"No gradient for parameter {name}")
                        gradients[name] = np.array(cuda.as_numpy(param.grad.data), copy=True)
                synchronize(xp)
                timing["gradient_copy_s"] = time.perf_counter() - started
                response, _, push = client.call("push", gradients, version=version,
                                                sample_count=len(shard), statistics=statistics)
                del gradients
                if response["version"] != version + 1:
                    raise ValueError("PS did not advance exactly one version")
                version = response["version"]
                pulled, weights, pull = client.call("pull", version=version)
                if pulled["version"] != version:
                    raise ValueError("Pulled parameters have the wrong version")
                started = time.perf_counter()
                load_parameters(params, weights, xp)
                synchronize(xp)
                timing["parameter_copy_s"] = time.perf_counter() - started
                del weights
                timing.update(upload_rpc_s=push["rpc_s"], pull_rpc_s=pull["rpc_s"],
                              encode_s=push["encode_s"] + pull["encode_s"],
                              decode_s=push["decode_s"] + pull["decode_s"],
                              sent_bytes=push["sent_bytes"] + pull["sent_bytes"],
                              received_bytes=push["received_bytes"] + pull["received_bytes"],
                              step_s=time.perf_counter() - step_started)
                timing["compute_s"] = timing["forward_s"] + timing["backward_s"]
                timing["communication_s"] = timing["upload_rpc_s"] + timing["pull_rpc_s"]
                add_numbers(totals, statistics)
                add_numbers(epoch_times, timing)
                events.log("batch", epoch=epoch, batch=batch_index, step=version,
                           sample_count=len(shard), server_wait_and_update_s=response["server_wait_and_update_s"],
                           server_aggregation_s=response["aggregation_s"], server_optimizer_s=response["optimizer_s"],
                           **timing, **task.quality(statistics))
                if rank == 0 and (batch_index == 1 or batch_index % 100 == 0):
                    steps = math.ceil(len(dataset) / config.global_batch_size)
                    print(f"[PS] epoch {epoch}/{config.epochs} step {batch_index}/{steps} "
                          f"worker0_mse={task.quality(statistics).get('mse')} "
                          f"step_s={timing['step_s']:.3f}", flush=True)
            train_elapsed = time.perf_counter() - epoch_started
            validation, evaluation_s = ({}, 0.0)
            if rank == 0:
                validation, evaluation_s = evaluate(task, model, val_data, xp, config.eval_batch_size)
            events.log("epoch", epoch=epoch, train_elapsed_s=train_elapsed,
                       statistics=totals, timings=epoch_times, validation=validation)
            client.call("epoch_end", epoch=epoch, statistics=totals, timings=epoch_times,
                        train_elapsed_s=train_elapsed, validation=validation, evaluation_time_s=evaluation_s)
        client.call("finish")
        events.log("completed", version=version)
    except BaseException as exc:
        events.log("failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if client is not None:
            client.close()


def single_main(config_dict, manifest, run_dir, running_event):
    config = TrainingConfig.from_dict(config_dict)
    path = Path(run_dir)
    events = EventWriter(path / "workers" / "worker-0.jsonl", path.name, "single", 0)
    recorder = MetricRecorder(run_dir=run_dir)
    try:
        events.log("starting", device=config.device)
        task, model, params, xp, dataset = make_state(config, manifest)
        events.log("device_ready", **accelerator_info(xp))
        val_data = task.load_split(manifest, "val")
        optimizer = make_optimizer(config, model)
        warmup(task, model, dataset, xp, config, 0)
        epochs, version, best = [], 0, float("inf")
        events.log("running")
        running_event.set()
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.perf_counter()
            totals, timings = {}, {}
            for indices in epoch_batches(len(dataset), config.global_batch_size, config.seed, epoch):
                started = time.perf_counter()
                statistics, step_times = compute_batch(task, model, dataset, indices, xp)
                update_started = time.perf_counter()
                optimizer.update()
                synchronize(xp)
                step_times["optimizer_s"] = time.perf_counter() - update_started
                step_times.update(compute_s=step_times["forward_s"] + step_times["backward_s"],
                                  communication_s=0.0, upload_rpc_s=0.0, pull_rpc_s=0.0,
                                  sent_bytes=0, received_bytes=0, step_s=time.perf_counter() - started)
                version += 1
                add_numbers(totals, statistics)
                add_numbers(timings, step_times)
                events.log("batch", epoch=epoch, step=version, sample_count=len(indices),
                           **step_times, **task.quality(statistics))
                recorder.log_batch(version, epoch, schema_version=1, role="single",
                                   **task.quality(statistics))
                if version == 1 or version % 100 == 0:
                    print(f"[Single] epoch {epoch}/{config.epochs} step {version} "
                          f"mse={task.quality(statistics).get('mse')} "
                          f"step_s={step_times['step_s']:.3f}", flush=True)
            training_s = time.perf_counter() - epoch_started
            validation, evaluation_s = evaluate(task, model, val_data, xp, config.eval_batch_size)
            summary = {"epoch": epoch, "train_time_s": training_s,
                       "epoch_wall_s": time.perf_counter() - epoch_started,
                       "samples": totals["sample_count"], "samples_per_sec": totals["sample_count"] / max(training_s, 1e-12),
                       "aggregation_s": 0.0, "optimizer_s": timings.get("optimizer_s", 0),
                       "worker_time_sums": timings, "train": task.quality(totals),
                       "validation": validation, "evaluation_time_s": evaluation_s}
            epochs.append(summary)
            recorder.log_epoch(epoch, schema_version=1, role="single", epoch_time=training_s,
                               samples_per_sec=summary["samples_per_sec"], **summary["train"],
                               **{"val_" + k: v for k, v in validation.items()})
            started = time.perf_counter()
            weights = export_parameters(params)
            if any(not np.isfinite(value).all() for value in weights.values()):
                raise FloatingPointError("Non-finite model parameters")
            np.savez_compressed(path / "checkpoints" / "last.npz", **weights)
            if validation.get("mse") is not None and validation["mse"] < best:
                best = validation["mse"]
                np.savez_compressed(path / "checkpoints" / "best.npz", **weights)
            del weights
            summary["checkpoint_time_s"] = time.perf_counter() - started
            write_json(path / "training_result.json", {
                "schema_version": 1, "epochs": epochs, "final_version": version,
                "train_time_s": sum(e["train_time_s"] for e in epochs),
                "validation": validation, "checkpoint_kind": "model_weights_only"})
        events.log("completed", version=version)
    except BaseException as exc:
        events.log("failed", error=str(exc), traceback=traceback.format_exc())
        raise
