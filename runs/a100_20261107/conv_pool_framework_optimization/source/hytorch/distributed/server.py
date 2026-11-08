"""CPU Parameter Server: versioned, sample-weighted synchronous updates."""
import hmac
import math
import threading
import time
from pathlib import Path

import numpy as np
from hytorch.core import Variable
from hytorch.recorder import MetricRecorder

from .config import TrainingConfig
from .monitor import EventWriter, write_json
from .task import create_task, export_parameters, make_optimizer, named_parameters, seed_process
from .transport import start_transport


class ParameterServer:
    def __init__(self, config, manifest, run_dir, token):
        self.config, self.manifest, self.token = config, manifest, token
        self.run_dir = Path(run_dir)
        seed_process(config.seed)
        self.task = create_task(config)
        self.model = self.task.build_model()
        self.params = named_parameters(self.model)
        self.optimizer = make_optimizer(config, self.model)
        self.recorder = MetricRecorder(run_dir=run_dir)
        self.events = EventWriter(self.run_dir / "ps_events.jsonl", self.run_dir.name, "ps")
        self.condition = threading.Condition(threading.RLock())
        self.groups, self.initialized, self.pulled = {}, set(), {}
        self.version, self.failed, self.completed = 0, None, False
        self.steps_per_epoch = math.ceil(manifest["train_size"] / config.global_batch_size)
        self.epochs = []
        self.epoch_started = None
        self.aggregate_s = self.update_s = 0.0
        self.best_quality = float("inf")

    def abort(self, message):
        with self.condition:
            if not self.completed and self.failed is None:
                self.failed = message
                self.events.log("failed", error=message, version=self.version)
            self.condition.notify_all()

    def _barrier(self, key, rank, value, reduce):
        group = self.groups.setdefault(key, {"values": {}, "done": False, "departed": 0})
        if rank in group["values"]:
            raise ValueError(f"Duplicate contribution from worker {rank}: {key}")
        group["values"][rank] = value
        if len(group["values"]) == self.config.num_workers:
            group["result"] = reduce(group["values"])
            group["done"] = True
            self.condition.notify_all()
        deadline = time.perf_counter() + self.config.rpc_timeout_s
        while not group["done"]:
            if self.failed:
                raise RuntimeError(self.failed)
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                missing = sorted(set(range(self.config.num_workers)) - set(group["values"]))
                raise TimeoutError(f"Barrier {key} timed out; missing workers {missing}")
            self.condition.wait(timeout=remaining)
        group["departed"] += 1
        if group["departed"] == self.config.num_workers:
            del self.groups[key]
        return group["result"]

    def dispatch(self, metadata, tensors):
        with self.condition:
            supplied = metadata.get("token", "")
            if not isinstance(supplied, str) or not hmac.compare_digest(supplied, self.token):
                raise ValueError("Invalid run token")
            rank = metadata.get("rank")
            if type(rank) is not int or not 0 <= rank < self.config.num_workers:
                raise ValueError("Invalid worker rank")
            if self.failed:
                raise RuntimeError(self.failed)
            op = metadata.get("op")
            if op == "init":
                if rank in self.initialized or self.version:
                    raise ValueError("Worker already initialized or training already started")
                self.initialized.add(rank)
                self.pulled[rank] = 0
                return {"ok": True, "version": 0}, export_parameters(self.params)
            if rank not in self.initialized:
                raise ValueError("Worker must initialize first")
            if op == "ready":
                if self.epoch_started is not None:
                    raise ValueError("Training already started")
                def begin(values):
                    self.epoch_started = time.perf_counter()
                    self.events.log("running", worker_count=len(values))
                    return {"ok": True, "version": 0}
                return self._barrier(("ready", 0), rank, {}, begin), {}
            if op == "push":
                return self._push(rank, metadata, tensors), {}
            if op == "pull":
                if metadata.get("version") != self.version or self.pulled[rank] != self.version - 1:
                    raise ValueError("Pull version mismatch or duplicate pull")
                self.pulled[rank] = self.version
                return {"ok": True, "version": self.version}, export_parameters(self.params)
            if op == "epoch_end":
                epoch = metadata.get("epoch")
                if epoch != len(self.epochs) + 1 or self.version != epoch * self.steps_per_epoch:
                    raise ValueError("Epoch boundary/version mismatch")
                if self.pulled[rank] != self.version:
                    raise ValueError("Worker must pull final parameters before evaluation")
                return self._barrier(("epoch", epoch), rank, metadata, self._finish_epoch), {}
            if op == "finish":
                if len(self.epochs) != self.config.epochs:
                    raise ValueError("Training is incomplete")
                def finish(values):
                    self.completed = True
                    self.events.log("completed", version=self.version)
                    return {"ok": True, "version": self.version}
                return self._barrier(("finish", 0), rank, {}, finish), {}
            raise ValueError(f"Unsupported operation: {op}")

    def _push(self, rank, metadata, gradients):
        started = time.perf_counter()
        version = metadata.get("version")
        if self.epoch_started is None or version != self.version or self.pulled[rank] != version:
            raise ValueError("Stale gradient or worker did not pull the current parameters")
        epoch = version // self.steps_per_epoch + 1
        step = version % self.steps_per_epoch
        if epoch != len(self.epochs) + 1 or epoch > self.config.epochs:
            raise ValueError("Worker crossed an epoch barrier")
        total = min(self.config.global_batch_size,
                    self.manifest["train_size"] - step * self.config.global_batch_size)
        expected = total // self.config.num_workers + int(rank < total % self.config.num_workers)
        count = metadata.get("sample_count")
        if type(count) is not int or count != expected:
            raise ValueError("Worker sample count does not match its global-batch shard")
        if set(gradients) != (set(self.params) if count else set()):
            raise ValueError("Missing or unexpected parameter gradients")
        for name, grad in gradients.items():
            param = self.params[name].data
            if grad.shape != param.shape or grad.dtype != param.dtype or not np.isfinite(grad).all():
                raise ValueError(f"Gradient shape/dtype/value mismatch: {name}")
        statistics = metadata["statistics"]
        if statistics.get("sample_count") != count:
            raise ValueError("Statistics sample count mismatch")
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in statistics.values()):
            raise ValueError("Non-finite training statistics")
        result = self._barrier(("step", version), rank, (count, gradients, statistics), self._update)
        # Includes barrier wait and server computation, excludes decode/network/response encoding.
        return dict(result, server_wait_and_update_s=time.perf_counter() - started)

    def _update(self, values):
        started = time.perf_counter()
        count = sum(value[0] for value in values.values())
        if count <= 0:
            raise ValueError("An optimizer step must contain at least one sample")
        statistics = {}
        for rank in sorted(values):
            for key, value in values[rank][2].items():
                statistics[key] = statistics.get(key, 0) + value
        self.model.cleargrads()
        for name, param in self.params.items():
            gradient = np.zeros_like(param.data)
            for rank in sorted(values):
                n, grads, _ = values[rank]
                if n:
                    gradient += grads[name] * (n / count)
            param.grad = Variable(gradient)
        aggregate_s = time.perf_counter() - started
        started = time.perf_counter()
        self.optimizer.update()
        if any(not np.isfinite(p.data).all() for p in self.params.values()):
            raise FloatingPointError("Optimizer produced non-finite parameters")
        update_s = time.perf_counter() - started
        self.version += 1
        self.aggregate_s += aggregate_s
        self.update_s += update_s
        self.recorder.log_batch(self.version, (self.version - 1) // self.steps_per_epoch + 1,
                                schema_version=1, role="ps", samples=count,
                                aggregation_s=aggregate_s, optimizer_s=update_s,
                                **self.task.quality(statistics))
        return {"ok": True, "version": self.version, "aggregation_s": aggregate_s,
                "optimizer_s": update_s}

    def _finish_epoch(self, values):
        epoch = len(self.epochs) + 1
        totals, timings = {}, {}
        for value in values.values():
            for key, number in value["statistics"].items():
                totals[key] = totals.get(key, 0) + number
            for key, number in value["timings"].items():
                timings[key] = timings.get(key, 0) + number
        if totals["sample_count"] != self.manifest["train_size"]:
            raise ValueError("Epoch samples were duplicated or dropped")
        # Each worker reports time at completion of its training loop, before validation.
        # PS wall clock at entry to epoch_end includes validation; max worker duration doesn't.
        training_s = max(v["train_elapsed_s"] for v in values.values())
        validation = values[0]["validation"]
        summary = {"epoch": epoch, "train_time_s": training_s,
                   "epoch_wall_s": time.perf_counter() - self.epoch_started,
                   "samples": totals["sample_count"],
                   "samples_per_sec": totals["sample_count"] / max(training_s, 1e-12),
                   "aggregation_s": self.aggregate_s, "optimizer_s": self.update_s,
                   "worker_time_sums": timings,
                   "train": self.task.quality(totals), "validation": validation,
                   "evaluation_time_s": values[0]["evaluation_time_s"]}
        self.epochs.append(summary)
        self.recorder.log_epoch(epoch, schema_version=1, role="ps", epoch_time=training_s,
                                samples_per_sec=summary["samples_per_sec"],
                                **summary["train"],
                                **{"val_" + k: v for k, v in validation.items()})
        checkpoint_started = time.perf_counter()
        self._checkpoint("last.npz")
        quality = validation.get("mse")
        if quality is not None and quality < self.best_quality:
            self.best_quality = quality
            self._checkpoint("best.npz")
        summary["checkpoint_time_s"] = time.perf_counter() - checkpoint_started
        write_json(self.run_dir / "training_result.json", self.result())
        print(f"[PS] epoch {epoch}/{self.config.epochs} complete: "
              f"train_s={training_s:.2f} val_mse={validation.get('mse')} "
              f"val_mae={validation.get('mae')}", flush=True)
        self.aggregate_s = self.update_s = 0.0
        self.epoch_started = time.perf_counter()
        return {"ok": True, "version": self.version, "epoch": epoch}

    def _checkpoint(self, filename):
        np.savez_compressed(self.run_dir / "checkpoints" / filename, **export_parameters(self.params))

    def result(self):
        return {"schema_version": 1, "epochs": self.epochs, "final_version": self.version,
                "train_time_s": sum(e["train_time_s"] for e in self.epochs),
                "validation": self.epochs[-1]["validation"] if self.epochs else {},
                "checkpoint_kind": "model_weights_only"}


def server_main(config_dict, manifest, run_dir, token, ready_pipe, stop_event):
    config = TrainingConfig.from_dict(config_dict)
    server = None
    stop_transport = None
    try:
        server = ParameterServer(config, manifest, run_dir, token)
        address, stop_transport = start_transport(config, server.dispatch, server.abort)
        ready_pipe.send({"ok": True, "address": address})
        ready_pipe.close()
        while not stop_event.wait(0.1):
            if server.failed:
                raise RuntimeError(server.failed)
        if not server.completed:
            raise RuntimeError("Launcher stopped PS before all workers finished")
    except BaseException as exc:
        if server is not None:
            server.abort(str(exc))
        try:
            ready_pipe.send({"ok": False, "error": str(exc)})
        except (OSError, EOFError):
            pass
        raise
    finally:
        ready_pipe.close()
        if stop_transport is not None:
            stop_transport()
