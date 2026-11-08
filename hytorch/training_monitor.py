"""Optional training telemetry; no CUDA, NVML or psutil imports at module import.

GPU indices are *logical CUDA indices*, resolved by runtime UUID/PCI identity.
Throughput mode never synchronizes phases/steps. ``stop`` (or explicit
``flush_events`` at a safe boundary) completes the recorded streams. Event
intervals include stream idle time and are not kernel-only measurements.
"""
from contextlib import contextmanager, nullcontext
import csv
import importlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time


_NULL = nullcontext()
_PHASES = ("data_load", "data_prepare", "h2d", "forward", "loss", "backward", "optimizer")


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _finite(value):
    if value is None:
        return None
    return value if math.isfinite(value) else None


class _CudaTimer:
    def __init__(self, backend, device):
        self.backend, self.device = backend, device
        self.module = importlib.import_module("cupy" if backend == "cupy" else "torch")
        if backend == "cupy":
            self.selector = ("pci", self.module.cuda.Device(device).pci_bus_id)
            self.pool = self.module.get_default_memory_pool()
        else:
            self.selector = ("uuid", _text(self.module.cuda.get_device_properties(device).uuid))

    def stream(self):
        if self.backend == "cupy":
            if self.module.cuda.runtime.getDevice() != self.device:
                raise ValueError("TrainingMonitor current CUDA device changed")
            return self.module.cuda.get_current_stream()
        return self.module.cuda.current_stream(self.device)

    def stream_id(self, stream):
        return stream.ptr if self.backend == "cupy" else stream.cuda_stream

    def event(self):
        return self.module.cuda.Event() if self.backend == "cupy" else self.module.cuda.Event(enable_timing=True)

    def elapsed(self, start, end):
        ms = self.module.cuda.get_elapsed_time(start, end) if self.backend == "cupy" else start.elapsed_time(end)
        return ms / 1000

    def allocator(self):
        if self.backend == "cupy":
            return {"cupy_active_bytes": self.pool.used_bytes(), "cupy_reserved_bytes": self.pool.total_bytes()}
        return {"torch_allocated_bytes": self.module.cuda.memory_allocated(self.device),
                "torch_reserved_bytes": self.module.cuda.memory_reserved(self.device)}


class _Resources:
    """Owned by the sampling thread. Unsupported counters stay null."""
    def __init__(self, selector):
        self.nvml = self.handle = self.psutil = self.process = None
        self.errors = []
        self.identity = dict(gpu_uuid=None, physical_gpu_index=None, gpu_name=None, pci_bus_id=None)
        self.first_cpu = True
        try:
            self.psutil = importlib.import_module("psutil")
            self.process = self.psutil.Process(os.getpid())
            self.psutil.cpu_percent(None)
            self.process.cpu_percent(None)
        except Exception as exc:
            self.errors.append("psutil: " + str(exc))
            self.psutil = self.process = None
        if selector is None:
            self.errors.append("NVML: no runtime UUID/PCI identity; no physical index guessed")
            return
        try:
            nvml = importlib.import_module("pynvml")
            nvml.nvmlInit()
            self.nvml = nvml
            kind, value = selector
            self.handle = (nvml.nvmlDeviceGetHandleByUUID(value) if kind == "uuid"
                           else nvml.nvmlDeviceGetHandleByPciBusId(value))
            self.identity = dict(gpu_uuid=_text(nvml.nvmlDeviceGetUUID(self.handle)),
                                 physical_gpu_index=nvml.nvmlDeviceGetIndex(self.handle),
                                 gpu_name=_text(nvml.nvmlDeviceGetName(self.handle)),
                                 pci_bus_id=_text(nvml.nvmlDeviceGetPciInfo(self.handle).busId))
        except Exception as exc:
            self.errors.append("NVML: " + str(exc))
            self.handle = None

    def sample(self):
        gpu = {k: None for k in ("gpu_utilization_pct", "memory_utilization_pct", "gpu_used_bytes",
                               "gpu_total_bytes", "gpu_temperature_c", "gpu_power_w", "sm_clock_mhz",
                               "memory_clock_mhz", "process_gpu_used_bytes", "compute_processes_json")}
        cpu = {k: None for k in ("system_cpu_pct", "process_cpu_pct", "process_rss_bytes", "process_threads")}
        statuses = {}

        def get(label, fn):
            try:
                value = fn()
                statuses[label] = "OK"
                return value
            except Exception as exc:
                statuses[label] = type(exc).__name__ + ": " + str(exc)
                return None

        if self.handle is not None:
            n, h = self.nvml, self.handle
            util = get("utilization", lambda: n.nvmlDeviceGetUtilizationRates(h))
            mem = get("device_memory", lambda: n.nvmlDeviceGetMemoryInfo(h))
            if util is not None:
                gpu.update(gpu_utilization_pct=_finite(util.gpu), memory_utilization_pct=_finite(util.memory))
            if mem is not None:
                gpu.update(gpu_used_bytes=mem.used, gpu_total_bytes=mem.total)
            gpu["gpu_temperature_c"] = get("temperature", lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU))
            power = get("power", lambda: n.nvmlDeviceGetPowerUsage(h))
            gpu["gpu_power_w"] = power / 1000 if power is not None else None
            gpu["sm_clock_mhz"] = get("sm_clock", lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM))
            gpu["memory_clock_mhz"] = get("memory_clock", lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM))
            procs = get("compute_processes", lambda: n.nvmlDeviceGetComputeRunningProcesses(h))
            if procs is not None:
                rows = []
                unavailable = getattr(n, "NVML_VALUE_NOT_AVAILABLE", 2**64-1)
                unavailable = getattr(unavailable, "value", unavailable)
                for p in procs:
                    used = getattr(p, "usedGpuMemory", None)
                    rows.append(dict(pid=p.pid, used_bytes=used if used is not None and used < unavailable else None))
                gpu["compute_processes_json"] = json.dumps(rows)
                own = [p["used_bytes"] for p in rows if p["pid"] == os.getpid()]
                gpu["process_gpu_used_bytes"] = sum(own) if own and all(x is not None for x in own) else None
        else:
            statuses["nvml"] = "UNAVAILABLE"
        if self.psutil is not None:
            if not self.first_cpu:
                cpu["system_cpu_pct"] = get("system_cpu", lambda: self.psutil.cpu_percent(None))
                cpu["process_cpu_pct"] = get("process_cpu", lambda: self.process.cpu_percent(None))
            else:
                statuses["cpu_percent"] = "PRIMING: first sample is null"
            cpu["process_rss_bytes"] = get("rss", lambda: self.process.memory_info().rss)
            cpu["process_threads"] = get("threads", lambda: self.process.num_threads())
        else:
            statuses["psutil"] = "UNAVAILABLE"
        self.first_cpu = False
        return gpu, cpu, statuses

    def close(self):
        if self.nvml is not None:
            self.nvml.nvmlShutdown()


class TrainingMonitor:
    """Reusable optional telemetry with explicitly named timing semantics.

    ``gpu_index`` is a logical CUDA index. Supply ``cuda_backend='cupy'`` or
    ``'torch'`` to resolve it and record events; for resource-only clients without
    CUDA imports, supply an explicit ``gpu_uuid`` or ``gpu_pci_bus_id`` instead.
    No numeric CUDA_VISIBLE_DEVICES mapping is guessed.

    ``mode='throughput'`` records host_iteration_s; CUDA step_wall_s stays null.
    ``mode='latency'`` synchronizes once at step end, never at phase boundaries.
    Allocator peaks are opt-in diagnostics: CuPy hook covers allocations on the
    calling thread/default pool; PyTorch resets that device's peak counters.
    """
    def __init__(self, enabled=False, gpu_index=0, sample_interval=0.2,
                 output_dir="runs/monitor", *, timing=True, resources=True,
                 cuda_backend="cupy", mode="throughput", gpu_uuid=None,
                 gpu_pci_bus_id=None, allocator_peaks=False):
        if not math.isfinite(sample_interval) or sample_interval <= 0:
            raise ValueError("sample_interval must be finite and positive")
        if mode not in ("throughput", "latency") or cuda_backend not in (None, "cupy", "torch"):
            raise ValueError("Invalid mode or CUDA backend")
        if gpu_uuid is not None and gpu_pci_bus_id is not None:
            raise ValueError("Choose UUID or PCI identity, not both")
        if (gpu_uuid or gpu_pci_bus_id) and cuda_backend is not None and gpu_index is not None:
            raise ValueError("Explicit NVML identity requires cuda_backend=None; CUDA clients use runtime identity")
        if gpu_index is not None and (type(gpu_index) is not int or gpu_index < 0):
            raise ValueError("gpu_index must be a nonnegative logical CUDA index or None")
        self.enabled, self.timing, self.resources = enabled, timing, resources
        self.gpu_index, self.sample_interval = gpu_index, sample_interval
        self.output_dir, self.cuda_backend, self.mode = Path(output_dir), cuda_backend, mode
        self.explicit_selector = ("uuid", gpu_uuid) if gpu_uuid else (("pci", gpu_pci_bus_id) if gpu_pci_bus_id else None)
        self.allocator_peaks = allocator_peaks
        self.running = False
        self.thread = None
        self.run_dir = None
        self._reset()

    def _reset(self):
        self.steps, self.gpu_samples, self.cpu_samples, self.errors = [], [], [], []
        self._pending, self._streams, self._ids = [], {}, set()
        self._active = self._phase_name = self._timer = self._hook = None
        self._peak = {}
        self.event_status = "DISABLED" if self.cuda_backend is None else "NOT_STARTED"
        self.started_ns = self.stopped_ns = None

    def start(self):
        if not self.enabled:
            return self
        if self.running or (self.thread is not None and self.thread.is_alive()):
            raise RuntimeError("TrainingMonitor is already running")
        self._reset()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=self.output_dir))
        self.owner = threading.get_ident()
        self.started_ns, self.started_unix_s = time.perf_counter_ns(), time.time()
        selector = self.explicit_selector
        if self.cuda_backend is not None and self.gpu_index is not None:
            try:
                self._timer = _CudaTimer(self.cuda_backend, self.gpu_index)
                # Runtime identity wins; reject misleading explicit cross-device selection.
                if selector is not None and selector != self._timer.selector:
                    raise ValueError("Use runtime mapping OR an explicit resource identity, not both")
                selector = self._timer.selector
                self.event_status = "ENABLED" if self.timing else "DISABLED"
            except Exception as exc:
                self.errors.append("CUDA timer unavailable: " + str(exc))
                self._timer = None
                self.event_status = "UNAVAILABLE"
        self.selector = selector
        self.running = True
        if self.allocator_peaks and self._timer is not None:
            self._start_peak_tracking()
        self._stop_event = threading.Event()
        if self.resources:
            self.thread = threading.Thread(target=self._sample_loop, name="HyTorchTrainingMonitor-"+self.run_dir.name, daemon=True)
            self.thread.start()
        return self

    def _start_peak_tracking(self):
        timer = self._timer
        if timer.backend == "torch":
            timer.module.cuda.reset_peak_memory_stats(timer.device)
            self._peak["source"] = "torch_allocator_peak_counters_since_start"
            return
        monitor = self
        class PeakHook(timer.module.cuda.MemoryHook):
            def malloc_postprocess(self, **kwargs):
                monitor._track_cupy_peak()
        self._hook = PeakHook()
        self._hook.__enter__()
        self._track_cupy_peak()

    def _track_cupy_peak(self):
        used = self._timer.pool.used_bytes()
        reserved = self._timer.pool.total_bytes()
        if used >= self._peak.get("active_bytes", -1):
            self._peak.update(active_bytes=used, monotonic_ns=time.perf_counter_ns(),
                              phase=self._phase_name, step_id=self._active["step_id"] if self._active else None)
        self._peak["reserved_bytes"] = max(reserved, self._peak.get("reserved_bytes", 0))
        self._peak["source"] = "cupy_default_pool_allocation_hook_main_thread; host_allocation_phase_not_GPU_execution_phase"

    def _sample_loop(self):
        provider = None
        try:
            provider = _Resources(self.selector)
            self.errors.extend(provider.errors)
            allocator_available = self._timer is not None
            # CuPy MemoryPool counters are per *current* device. A fresh sampler
            # thread defaults to logical device 0; bind nonzero targets once on
            # this thread only. This neither synchronizes nor touches device 0.
            if self._timer is not None and self._timer.backend == "cupy" and self._timer.device != 0:
                try:
                    self._timer.module.cuda.runtime.setDevice(self._timer.device)
                except Exception as exc:
                    allocator_available = False
                    self.errors.append("sampler allocator device: " + str(exc))
            deadline = time.perf_counter_ns()
            interval_ns = int(self.sample_interval * 1e9)
            while not self._stop_event.is_set():
                now = time.perf_counter_ns()
                unix_s = time.time()
                active, phase = self._active, self._phase_name
                gpu, cpu, status = provider.sample()
                common = dict(timestamp_unix_s=unix_s, monotonic_ns=now,
                              elapsed_s=(now-self.started_ns)/1e9, scheduled_monotonic_ns=deadline,
                              schedule_delay_s=max(0, now-deadline)/1e9, sample_interval_s=self.sample_interval,
                              step_id=active["step_id"] if active else None,
                              host_phase=phase, scope="sampled; not a per-step GPU attribution")
                gpu.update(common, **provider.identity, logical_cuda_index=self.gpu_index if self._timer else None,
                           cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), scope="whole_physical_GPU_including_other_processes")
                if allocator_available:
                    try:
                        gpu.update(self._timer.allocator())
                        status["allocator"] = "SAMPLED; not exact peak"
                    except Exception as exc:
                        status["allocator"] = str(exc)
                gpu["status_json"] = json.dumps(status)
                cpu.update(common, status_json=json.dumps(status), pid=os.getpid(), cpu_percent_unit="system:0..100; process:100 per busy core")
                end = time.perf_counter_ns()
                gpu["sample_collection_s"] = cpu["sample_collection_s"] = (end-now)/1e9
                self.gpu_samples.append(gpu)
                self.cpu_samples.append(cpu)
                deadline += interval_ns
                if deadline < end:
                    deadline += ((end-deadline)//interval_ns+1)*interval_ns
                self._stop_event.wait(max(0, (deadline-time.perf_counter_ns())/1e9))
        except Exception as exc:
            self.errors.append("sampling: " + repr(exc))
        finally:
            if provider is not None:
                try:
                    provider.close()
                except Exception as exc:
                    self.errors.append("NVML close: " + str(exc))

    def _check_owner(self):
        if not self.running or self.owner != threading.get_ident():
            raise RuntimeError("Timing contexts require a running monitor on its owner thread")

    def _event(self):
        if self._timer is None or not self.timing or self.event_status != "ENABLED":
            return None
        try:
            stream = self._timer.stream()
            key = self._timer.stream_id(stream)
            self._streams[key] = stream
            event = self._timer.event()
            event.record(stream)
            return event, key
        except Exception as exc:
            self.errors.append("CUDA event: " + repr(exc))
            self.event_status = "UNAVAILABLE"
            return None

    def step(self, step_id=None, *, epoch=None, batch_size=None, data_bytes=None):
        if not self.enabled or not self.timing:
            return _NULL
        return self._step(step_id, epoch, batch_size, data_bytes)

    @contextmanager
    def _step(self, step_id, epoch, batch_size, data_bytes):
        self._check_owner()
        if self._active is not None:
            raise RuntimeError("Nested steps are not supported")
        step_id = len(self.steps) if step_id is None else step_id
        if step_id in self._ids:
            raise ValueError("step_id must be unique within a run")
        self._ids.add(step_id)
        row = dict(step_id=step_id, epoch=epoch, batch_size=batch_size, data_bytes=data_bytes,
                   mode=self.mode, timestamp_unix_s=time.time(), start_monotonic_ns=time.perf_counter_ns(),
                   status="OK", error=None, step_wall_s=None, step_gpu_interval_s=None,
                   gpu_event_status=self.event_status)
        for name in _PHASES:
            row[name+"_cpu_s"] = row[name+"_gpu_event_s"] = None
        self._active = row
        first = self._event()
        try:
            yield self
        except BaseException as exc:
            row.update(status="ERROR", error=type(exc).__name__+": "+str(exc))
            raise
        finally:
            try:
                last = self._event()
                if first is not None and last is not None:
                    self._pending.append((row, "step_gpu_interval_s", first, last))
                if self.event_status == "UNAVAILABLE":
                    row["gpu_event_status"] = "UNAVAILABLE"
                row["host_iteration_s"] = (time.perf_counter_ns()-row["start_monotonic_ns"])/1e9
                if self.mode == "latency" and self._timer is not None:
                    for stream in self._streams.values():
                        stream.synchronize()
                if (self.mode == "latency" and self._timer is not None) or self.cuda_backend is None or self.gpu_index is None:
                    row["step_wall_s"] = (time.perf_counter_ns()-row["start_monotonic_ns"])/1e9
                row["end_monotonic_ns"] = time.perf_counter_ns()
            finally:
                self.steps.append(row)
                self._active = None

    def set_step_metadata(self, *, batch_size=None, data_bytes=None):
        if not self.enabled or not self.timing:
            return
        self._check_owner()
        if self._active is None:
            raise RuntimeError("No active step")
        if batch_size is not None:
            self._active["batch_size"] = batch_size
        if data_bytes is not None:
            self._active["data_bytes"] = data_bytes

    def phase(self, name, *, gpu=True):
        if not self.enabled or not self.timing:
            return _NULL
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError("phase name must be an identifier")
        return self._phase(name, gpu)

    @contextmanager
    def _phase(self, name, gpu):
        self._check_owner()
        if self._active is None or self._phase_name is not None:
            raise RuntimeError("Phases require an active step and cannot be nested")
        self._phase_name = name
        start = time.perf_counter_ns()
        first = self._event() if gpu else None
        try:
            yield
        finally:
            try:
                last = self._event() if gpu else None
                key = name+"_cpu_s"
                self._active[key] = (self._active.get(key) or 0)+(time.perf_counter_ns()-start)/1e9
                if first is not None and last is not None:
                    self._pending.append((self._active, name+"_gpu_event_s", first, last))
            finally:
                self._phase_name = None

    def flush_events(self):
        """Explicit safe completion boundary; call between epochs for long runs.

        Retained event storage is O(steps * phases) since the preceding flush.
        No synchronization is performed by the resource sampling thread.
        """
        if not self.enabled:
            return
        self._check_owner()
        if self._active is not None:
            raise RuntimeError("flush_events must be outside a step")
        for stream in self._streams.values():
            stream.synchronize()
        for row, key, first, last in self._pending:
            if first[1] != last[1]:
                row["gpu_event_status"] = "UNSUPPORTED_CROSS_STREAM_INTERVAL"
                row[key] = None
            else:
                row[key] = (row.get(key) or 0)+self._timer.elapsed(first[0], last[0])
        self._pending.clear()

    def stop(self):
        if not self.enabled or not self.running:
            return
        self._check_owner()
        if self._active is not None:
            raise RuntimeError("Cannot stop inside a step")
        try:
            self.flush_events()
            if self.allocator_peaks and self._timer is not None and self.cuda_backend == "torch":
                t = self._timer
                self._peak.update(active_bytes=t.module.cuda.max_memory_allocated(t.device),
                                  reserved_bytes=t.module.cuda.max_memory_reserved(t.device))
        finally:
            if self._hook is not None:
                self._hook.__exit__(None, None, None)
                self._hook = None
            self._stop_event.set()
            if self.thread is not None:
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    self.errors.append("sampler did not stop within 5 seconds; samples may be incomplete")
            self.stopped_ns = time.perf_counter_ns()
            self.running = False
            self.export()

    def summary(self):
        peaks = {}
        for key in ("gpu_used_bytes", "process_gpu_used_bytes", "cupy_active_bytes", "cupy_reserved_bytes",
                    "torch_allocated_bytes", "torch_reserved_bytes"):
            vals = [r.get(key) for r in self.gpu_samples if r.get(key) is not None]
            peaks[key] = max(vals) if vals else None
        def mean(key):
            values = [r[key] for r in self.steps if r.get(key) is not None]
            return sum(values)/len(values) if values else None
        return dict(enabled=self.enabled, timing=self.timing, resources=self.resources, mode=self.mode,
                    cuda_backend=self.cuda_backend, logical_cuda_index=self.gpu_index,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    resolved_selector=getattr(self, "selector", None), gpu_event_status=self.event_status,
                    run_dir=str(self.run_dir) if self.run_dir else None, steps=len(self.steps),
                    sample_count=len(self.gpu_samples), sample_interval_s=self.sample_interval,
                    sampled_peaks_bytes=peaks, peak_kind="sampled maxima; not instantaneous/exact device peaks",
                    allocator_tracked_peak=self._peak or None, errors=list(self.errors),
                    mean_host_iteration_s=mean("host_iteration_s"),
                    mean_completed_step_wall_s=mean("step_wall_s"),
                    mean_step_gpu_interval_s=mean("step_gpu_interval_s"),
                    phase_means_cpu_s={p: mean(p+"_cpu_s") for p in _PHASES},
                    phase_means_gpu_event_s={p: mean(p+"_gpu_event_s") for p in _PHASES},
                    run_completed_wall_s=(self.stopped_ns-self.started_ns)/1e9 if self.stopped_ns else None,
                    event_limitation="stream intervals include idle gaps; concurrent streams may overlap; do not sum across streams")

    def export(self):
        if not self.enabled or self.run_dir is None:
            return
        if self.running:
            raise RuntimeError("Export after stop; use flush_events at an explicit safe boundary")
        for name, rows, defaults in (("STEP_TIMINGS", self.steps, ["step_id", "epoch", "batch_size", "step_wall_s", "host_iteration_s"]),
                                      ("GPU_SAMPLES", self.gpu_samples, ["timestamp_unix_s", "gpu_uuid", "gpu_used_bytes"]),
                                      ("CPU_SAMPLES", self.cpu_samples, ["timestamp_unix_s", "system_cpu_pct", "process_cpu_pct"])):
            fields = list(dict.fromkeys(defaults + [k for row in rows for k in row]))
            with (self.run_dir/(name+".csv")).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            (self.run_dir/(name+".json")).write_text(json.dumps(rows, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        (self.run_dir/"SUMMARY.json").write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        try:
            self.stop()
        except Exception:
            if exc_type is None:
                raise
            # Do not replace the original training exception with a telemetry error.
            self.errors.append("monitor cleanup/export failed during training exception")
        return False
