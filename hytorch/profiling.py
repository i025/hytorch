"""Training phase timing and device-wide telemetry, without a CUDA dependency."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import math


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class ResourceMonitor:
    """Sampled whole-device peak is NOT an exact per-process allocation peak."""
    def __init__(self, path, gpu_id=None, interval=1.0):
        if interval < 0.2:
            raise ValueError("Use interval >= 0.2 seconds to limit monitoring overhead")
        self.path, self.gpu_id, self.interval = Path(path), gpu_id, interval
        self.rows, self.errors = [], []
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        try:
            import psutil
            proc = psutil.Process()
            psutil.cpu_percent(None)
            proc.cpu_percent(None)
        except ImportError:
            psutil = proc = None
        with self.path.open("w", encoding="utf-8") as f:
            while not self.stop_event.is_set():
                row = dict(timestamp=time.time(), gpu_id=self.gpu_id,
                           scope="whole_device", gpu_utilization_pct=None,
                           gpu_memory_used_mib=None, gpu_memory_total_mib=None,
                           gpu_temperature_c=None, gpu_power_w=None,
                           cpu_utilization_pct=None, process_cpu_pct=None, process_rss_mib=None)
                if self.gpu_id is not None:
                    try:
                        result = subprocess.run([
                            "nvidia-smi", f"--id={self.gpu_id}",
                            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        values = result.stdout.strip().splitlines()[0].split(",")
                        for key, value in zip(list(row)[3:8], values):
                            try:
                                parsed = float(value.strip())
                                row[key] = parsed if math.isfinite(parsed) else None
                            except ValueError:
                                pass
                    except Exception as e:
                        row["gpu_error"] = str(e)
                if psutil is not None:
                    try:
                        row.update(cpu_utilization_pct=psutil.cpu_percent(None),
                                   process_cpu_pct=proc.cpu_percent(None),
                                   process_rss_mib=proc.memory_info().rss/1024**2)
                    except psutil.Error as e:
                        row["cpu_error"] = str(e)
                else:
                    row["cpu_error"] = "psutil not installed"
                self.rows.append(row)
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()
                self.stop_event.wait(self.interval)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=5)

    def summary(self):
        result = {"sample_count": len(self.rows), "scope": "whole_device",
                  "memory_peak_kind": "sampled_device_peak_including_other_processes",
                  "sampling_interval_s": self.interval}
        for key in ("gpu_utilization_pct", "gpu_memory_used_mib", "gpu_memory_total_mib",
                    "gpu_temperature_c", "gpu_power_w", "cpu_utilization_pct", "process_rss_mib"):
            values = [r[key] for r in self.rows if r[key] is not None]
            result[key + "_mean"] = sum(values)/len(values) if values else None
            result[key + "_max"] = max(values) if values else None
        result["errors"] = sorted({str(r[k]) for r in self.rows for k in ("gpu_error", "cpu_error") if k in r})
        return result


class StepProfiler:
    """Synchronized wall timing for diagnosis; boundary-only mode for throughput.

    Phase mode serializes phase boundaries deliberately. Step includes loading,
    transfer, forward+loss, cleargrad+backward, update, and unassigned Python work.
    JSON writing and metric conversion occur outside step timing. Use separate
    throughput runs to quantify observer overhead.
    """
    def __init__(self, path, synchronize=lambda: None, mode="phases", nvtx=False):
        if mode not in ("phases", "throughput"):
            raise ValueError("mode must be phases or throughput")
        self.path, self.sync, self.mode, self.nvtx = Path(path), synchronize, mode, nvtx
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.phase_times = {}

    def begin(self):
        self.sync()
        self.started = time.perf_counter()
        self.timestamp = time.time()
        self.phase_times = {}

    @contextmanager
    def phase(self, name):
        if self.mode == "phases":
            self.sync()
        start = time.perf_counter()
        if self.nvtx:
            import cupy as cp
            cp.cuda.nvtx.RangePush(name)
        try:
            yield
        finally:
            if self.mode == "phases":
                self.sync()
                self.phase_times[name + "_s"] = time.perf_counter() - start
            if self.nvtx:
                cp.cuda.nvtx.RangePop()

    def end(self, samples, epoch, step):
        self.sync()
        elapsed = time.perf_counter() - self.started
        phases = ("data", "transfer", "forward", "backward", "update")
        row = {"timestamp": self.timestamp, "epoch": epoch, "step": step,
               "samples": samples, "mode": self.mode, "step_s": elapsed,
               "samples_per_sec": samples/elapsed}
        row.update({p + "_s": self.phase_times.get(p + "_s") for p in phases})
        row["other_s"] = (max(0., elapsed-sum(self.phase_times.values()))
                          if self.mode == "phases" else None)
        self.rows.append(row)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def summary(self):
        total = sum(r["step_s"] for r in self.rows)
        result = {"steps": len(self.rows), "measured_step_total_s": total,
                  "mean_step_s": total/len(self.rows) if self.rows else None,
                  "samples_per_sec": sum(r["samples"] for r in self.rows)/total if total else None,
                  "mode": self.mode, "phase_fraction": {}}
        if total and self.mode == "phases":
            for key in ("data", "transfer", "forward", "backward", "update", "other"):
                result["phase_fraction"][key] = sum(r[key+"_s"] for r in self.rows)/total
        return result


def diagnose(timing, hardware):
    """Evidence-labelled hypotheses for subsequent controlled experiments."""
    advice = []
    fractions = timing.get("phase_fraction", {})
    if fractions:
        dominant = max(fractions, key=fractions.get)
        advice.append({"observation": f"Largest measured phase: {dominant}",
                       "fraction": fractions[dominant],
                       "next_action": {"data": "Compare workers/prefetch with fixed sample order",
                                       "transfer": "Compare device-resident batches; inspect H2D copies",
                                       "forward": "Use Nsight Systems then Compute on forward kernels",
                                       "backward": "Profile backward kernels; forward replacement alone has limited scope",
                                       "update": "Profile optimizer launch count before considering fusion",
                                       "other": "Inspect Python, callbacks and synchronization overhead"}[dominant]})
    util = hardware.get("gpu_utilization_pct_mean")
    if util is not None and util < 50:
        advice.append({"observation": "Mean sampled device utilization below 50%",
                       "next_action": "Check data waits and small kernel launches on timeline; not proof of either"})
    peak, total = hardware.get("gpu_memory_used_mib_max"), hardware.get("gpu_memory_total_mib_max")
    if peak is not None and total and peak/total > .9:
        advice.append({"observation": "Sampled device memory above 90% capacity",
                       "next_action": "Inspect im2col workspace and other processes; reduce batch and remeasure"})
    if not timing.get("steps"):
        advice.append({"observation": "No training steps", "next_action": "No bottleneck conclusion possible"})
    return advice
