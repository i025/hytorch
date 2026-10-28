"""Process-local event files and device-wide hardware sampling."""
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path


def power_status():
    """Record Windows AC/battery state without changing power settings."""
    if os.name != "nt":
        return {}
    import ctypes
    class Status(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ubyte) for name in
                    ("ac", "flags", "percent", "saver")] + [
                    ("life", ctypes.c_uint32), ("full_life", ctypes.c_uint32)]
    value = Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(value)):
        return {"ac_online": None, "power_status_error": "GetSystemPowerStatus failed"}
    return {"ac_online": bool(value.ac) if value.ac in (0, 1) else None,
            "battery_percent": value.percent if value.percent <= 100 else None,
            "battery_saver": bool(value.saver)}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class EventWriter:
    def __init__(self, path, run_id, role, rank=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id, self.role, self.rank = run_id, role, rank
        self.started = time.perf_counter()

    def log(self, record_type, **values):
        record = {"schema_version": 1, "run_id": self.run_id, "time": time.time(),
                  "elapsed": time.perf_counter() - self.started, "type": record_type,
                  "role": self.role, "rank": self.rank, "pid": os.getpid()}
        record.update(values)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


class HardwareMonitor:
    def __init__(self, config, run_dir, processes):
        self.config, self.processes = config, processes
        self.writer = EventWriter(Path(run_dir) / "hardware.jsonl", Path(run_dir).name, "monitor")
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.samples = []

    def start(self):
        self.thread.start()

    def _run(self):
        native = None
        try:
            import psutil
            psutil.cpu_percent(None)
            tracked = {name: psutil.Process(p.pid) for name, p in self.processes.items()}
            for proc in tracked.values():
                proc.cpu_percent(None)
        except Exception:
            psutil, tracked = None, {}
            if os.name == "nt":
                from .windows_monitor import WindowsCounters
                native = WindowsCounters(self.processes)
        while not self.stop_event.is_set():
            row = {"scope": "whole_device", "gpu_id": self.config.gpu_id,
                   "gpu_utilization_pct": None, "gpu_memory_used_mb": None,
                   "gpu_temperature_c": None, "gpu_power_w": None,
                   "cpu_utilization_pct": None, "processes": {}}
            row.update(power_status())
            try:
                command = ["nvidia-smi", f"--id={self.config.gpu_id}",
                           "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,pstate,clocks.gr,clocks.mem,power.limit",
                           "--format=csv,noheader,nounits"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=3,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if result.returncode:
                    raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
                values = result.stdout.strip().splitlines()[0].split(",")
                row["gpu_pstate"] = values[4].strip()
                for name, value in zip(("gpu_utilization_pct", "gpu_memory_used_mb",
                                        "gpu_temperature_c", "gpu_power_w", "gpu_graphics_clock_mhz",
                                        "gpu_memory_clock_mhz", "gpu_power_limit_w"), values[:4] + values[5:]):
                    row[name] = None
                    try:
                        row[name] = float(value.strip())
                    except ValueError:
                        pass
            except Exception as exc:
                row["gpu_monitor_error"] = str(exc)
            if psutil is not None:
                row["cpu_utilization_pct"] = psutil.cpu_percent(None)
                for name, proc in tracked.items():
                    try:
                        row["processes"][name] = {"pid": proc.pid, "cpu_pct": proc.cpu_percent(None),
                                                  "rss_mb": proc.memory_info().rss / 1024**2}
                    except psutil.Error:
                        pass
            elif native is not None:
                row["cpu_utilization_pct"], row["processes"] = native.sample()
            row["cpu_monitor_backend"] = "psutil" if psutil is not None else "windows_api" if native else None
            self.writer.log("hardware", **row)
            self.samples.append(row)
            self.stop_event.wait(self.config.monitor_interval_s)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        result = {"scope": "whole_device", "sample_count": len(self.samples),
                  "memory_peak_kind": "sampled_device_peak_including_other_processes"}
        for name in ("gpu_utilization_pct", "gpu_memory_used_mb", "gpu_temperature_c",
                     "gpu_power_w", "cpu_utilization_pct"):
            values = [s[name] for s in self.samples if s[name] is not None]
            result[name + "_mean"] = sum(values) / len(values) if values else None
            result[name + "_max"] = max(values) if values else None
        return result


def environment_info(config):
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in ("numpy", "cupy", "cupy-cuda11x", "cupy-cuda12x", "grpcio", "protobuf", "Pillow", "psutil"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "device": config.device, "gpu_id": config.gpu_id, "packages": packages,
            "cupy_accelerators": os.environ.get("CUPY_ACCELERATORS"),
            "cupy_accelerators_scope": "environment_inherited_by_training_processes"}
