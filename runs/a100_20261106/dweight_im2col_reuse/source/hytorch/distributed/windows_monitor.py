"""Dependency-free Windows counters when psutil is unavailable."""
import ctypes
import os
import time
from ctypes import wintypes


class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage")]


def ticks(value):
    return (value.dwHighDateTime << 32) | value.dwLowDateTime


class WindowsCounters:
    def __init__(self, processes):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        file_ptr = ctypes.POINTER(wintypes.FILETIME)
        self.kernel.GetSystemTimes.argtypes = [file_ptr, file_ptr, file_ptr]
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [file_ptr] * 4
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        self.processes = {name: p.pid for name, p in processes.items()}
        self.previous = {}
        self.system_previous = None
        self.sample()  # Prime delta counters; do not present the first sample as zero.

    def sample(self):
        idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
        cpu = None
        if self.kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            current = (ticks(idle), ticks(kernel) + ticks(user))
            if self.system_previous is not None:
                idle_delta = current[0] - self.system_previous[0]
                total_delta = current[1] - self.system_previous[1]
                if total_delta > 0:
                    cpu = min(100.0, max(0.0, 100 * (1 - idle_delta / total_delta)))
            self.system_previous = current
        result = {}
        for name, pid in self.processes.items():
            handle = self.kernel.OpenProcess(0x0400 | 0x0010, False, pid)
            if not handle:
                continue
            try:
                creation, exit_time, kernel, user = (wintypes.FILETIME() for _ in range(4))
                memory = MemoryCounters()
                memory.cb = ctypes.sizeof(memory)
                cpu_pct = None
                if self.kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                               ctypes.byref(kernel), ctypes.byref(user)):
                    now, used = time.perf_counter(), (ticks(kernel) + ticks(user)) / 1e7
                    if pid in self.previous:
                        last_time, last_used = self.previous[pid]
                        cpu_pct = max(0.0, 100 * (used - last_used) / max(now - last_time, 1e-9))
                    self.previous[pid] = (now, used)
                rss = None
                if self.psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
                    rss = memory.WorkingSetSize / 1024**2
                result[name] = {"pid": pid, "cpu_pct": cpu_pct, "rss_mb": rss}
            finally:
                self.kernel.CloseHandle(handle)
        return cpu, result
