"""Capture the selected physical/visible device mapping before acceptance."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import cupy as cp
from hytorch.backends.cuda.runtime import load_library, library_path

assert cp.cuda.runtime.getDeviceCount() == 1
cp.cuda.Device(0).use()
p = cp.cuda.runtime.getDeviceProperties(0)
lib = load_library()
print(json.dumps(dict(physical_gpu=1, visible_device_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
    logical_gpu=0, visible_device_count=cp.cuda.runtime.getDeviceCount(),
    name=p['name'].decode(), total_memory_bytes=p['totalGlobalMem'],
    compute_capability=cp.cuda.Device(0).compute_capability,
    pci_bus_id=cp.cuda.Device(0).pci_bus_id, runtime=cp.cuda.runtime.runtimeGetVersion(),
    driver=cp.cuda.runtime.driverGetVersion(), cupy=cp.__version__,
    library=str(library_path()), library_abi=lib.hy_cuda_abi_version()), indent=2))
cp.show_config()
