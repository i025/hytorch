"""CPU environment record without initializing GPU contexts or computing hashes."""
import ctypes
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
print('UTC:',subprocess.check_output(['date','-u','--iso-8601=seconds'],text=True).strip())
print('base_commit:',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
print('Python:',sys.version);print('OS:',platform.platform())
print(Path('/etc/os-release').read_text())
print(subprocess.check_output(['lscpu'],text=True))
for name in ('numpy','cupy-cuda12x','torch','torchvision','pytest'):
 print(name,importlib.metadata.version(name))
for key in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','CUPY_TF32','NVIDIA_TF32_OVERRIDE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):
 print(key,os.environ.get(key))
np.show_config()
path=next(Path(np.__file__).parent.parent.glob('numpy.libs/libopenblas*.so'))
lib=ctypes.CDLL(str(path));print('NumPy BLAS file:',path)
for name in ('openblas_get_config64_','openblas_get_corename64_'):
 fn=getattr(lib,name);fn.restype=ctypes.c_char_p;print(name,fn().decode())
print(subprocess.check_output(['nvcc','--version'],text=True))
print(subprocess.check_output(['gcc','--version'],text=True))
