"""Check recorded worst elements against ascending-K IEEE FP32 FMA accumulation."""
import ctypes
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[4]
source=root/'runs/gemm_diagnosis_01/fp64-dot-samples.json'
rows=json.loads(source.read_text());fma=ctypes.CDLL('libm.so.6').fmaf
fma.argtypes=[ctypes.c_float,ctypes.c_float,ctypes.c_float];fma.restype=ctypes.c_float
for n in sorted({r['n'] for r in rows}):
 rng=np.random.default_rng(42+n);a=rng.standard_normal((n,n),dtype=np.float32);b=rng.standard_normal((n,n),dtype=np.float32)
 for row in [r for r in rows if r['n']==n]:
  acc=0.
  for x,y in zip(a[row['row']],b[:,row['col']]): acc=fma(float(x),float(y),acc)
  row.update(sequential_fp32_fma=acc,exact_match_gpu_fma=acc==row['gpu_fp32'])
out=root/'docs/experiments/a100-gpu-window/analysis/gemm-diagnosis-01/fma-reproduction.json'
with out.open('x') as f: json.dump(rows,f,indent=2)
print({n:dict(samples=sum(r['n']==n for r in rows),matched=sum(r['n']==n and r['exact_match_gpu_fma'] for r in rows)) for n in (1024,2048,4096)})
