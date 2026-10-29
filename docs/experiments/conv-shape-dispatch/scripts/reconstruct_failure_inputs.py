"""Offline reconstruction of the first failing SpatialNet batch, not a capture."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from benchmarks.bench_cuda_training import SpatialNet,initialize

out=ROOT/'runs/conv_shape_spatial_failure_inputs_01';out.mkdir(exist_ok=False)
state=initialize(SpatialNet(),(3,16,16),42)
rng=np.random.default_rng(42)
x=rng.normal(size=(4,3,16,16)).astype(np.float32)*.1
target=rng.normal(size=(4,1)).astype(np.float32)*.1
np.savez(out/'reconstructed-first-batch.npz',input=x,target=target,**state)
(out/'provenance.json').write_text(json.dumps(dict(
    provenance='Reconstructed on CPU from unchanged initialization and RNG seed; NOT captured at failure',
    seed=42,step=0,initialization='SpatialNet parameter shapes/initialization unchanged by the FP32 mean fix',
    original_error='GPU gy/parameter gradients promoted to FP64 by F.average; native dInput or fused SGD rejected dtype',
    original_log='runs/conv_shape_measurements_wait_01/attempt-01/results/gpu-tests.log',
    failed_forward='relu(conv1) -> relu(conv2) -> relu(conv3) -> F.average(axis=(2,3)) -> fc1 -> tanh',
    gpu_replay_performed=False),indent=2)+'\n')
print(str(out))
