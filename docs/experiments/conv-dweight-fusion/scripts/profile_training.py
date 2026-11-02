"""Reuse verified convolution component NVTX; mark the native dWeight ABI."""
from pathlib import Path
import importlib.util
import sys

ROOT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('stage4f_profile',ROOT/'docs/experiments/conv-dinput-optimization/scripts/profile_training.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)


class WeightRanges(previous.KernelRanges):
    def __enter__(self):
        super().__enter__()
        from hytorch.backends.cuda import runtime
        original=runtime.conv2d_dweight
        def call(*args,**kwargs):
            with self.range('OP/internal:native.conv_dweight'):
                return original(*args,**kwargs)
        self.patch(runtime,'conv2d_dweight',call)
        return self


if __name__=='__main__':
    previous.previous.OperatorRanges=WeightRanges
    previous.previous.main()
