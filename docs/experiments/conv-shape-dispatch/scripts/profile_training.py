"""Reuse 4D ranges and record actual metadata-only auto choices."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('stage4d_ranges',ROOT/'docs/experiments/conv-backward-fusion/scripts/profile_training.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)


class DispatchRanges(previous.FusionRanges):
    def __enter__(self):
        super().__enter__()
        self.decisions=[]
        from hytorch.backends.cuda import runtime
        original=runtime.conv_dinput_dispatch

        def call(x,w,gy,stride=1,pad=0,**kwargs):
            with self.range('OP/dispatch:conv_dinput'):
                mode,reason=original(x,w,gy,stride,pad,**kwargs)
            self.decisions.append(dict(phase=self.phase,input_shape=x.shape,weight_shape=w.shape,
                stride=stride,pad=pad,dtype=str(x.dtype),contiguous=all(v.flags.c_contiguous for v in (x,w,gy)),
                selected=mode,reason=reason))
            return mode,reason
        self.patch(runtime,'conv_dinput_dispatch',call)
        return self

    def summary(self):
        result=super().summary();result['dispatch_decisions']=self.decisions
        return result


if __name__=='__main__':
    previous.previous.OperatorRanges=DispatchRanges
    previous.previous.main()
