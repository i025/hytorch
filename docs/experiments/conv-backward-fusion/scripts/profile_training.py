"""Extend stage 4A attribution for the single new runtime dInput call."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/experiments/donkeynet-backward-profiling/scripts'))
import profile_training as previous


class FusionRanges(previous.OperatorRanges):
    def __enter__(self):
        super().__enter__()
        from hytorch.backends.cuda import runtime
        original=runtime.conv2d_dinput

        def call(*args,**kwargs):
            parent=self.nodes[-1]
            label='OP/part:Conv2d.dInput|layer='+parent[3]
            label+='|leaf_input='+str(parent[2].inputs[0].creator is None).lower()
            with self.range(label),self.range('OP/internal:native.conv_dinput'):
                return original(*args,**kwargs)
        self.patch(runtime,'conv2d_dinput',call)
        return self


if __name__=='__main__':
    previous.OperatorRanges=FusionRanges
    previous.main()
