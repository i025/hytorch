"""Reuse phase/operator tracing; distinguish existing and four-chain gathers."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/experiments/donkeynet-backward-profiling/scripts'))
import profile_training as previous


class KernelRanges(previous.OperatorRanges):
    def __enter__(self):
        super().__enter__()
        from hytorch.backends.cuda import runtime
        original=runtime.conv2d_dinput
        def call(*args,**kwargs):
            parent=self.nodes[-1];variant=kwargs.get('variant','fused')
            with self.range('OP/part:Conv2d.dInput|layer='+parent[3]),self.range('OP/internal:native.conv_dinput:'+variant):
                return original(*args,**kwargs)
        self.patch(runtime,'conv2d_dinput',call)
        return self


if __name__=='__main__':
    previous.OperatorRanges=KernelRanges
    previous.main()
