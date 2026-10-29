"""Explicit arithmetic semantics for sampled GEMM diagnosis; no production changes."""
import ctypes
import math
import numpy as np

_libm=ctypes.CDLL('libm.so.6')
_fmaf=_libm.fmaf
_fmaf.argtypes=[ctypes.c_float,ctypes.c_float,ctypes.c_float]
_fmaf.restype=ctypes.c_float


def sequential_fma(a,b):
    acc=0.0
    for x,y in zip(a,b):
        acc=_fmaf(float(x),float(y),acc)
    return np.float32(acc)


def pairwise_sum(values):
    values=np.asarray(values,dtype=np.float32).copy()
    while len(values)>1:
        pairs=len(values)//2
        partial=np.add(values[:2*pairs:2],values[1:2*pairs:2],dtype=np.float32)
        values=np.concatenate((partial,values[-1:])) if len(values)%2 else partial
    return values[0]


def pairwise_product(a,b):
    return pairwise_sum(np.multiply(a,b,dtype=np.float32))


def blocked_fma(a,b,block):
    partial=np.array([sequential_fma(a[i:i+block],b[i:i+block]) for i in range(0,len(a),block)],dtype=np.float32)
    return pairwise_sum(partial)


def sequential_fp64(a,b):
    # Binary32 inputs convert exactly; their products fit in binary64's significand.
    acc=0.0
    for x,y in zip(a,b):
        acc+=float(x)*float(y)
    return acc


def ulp_distance(a,b):
    def ordered(value):
        bits=int(np.float32(value).view(np.uint32))
        return 0x80000000-(bits&0x7fffffff) if bits&0x80000000 else 0x80000000+bits
    return abs(ordered(a)-ordered(b))


def metrics(actual,reference,numpy32):
    value=float(actual);absolute=abs(value-reference)
    return dict(value=value,abs_error_fp64=absolute,
        relative_error_fp64=absolute/abs(reference) if reference else None,
        ulp_vs_rounded_fp64=ulp_distance(actual,np.float32(reference)),
        abs_error_numpy32=abs(value-float(numpy32)),
        pass_original_numpy=abs(value-float(numpy32))<=1e-4+1e-4*abs(float(numpy32)),
        pass_same_tolerance_fp64=absolute<=1e-4+1e-4*abs(reference))


def summarize(rows):
    groups=[]
    for n in sorted({r['n'] for r in rows}):
        for group in ('all','historical_worst','near_zero','random_normal'):
            selected=[r for r in rows if r['n']==n and (group=='all' or r['stratum']==group)]
            for method in sorted({r['strategy'] for r in selected}):
                values=[r for r in selected if r['strategy']==method]
                error=np.array([r['abs_error_fp64'] for r in values],np.float64)
                groups.append(dict(n=n,stratum=group,strategy=method,samples=len(values),
                    mean_abs_error=float(np.mean(error)),median_abs_error=float(np.median(error)),
                    p95_abs_error=float(np.percentile(error,95)),max_abs_error=float(np.max(error)),
                    rms_error=float(np.sqrt(np.mean(error**2))),
                    max_ulp=max(r['ulp_vs_rounded_fp64'] for r in values),
                    fail_original_numpy=sum(not r['pass_original_numpy'] for r in values),
                    fail_same_tolerance_fp64=sum(not r['pass_same_tolerance_fp64'] for r in values),
                    finite_relative_count=sum(r['relative_error_fp64'] is not None for r in values),
                    median_relative_error=float(np.median([r['relative_error_fp64'] for r in values if r['relative_error_fp64'] is not None])) ))
    return groups
