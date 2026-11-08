"""Observe real captured CUDA graph kernels, not a copy of the dispatch predicate."""
import os,json
from pathlib import Path
import numpy as np
import pytest
from hytorch.backends.cuda import runtime as rt

# M,N,K, expected kernel. Explicit cases straddle dimension and CTA thresholds.
CASES=[(63,8192,65,'shared'),(64,8192,63,'shared'),
       (64,8192,64,'register'),(65,4096,65,'register'),
       (8192,63,65,'shared'),(8192,64,64,'register'),
       (64,8128,65,'shared'),(64,8192,65,'register'),(64,8256,65,'register'),
       (512,1024,63,'shared'),(512,1024,64,'register'),(512,1024,65,'register'),
       (511,1023,67,'register'),(449,897,71,'shared'),
       (127,4033,129,'register'),(97,193,113,'shared'),
       (240,64,432,'shared'),(72,64,576,'shared'),
       (1024,1024,257,'register'),(65,65,65,'shared'),(4,100,1152,'shared')]
@pytest.fixture(scope='module')
def gpu():
    try:
        import cupy as cp
        cp.cuda.Device(0).use();cp.zeros(1)
    except Exception as exc:
        if os.environ.get('HYTORCH_REQUIRE_CUDA')=='1':pytest.fail(str(exc))
        pytest.skip('CUDA unavailable')
    lib=rt.load_library();return cp,lib

@pytest.mark.parametrize('tile',[16,32])
@pytest.mark.parametrize('m,n,k,expected',CASES)
def test_observed_dispatch_and_numerics(gpu,m,n,k,expected,tile):
    cp,lib=gpu;rt._library=lib
    rng=np.random.default_rng(701+m+n+k)
    a=rng.normal(0,.125,(m,k)).astype(np.float32);b=rng.normal(0,.125,(k,n)).astype(np.float32)
    da,db=cp.asarray(a),cp.asarray(b);out=cp.empty((m,n),dtype=cp.float32)
    cp.cuda.get_current_stream().synchronize()
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        rt.gemm(da,db,'shared',tile,out=out);stream.synchronize()
        stream.begin_capture()
        for _ in range(3):assert rt.gemm(da,db,'shared',tile,out=out) is out
        graph=stream.end_capture();dot=graph.debug_dot_str(flags=1)
        assert 'gemm_'+expected in dot,dot
        assert ('gemm_shared' if expected=='register' else 'gemm_register') not in dot,dot
        graph.launch();stream.synchronize()
    actual=cp.asnumpy(out);ref=a.astype(np.float64)@b.astype(np.float64)
    np.testing.assert_allclose(actual,ref,atol=5e-5,rtol=2e-5)
    path=os.environ.get('HYTORCH_DISPATCH_EVIDENCE')
    if path:
        p=Path(path);p.mkdir(parents=True,exist_ok=True)
        (p/f'{m}_{n}_{k}_t{tile}.dot').write_text(dot)
        (p/f'{m}_{n}_{k}_t{tile}.json').write_text(json.dumps(dict(M=m,N=n,K=k,tile=tile,expected=expected,actual_graph_kernel='gemm_'+expected,fp64_max_abs=float(np.max(np.abs(actual-ref))),fp64_relative_l2=float(np.linalg.norm(actual-ref)/np.linalg.norm(ref))),indent=2))

@pytest.mark.parametrize('variant',['basic','blocked'])
@pytest.mark.parametrize('tile',[16,32])
def test_other_variants_not_redirected(gpu,variant,tile):
    cp,lib=gpu;rt._library=lib
    a=cp.ones((64,64),cp.float32);b=cp.ones((64,8192),cp.float32);out=cp.empty((64,8192),cp.float32)
    cp.cuda.get_current_stream().synchronize();stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture();rt.gemm(a,b,variant,tile,out=out);g=stream.end_capture()
        assert 'gemm_'+variant in g.debug_dot_str(flags=1)
        g.launch();stream.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(out),np.full((64,8192),64,np.float32))
