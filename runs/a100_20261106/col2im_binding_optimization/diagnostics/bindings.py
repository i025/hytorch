"""Experiment-only baseline loader and unchecked diagnostic; never imported by production."""
import importlib.util,contextlib,struct
from pathlib import Path
from common import *
ROOT=Path.cwd();O=ROOT/'runs/a100_20261106/col2im_binding_optimization'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module('old_col2im_runtime',O/'baseline/hytorch/backends/cuda/runtime.py')
cache=module('cache_only_runtime',O/'raw/cache_only_runtime.py')
def prepare():
 # All modes use exactly the same compiled CUDA library/kernel in the same process.
 old._library=cache._library=rt.load_library()
def launch(mode,col,shape,stride,pad,out):
 if mode=='old':return old.col2im(col,shape,stride,pad,out)
 if mode=='cache':return cache.col2im(col,shape,stride,pad,out)
 if mode=='fast':return rt.col2im(col,shape,stride,pad,out)
 return run('cupy',col,shape,stride,pad,out)
@contextlib.contextmanager
def mode_context(mode):
 original=rt.col2im
 if mode=='old':rt.col2im=old.col2im
 try:
  with rt.backend(matmul='shared',conv='im2col',pool='native',col2im='cupy' if mode=='cupy' else 'native'):yield
 finally:rt.col2im=original
