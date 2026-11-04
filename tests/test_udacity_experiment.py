import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT=Path(__file__).resolve().parents[1]


def load(name):
    path=ROOT/'docs/experiments/udacity-real-training/scripts'/name
    spec=importlib.util.spec_from_file_location('udacity_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_fixed_sequential_plan(tmp_path):
    module=load('run_window.py')
    plan=module.jobs(SimpleNamespace(prepared=tmp_path/'prepared',output=tmp_path/'output'))
    assert len(plan)==47 and len({j['name'] for j in plan})==47
    assert [j['name'] for j in plan[-2:]]==['trace-F','trace-I']
    for task in ('compute','io','phases'):
        for repeat in range(1,4):
            jobs=[j for j in plan if j['name'].startswith(task+'-') and j['name'].endswith('-r'+str(repeat))]
            assert {j['method'] for j in jobs}==set('AFIE')
            assert all('--no-fingerprints' in j['command'] for j in jobs)


def test_explicit_backend_options():
    import sys
    from unittest.mock import patch
    with patch.object(sys,'path',[str(ROOT/'docs/experiments/udacity-real-training/scripts'),*sys.path]):
        engine=load('engine.py')
    assert engine.CONFIGS['A']=={}
    assert engine.CONFIGS['I']==dict(engine.CONFIGS['F'],conv_dweight='fused')
    assert engine.CONFIGS['F']['conv_backward']=='fused'
