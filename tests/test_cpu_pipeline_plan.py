"""The future GPU runner's data policy can be verified without CUDA."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_config', ROOT/'docs/experiments/cpu-training-pipeline/scripts/training_config.py')
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)


@pytest.mark.parametrize('method,engine,workers', [
    ('A','A',0), ('I','I',0), ('P','P',2), ('R','R',0),
    ('J','J',2), ('E','E',0), ('E-prefetch','E',2)])
def test_shared_training_configuration(method, engine, workers):
    protocol = dict(throughput=[7,2,7], preprocessing={'color':'RGB'})
    config = config_module.configuration(method, 'spatial', protocol)
    assert config['engine']==engine and config['workers']==workers
    assert config['indices']==[7,2,7]
    assert config['batch']==4 and config['lr']==.001 and config['dtype']=='float32'
    assert config['dropout']==0 and config['tf32'] is False
    assert config['no_fingerprints'] is True
    assert config['open_stream'] is True
