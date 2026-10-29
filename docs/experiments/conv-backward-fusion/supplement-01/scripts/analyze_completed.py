"""Reuse 4D analysis for completed measurements and a later matched trace set."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'docs/experiments/conv-backward-fusion/scripts'))
from analyze_results import training,micro,traces

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--stage',choices=['measurements','profiling'],required=True)
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--trace-raw',type=Path)
p.add_argument('--analysis',type=Path)
a=p.parse_args()
if a.stage=='measurements':
    a.output.mkdir(parents=True,exist_ok=False)
    training(a.raw,a.output);micro(a.raw,a.output)
    (a.output/'measurements-completion.json').write_text(json.dumps(dict(status='PASSED',
        raw=str(a.raw.resolve()),scope='36 independent training runs and 9 microbenchmark runs'),indent=2)+'\n')
else:
    assert json.loads((a.output/'measurements-completion.json').read_text())['status']=='PASSED'
    assert json.loads((a.trace_raw/'completion.json').read_text())['status']=='COMPLETED'
    traces(a.trace_raw,a.analysis,a.output)
    (a.output/'completion.json').write_text(json.dumps(dict(status='PASSED',
        raw=str(a.raw.resolve()),trace_raw=str(a.trace_raw.resolve()),
        measurement_gpu=json.loads((a.raw/'training-throughput-F-r1/config.json').read_text())['monitor_id'],
        profiling_gpu=json.loads((a.trace_raw/'trace-training-F/config.json').read_text())['monitor_id'],
        timing='Uninstrumented measurements and Nsight are separate windows; comparisons stay within each matched window'),indent=2)+'\n')
print(a.stage+' PASSED')
