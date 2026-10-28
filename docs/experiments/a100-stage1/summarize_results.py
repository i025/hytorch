"""Check and visualize actual stage 1 records; no training or GPU computation."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT/name).read_text())

def lines(name):
    return [json.loads(line) for line in (ROOT/name).read_text().splitlines()]

def save(name, value):
    with (ROOT/name).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)

def counts(name):
    suite = ET.parse(ROOT/name).getroot().find('testsuite')
    result = {k: int(suite.attrib[k]) for k in ('tests','failures','errors','skipped')}
    result['passed'] = result['tests']-result['failures']-result['errors']-result['skipped']
    result['elapsed_s'] = float(suite.attrib['time'])
    return result

host = counts('host-tests/tests-02.xml')
gpu = counts('validation/tests.xml')
validation = read('validation/environment.json')
smoke = read('smoke/summary.json')
config = read('smoke/config.json')
steps, losses, hardware = [lines('smoke/'+f+'.jsonl') for f in ('steps','losses','hardware')]
assert validation['status']=='PASSED' and validation['gpu_status']=='EXECUTED'
assert gpu['failures']==gpu['errors']==gpu['skipped']==0 and gpu['passed']==37
assert smoke['status']=='COMPLETED' and smoke['gpu_status']=='EXECUTED'
assert len(steps)==len(losses)==smoke['timing']['steps']==4
assert all(np.isfinite(r['loss']) for r in losses)
assert all(smoke['dispatch_counts'][name]>0 for name in ('gemm_shared','conv_im2col','pool_native'))
assert config['no_fingerprints'] and config['initial_state_sha256'] is None and config['dataset_sha256'] is None
assert smoke['final_state_sha256'] is None
assert all((a['epoch'],a['step'],a['timestamp'])==(b['epoch'],b['step'],b['timestamp']) for a,b in zip(steps,losses))
weighted_loss=sum(r['loss']*r['samples'] for r in losses)/sum(r['samples'] for r in losses)
assert np.isclose(weighted_loss,smoke['epochs'][0]['train_loss'])
with np.load(ROOT.parents[2]/'runs/gpu_smoke_01/final_weights.npz') as weights:
    finite_weights=all(np.isfinite(weights[key]).all() for key in weights.files)
assert finite_weights
save('smoke/checks.json',dict(status='PASSED',steps=len(steps),finite_step_losses=True,
     finite_final_weights=True,weighted_loss=weighted_loss,hashes_disabled=True,
     dispatch_counts=smoke['dispatch_counts'],hardware_samples=len(hardware)))
save('host-tests/summary-02.json',host)
save('validation/summary.json',gpu)

comparisons=lines('validation/numeric-comparisons-01.jsonl')
groups=defaultdict(list)
for row in comparisons:
    assert row['finite']
    groups[row['test'].split('::')[-1].split('[')[0]].append(row)
metrics=[]
for name, rows in groups.items():
    metrics.append(dict(test_group=name,comparisons=len(rows),
        max_abs_error=max(r['max_abs_error'] for r in rows),
        max_relative_l2_error=max(r['relative_l2_error'] for r in rows)))
with (ROOT/'validation/numeric-summary.csv').open('x') as out:
    writer=csv.DictWriter(out,fieldnames=list(metrics[0]))
    writer.writeheader(); writer.writerows(metrics)

fig, ax = plt.subplots(figsize=(8,4.5))
x=np.arange(2)
for offset,key,color in [(-.25,'passed','#287d59'),(0,'failures','#c44b4b'),(.25,'skipped','#cb982a')]:
    bars=ax.bar(x+offset,[host[key],gpu[key]],width=.25,label=key.capitalize(),color=color)
    ax.bar_label(bars,padding=3)
ax.set_xticks(x,['Host regression (final)','GPU correctness'])
ax.set_ylim(0,44); ax.set_ylabel('Test cases'); ax.legend(loc='upper left')
ax.set_title('Stage 1 acceptance: actual JUnit results')
fig.text(.5,.02,'Log visualization; 2 optional PyTorch host tests skipped',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(ROOT/'figures/acceptance-results.png',dpi=160); plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(10,4))
axes[0].plot([r['step'] for r in losses],[r['loss'] for r in losses],marker='o')
axes[0].set(xlabel='Training step',ylabel='Batch MSE loss',title='TinyNet: 8 synthetic samples')
axes[1].bar([r['step'] for r in steps],[r['step_s']*1000 for r in steps],color='#376f9e')
axes[1].set(xlabel='Training step',ylabel='Step time (ms)',title='Synchronized phase timing')
for ax in axes: ax.set_xticks([1,2,3,4])
fig.text(.5,.01,'Log visualization; different batches, not evidence of convergence or speedup',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(ROOT/'figures/tinynet-loss-and-time.png',dpi=160); plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(11,4))
start=steps[0]['timestamp']
for ax,key,label in zip(axes,['gpu_utilization_pct','gpu_memory_used_mib','gpu_power_w'],['GPU utilization (%)','Device memory (MiB)','Power (W)']):
    observed=[r for r in hardware if r[key] is not None]
    ax.scatter([r['timestamp']-start for r in observed],[r[key] for r in observed],s=55)
    for r in observed: ax.annotate(str(r[key]),(r['timestamp']-start,r[key]),xytext=(7,7),textcoords='offset points')
    ax.set(xlabel='Seconds since first step',ylabel=label)
fig.suptitle('GPU 1: only one hardware sample; no time-series trend available')
fig.text(.5,.01,'Log visualization; whole-device sampling, interval 1 s; not a continuous GPU activity trace',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.05,1,.94]); fig.savefig(ROOT/'figures/gpu-monitor-sample.png',dpi=160); plt.close(fig)
print(json.dumps(dict(host=host,gpu=gpu,comparisons=len(comparisons),numeric_groups=metrics,
                     smoke=read('smoke/checks.json')),indent=2))
