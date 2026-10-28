"""Summarize actual host parity assertions; no synthetic values."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
rows=[json.loads(line) for line in (root/'pytorch-parity/numeric-comparisons-02.jsonl').read_text().splitlines()]
summary=[]
for model in ('tiny','donkey'):
    for phase in ('initial','forward','loss','gradient','updated'):
        selected=[r for r in rows if f'[{model}-' in r['test'] and r['comparison'].split(':')[0]==phase]
        summary.append(dict(model=model,comparison=phase,assertions=len(selected),
            max_abs_error=max(r['max_abs_error'] for r in selected),
            max_relative_l2_error=max(r['relative_l2_error'] for r in selected),
            atol=selected[0].get('atol',0),rtol=selected[0].get('rtol',0)))
with (root/'pytorch-parity/summary-after-fix.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
(root/'figures').mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,key in zip(axes,('max_abs_error','max_relative_l2_error')):
    for model in ('tiny','donkey'):
        chosen=[r for r in summary if r['model']==model]
        ax.plot([r['comparison'] for r in chosen],[r[key] for r in chosen],'o-',label=model)
    ax.set_yscale('symlog',linthresh=1e-10);ax.set_ylabel(key);ax.legend();ax.grid(alpha=.2)
fig.suptitle('Measured CPU PyTorch parity: after dtype fix: full suite 25 passed / targeted parity 2 passed')
fig.tight_layout();fig.savefig(root/'figures/host-parity-after-fix.png',dpi=160)
print(json.dumps(summary,indent=2))
