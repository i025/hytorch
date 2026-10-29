"""Read-only stage 4C evidence and shape-derived work estimates, not 4D results."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/experiments/fused-sgd-cuda/scripts'))
from analyze_fused_results import csv_rows,write_csv
from bench_backward import SHAPES

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
prior=ROOT/'docs/experiments/conv-dinput-pruning/results'
components=csv_rows(prior/'gradient-components.csv')
layouts=csv_rows(prior/'layout-kernels.csv')
rows=[]
for layer,(shape,weight,stride) in SHAPES.items():
    n,c,h,w=shape;oc,_,kh,kw=weight
    oh,ow=(h-kh)//stride+1,(w-kw)//stride+1
    values={r['part']:r for r in components if r['method']=='D' and r['layer']==layer}
    row=dict(source='stage4C D trace; historical, not stage4D',layer=layer,
        input_shape=str(shape),weight_shape=str(weight),stride=stride,pad=0,
        dinput_gpu_us=float(values['dInput']['gpu_kernel_ms'])*1000/16,
        dweight_gpu_us=float(values['dWeight']['gpu_kernel_ms'])*1000/16,
        dinput_kernels_per_step=int(values['dInput']['gpu_kernel_count'])/16,
        dweight_kernels_per_step=int(values['dWeight']['gpu_kernel_count'])/16,
        gcol_bytes=4*n*c*kh*kw*oh*ow,
        valid_fma_count=n*c*kh*kw*oh*ow*oc,
        fused_output_bytes=4*n*c*h*w,
        naive_operand_read_bytes=8*n*c*kh*kw*oh*ow*oc)
    for category in ['copy','col2im','tensordot']:
        rr=[r for r in layouts if r['method']=='D' and r['layer']==layer and r['part']=='dInput' and r['category']==category]
        row[category+'_gpu_us']=sum(float(r['gpu_ms']) for r in rr)*1000/16
    rows.append(row)
write_csv(a.output/'historical-costs.csv',rows)
summary=dict(scope='Historical stage4C D; shape estimates are not new GPU measurements',
    dinput_us=sum(r['dinput_gpu_us'] for r in rows),dweight_us=sum(r['dweight_gpu_us'] for r in rows),
    col2im_us=sum(r['col2im_gpu_us'] for r in rows),copy_us=sum(r['copy_gpu_us'] for r in rows),
    gcol_bytes_per_step=sum(r['gcol_bytes'] for r in rows),
    potential_launches_removed=8,warning='Copy is included in tensordot; do not add both categories')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4))
for offset,field,label,color in [(-.18,'dinput_gpu_us','dInput','#4477aa'),(.18,'dweight_gpu_us','dWeight','#dd9933')]:
    axes[0].bar([i+offset for i in range(4)],[r[field] for r in rows],width=.36,label=label,color=color)
axes[0].set_xticks(range(4),SHAPES);axes[0].set_ylabel('Historical GPU kernel sums (us / step)');axes[0].legend()
axes[1].bar(list(SHAPES),[r['gcol_bytes']/2**20 for r in rows],color='#4477aa')
axes[1].set_ylabel('gcol size from tensor shapes (MiB)')
fig.suptitle('Design evidence: stage 4C trace and shape-derived storage')
fig.text(.01,.01,'Historical-data visualization; not a stage 4D speed result or screenshot.',fontsize=9)
fig.tight_layout(rect=(0,.06,1,1));fig.savefig(a.output/'historical-design-evidence.png',dpi=170);plt.close(fig)
print(json.dumps(summary,indent=2))
