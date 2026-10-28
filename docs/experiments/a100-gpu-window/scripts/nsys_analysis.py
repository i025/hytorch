"""Export measured CUDA/NVTX intervals; interval coverage is not SM utilization."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
c=sqlite3.connect(a.input)
ranges=c.execute('select start,end,text from NVTX_EVENTS where end is not null order by start').fetchall()
lo=min(r[0] for r in ranges);hi=max(r[1] for r in ranges)
kernels=c.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()
copies=c.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY order by start').fetchall()
intervals=sorted((max(start,lo),min(end,hi)) for start,end in kernels+copies if end>lo and start<hi)
merged=[]
for start,end in intervals:
 if merged and start<=merged[-1][1]: merged[-1][1]=max(end,merged[-1][1])
 else: merged.append([start,end])
summary=dict(training_nvtx_envelope_ms=(hi-lo)/1e6,activity_covered_ms=sum(end-start for start,end in merged)/1e6,
    activity_coverage_fraction=sum(end-start for start,end in merged)/(hi-lo),
    coverage_definition='Union of measured kernel and memcpy intervals clipped to first-to-last training NVTX envelope; NOT SM utilization, includes profiling overhead',
    whole_trace_kernel_count=len(kernels),training_envelope_kernel_count=sum(start<hi and end>lo for start,end in kernels),
    whole_trace_kernel_duration_sum_ms=sum(end-start for start,end in kernels)/1e6)
(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
with (a.output/'intervals.csv').open('x',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['kind','relative_start_ms','duration_ms','name'])
 for start,end,label in ranges: w.writerow(['NVTX',(start-lo)/1e6,(end-start)/1e6,label])
 for kind,items in [('kernel',kernels),('memcpy',copies)]:
  for start,end in items: w.writerow([kind,(start-lo)/1e6,(end-start)/1e6,''])
fig,axes=plt.subplots(2,1,figsize=(12,6))
labels=['data','transfer','forward','backward','update']
for ax,window in zip(axes,[(0,(hi-lo)/1e6),(0,20)]):
 for i,label in enumerate(labels):
  ax.broken_barh([((start-lo)/1e6,(end-start)/1e6) for start,end,name in ranges if name==label],(i,.7),facecolors=plt.get_cmap('tab10')(i))
 for i,(label,items) in enumerate([('GPU kernel',kernels),('GPU copy',copies)],len(labels)):
  ax.broken_barh([((start-lo)/1e6,(end-start)/1e6) for start,end in items],(i,.7))
 ax.set_yticks([i+.35 for i in range(7)]);ax.set_yticklabels(labels+['GPU kernel','GPU copy']);ax.set_xlim(*window);ax.set_xlabel('ms relative to first training NVTX range')
fig.suptitle('Measured Nsight Systems intervals (instrumented; not benchmark latency)');fig.tight_layout();fig.savefig(a.output/'timeline.png',dpi=180)
print(json.dumps(summary,indent=2))
