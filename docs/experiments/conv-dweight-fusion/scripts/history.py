"""CPU extraction of prior 4F dWeight events; never a new GPU measurement."""
import csv
import json
from pathlib import Path
from cases import cases

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parents[1]/'results/history-dweight.csv'
rows=[]
for model,offset in [('donkey',0),('spatial',5)]:
    folder=ROOT/'runs/conv_dinput_opt_trace_analysis_01'/('trace-training-'+model+'-F')
    with (folder/'kernel-events.csv').open() as f:kernels=list(csv.DictReader(f))
    with (folder/'nvtx-events.csv').open() as f:ranges=list(csv.DictReader(f))
    for layer in range(1,6 if model=='donkey' else 4):
        rr=[r for r in ranges if r['phase']=='backward' and r['name'].startswith('OP/part:Conv2d.dWeight|layer=conv'+str(layer))]
        ids={int(r['sqlite_rowid']) for r in rr}
        kk=[k for k in kernels if ids.intersection(json.loads(k['nvtx_ids']))]
        case=cases()[124+offset+layer-1];n,c,h,w=case['shape'];oc,_,kh,kw=case['weight_shape'];s,p=case['stride'],case['pad']
        oh,ow=(h+2*p-kh)//s+1,(w+2*p-kw)//s+1
        row=dict(model=model,layer='conv'+str(layer),shape=str(case['shape']),weight_shape=str(case['weight_shape']),
            stride=s,pad=p,calls=len(rr),host_us=sum(float(r['duration_ms']) for r in rr)*1000/len(rr),
            kernels_per_call=len(kk)/len(rr),kernel_us=sum(float(k['duration_ms']) for k in kk)*1000/len(rr),
            im2col_bytes=4*n*c*kh*kw*oh*ow,native_partial_bytes=4*((n*oh*ow+127)//128)*oc*c*kh*kw,
            source=str(folder.resolve()))
        for label,test in [('im2col',lambda name:name.startswith('im2col')),('copy',lambda name:'cupy_copy' in name),('gemm',lambda name:'gemm' in name.lower())]:
            selected=[k for k in kk if test(k['name'])]
            row[label+'_count']=len(selected)/len(rr);row[label+'_us']=sum(float(k['duration_ms']) for k in selected)*1000/len(rr)
        rows.append(row)
with OUT.open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Preserved',len(rows),'historical dWeight layer rows, not current measurements')
