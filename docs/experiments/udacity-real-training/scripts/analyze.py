"""Offline shared-GPU analysis; all original timings and failed comparisons survive."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/experiments/fused-sgd-cuda/scripts'))
from analyze_fused_results import arrays,identical,difference,statistics


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def lines(path):return [json.loads(line) for line in path.read_text().splitlines()]
def js(path):return json.loads(path.read_text())


def write(path,rows):
    if not rows:return
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']==''
    a.analysis.mkdir(parents=True,exist_ok=False);a.output.mkdir(parents=True,exist_ok=False)
    window=js(a.raw/'window.json');assert window['status']=='COMPLETED' and window['physical_gpu']==1 and not window['gpu_reset_called']
    protocol=js(a.prepared/'protocol.json');initial=arrays(a.raw/'compute-A-r1/initial_weights.npz')
    raw,epochs,steps,losses,errors,predictions=[],[],[],[],[],[]
    commands=js(a.raw/'commands.json');assert len(commands)==47 and all(r['returncode']==0 for r in commands)
    for job in commands:
        if job['name']=='correctness-real':continue
        name=job['name'];folder=a.raw/name;config=js(folder/'config.json');summary=js(folder/'summary.json')
        assert config['protocol']==protocol and config['cuda_visible_devices']==window['uuid'] and config['no_fingerprints']
        assert identical(arrays(folder/'initial_weights.npz'),initial)
        rr,ll=lines(folder/'steps.jsonl'),lines(folder/'losses.jsonl')
        assert len(rr)==len(ll) and all(np.isfinite(r['loss']) for r in ll)
        method=config['method'];task=config['task'];repeat=int(name[-1]) if '-r' in name else 0
        if method=='A':assert summary['dispatch_counts']=={}
        if method=='I':assert summary['dispatch_counts']['conv_dweight_fused']==5*len(rr)
        for record in lines(folder/'epochs.jsonl'):
            epochs.append(dict(record,run=name,method=method,task=task,repeat=repeat))
        steps.extend(dict(r,run=name,method=method,task=task,repeat=repeat) for r in rr)
        losses.extend(dict(r,run=name,method=method,task=task,repeat=repeat) for r in ll)
        wall=sum(r['train_wall_s'] for r in epochs if r['run']==name and r['epoch']>0)
        samples=sum(r['samples'] for r in rr)
        row=dict(run=name,method=method,task=task,repeat=repeat,steps=len(rr),samples=samples,wall_s=wall,
            samples_per_sec=samples/wall,complete_step_ms=wall*1000/len(rr),
            mean_read_preprocess_ms=np.mean([r['read_preprocess_s'] for r in ll])*1000,
            loss_first=ll[0]['loss'],loss_last=ll[-1]['loss'],dispatch=json.dumps(summary['dispatch_counts']))
        if task in ('phases','train'):
            for phase in ('data','transfer','forward','backward','update'):row[phase+'_ms']=np.mean([r[phase+'_s'] for r in rr])*1000
        raw.append(row)
        if task in ('compute','io','phases','train'):
            suffix=('-r'+str(repeat)) if repeat else ''
            for reference in ('A','E',*(['F'] if method=='I' else [])):
                ref=a.raw/(task+'-'+reference+suffix);ww=arrays(ref/'final_weights.npz');actual=arrays(folder/'final_weights.npz')
                ref_loss=np.array([r['loss'] for r in lines(ref/'losses.jsonl')]);act_loss=np.array([r['loss'] for r in ll])
                for key in [*actual,'loss']:
                    av,bv=(act_loss,ref_loss) if key=='loss' else (actual[key],ww[key])
                    errors.append(dict(run=name,reference=reference,parameter=key,**difference(av,bv)))
        if task=='train':
            pp=lines(folder/'validation-predictions.jsonl');predictions.extend(dict(r,method=method) for r in pp)
            for epoch in range(4):
                selected=[r for r in pp if r['epoch']==epoch];assert len(selected)==len(protocol['validation'])
                delta=np.array([r['prediction']-r['target'] for r in selected])
                recorded=next(r for r in epochs if r['run']==name and r['epoch']==epoch)
                assert abs(float(np.mean(delta**2))-recorded['validation_loss'])<1e-12
    stats=[];gains=[]
    for task in ('compute','io','phases'):
        for method in 'AFIE':
            rr=[r for r in raw if (r['task'],r['method'])==(task,method)];assert len(rr)==3 and all(r['steps']==256 for r in rr)
            metrics=['wall_s','samples_per_sec','complete_step_ms','mean_read_preprocess_ms']+(['data_ms','transfer_ms','forward_ms','backward_ms','update_ms'] if task=='phases' else [])
            stats.extend(dict(task=task,method=method,metric=metric,**statistics([r[metric] for r in rr])) for metric in metrics)
        if task!='phases':
            def mean(method,metric):return next(r['mean'] for r in stats if (r['task'],r['method'],r['metric'])==(task,method,metric))
            for new,old in [('F','A'),('I','F'),('I','A'),('I','E')]:
                ratio=mean(new,'samples_per_sec')/mean(old,'samples_per_sec');speed=mean(old,'wall_s')/mean(new,'wall_s')
                gains.append(dict(task=task,comparison=new+'/'+old,throughput_ratio=ratio,throughput_gain_pct=(ratio-1)*100,
                    wall_speedup=speed,wall_reduction_pct=(1-1/speed)*100,resource_condition='SHARED_GPU'))
    for name,rows in [('training-raw',raw),('epochs',epochs),('steps',steps),('losses',losses),('numerical-errors',errors),
        ('validation-predictions',predictions),('timing-summary',stats),('speedups',gains)]:write(a.output/(name+'.csv'),rows)
    trace_rows=[];parts=[]
    for method in 'FI':
        name='trace-'+method;db=a.analysis/(name+'.sqlite');parsed=a.analysis/name
        for stage,command in [('export',['nsys','export','--type','sqlite','--output',str(db),str(a.raw/(name+'.nsys-rep'))]),
            ('parse',[sys.executable,str(ROOT/'docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py'),
                '--input',str(db),'--output',str(parsed),'--label',name])]:
            with (a.analysis/(name+'-'+stage+'.log')).open('x') as f:done=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
            with (a.analysis/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=command,returncode=done.returncode))+'\n')
            done.check_returncode()
        assert js(parsed/'profiling-summary.json')['unmatched_kernel_api_count']==0
        trace_rows.extend(dict(r,method=method) for r in read(parsed/'phase-summary.csv'))
        kk=read(parsed/'kernel-events.csv')
        for k in kk:k['ancestors']=set(json.loads(k['nvtx_ids']))
        for r in read(parsed/'nvtx-events.csv'):
            if not r['name'].startswith('OP/part:Conv2d.dWeight|layer='):continue
            if r['phase']!='backward':continue
            events=[k for k in kk if int(r['sqlite_rowid']) in k['ancestors']]
            row=dict(method=method,layer=r['name'].split('layer=')[1],host_us=float(r['duration_ms'])*1000,
                kernels=len(events),kernel_us=sum(float(k['duration_ms']) for k in events)*1000)
            for label,test in [('Im2col',lambda n:n.startswith('im2col')),('Copy',lambda n:'cupy_copy' in n),
                ('GEMM',lambda n:'gemm' in n.lower()),('tiles',lambda n:'conv_dweight_tiles' in n),('reduce',lambda n:'conv_dweight_reduce' in n)]:
                selected=[k for k in events if test(k['name'])];row[label+'_count']=len(selected);row[label+'_us']=sum(float(k['duration_ms']) for k in selected)*1000
            if method=='I':assert row['kernels']==2 and row['tiles_count']==row['reduce_count']==1 and row['Copy_count']==row['Im2col_count']==row['GEMM_count']==0
            parts.append(row)
        assert sum(r['method']==method for r in parts)==80
    write(a.output/'trace-phases.csv',trace_rows);write(a.output/'dweight-components.csv',parts)
    resources=lines(a.raw/'resources.jsonl');flat=[];processes=[]
    original={r['pid'] for r in js(a.raw/'before.json')['processes']}
    for r in resources:
        flat.append(dict(timestamp_utc=r['timestamp_utc'],job=r['job'],**r['gpu'],
            original_present=len(r['original_pids_present']),original_missing=len(r['original_pids_missing']),
            own_memory_mib=sum(int(v['memory_mib']) for v in r['processes'] if v['pid'] in r['own_gpu_pids'] and v['memory_mib'].isdigit())))
        for line in r['pmon'].splitlines():
            fields=line.split()
            if len(fields)>=5 and fields[0]=='1' and fields[1].isdigit():
                processes.append(dict(timestamp_utc=r['timestamp_utc'],job=r['job'],pid=int(fields[1]),baseline_original=int(fields[1]) in original,
                    sm_pct=int(fields[3]) if fields[3].isdigit() else None,memory_activity_pct=int(fields[4]) if fields[4].isdigit() else None))
    write(a.output/'resources.csv',flat);write(a.output/'process-activity.csv',processes)
    correct=js(a.raw/'correctness-real/summary.json');assert correct['status']=='PASSED'
    manifest=read(a.prepared/'manifest.csv')
    train_labels=np.array([float(r['steering']) for r in manifest if r['split']=='train'])
    validation_labels=np.array([float(r['steering']) for r in manifest if r['split']=='validation'])
    baselines=[]
    for name,value in [('zero',0.),('training_mean',float(train_labels.mean()))]:
        delta=value-validation_labels
        baselines.append(dict(predictor=name,prediction=value,validation_mse=float(np.mean(delta**2)),
            validation_mae=float(np.mean(abs(delta))),validation_rmse=float(np.sqrt(np.mean(delta**2)))))
    write(a.output/'constant-baselines.csv',baselines)
    result=dict(status='COMPLETED',resource_condition='SHARED_GPU',window=window,correctness=correct,
        parameter_loss_comparisons=len(errors),numerical_passed=all(r['passed'] for r in errors),
        failed_numerical_comparisons=sum(not r['passed'] for r in errors),max_numerical_abs=max(r['max_abs'] for r in errors),
        original_pids=sorted(original),resource_samples=len(resources),observed_original_missing_samples=sum(bool(r['original_pids_missing']) for r in resources),
        max_own_sampled_memory_mib=max(r['own_memory_mib'] for r in flat),original_latency='NOT_AVAILABLE',trace_count=2,
        numerical_failures_by_task={task:sum(not r['passed'] for r in errors if r['run'].startswith(task+'-')) for task in ('compute','io','phases','train')},
        prepared=str(a.prepared.resolve()),raw=str(a.raw.resolve()),analysis=str(a.analysis.resolve()))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(summary=result,gains=gains),indent=2))


if __name__=='__main__':main()
