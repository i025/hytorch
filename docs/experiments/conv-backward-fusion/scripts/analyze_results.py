"""Analyze a completed stage 4D GPU window; never substitute historical timings."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/experiments/fused-sgd-cuda/scripts'))
from analyze_fused_results import arrays,identical,statistics,difference,csv_rows,write_csv,json_file,json_lines

METHODS='ABCDFE'
PHASES={'data','transfer','forward','backward','update'}


def error(actual,reference):
    result=difference(actual,reference)
    norm=np.linalg.norm(reference.astype(np.float64))
    result['relative_l2']=float(np.linalg.norm(actual.astype(np.float64)-reference)/norm) if norm else None
    return result


def training(raw,out):
    rows,steps,losses,errors,checks=[],[],[],[],[]
    for mode in ('throughput','phases'):
        for repeat in range(1,4):
            base=raw/f'training-{mode}-A-r{repeat}'
            initial,inputs,weights=[arrays(base/name) for name in ('initial_weights.npz','comparison_inputs.npz','final_weights.npz')]
            base_loss=np.array([r['loss'] for r in json_lines(base/'losses.jsonl')])
            for method in METHODS:
                path=raw/f'training-{mode}-{method}-r{repeat}'
                config,summary=json_file(path/'config.json'),json_file(path/'summary.json')
                assert config['no_fingerprints'] and not config['nvtx']
                assert summary['status']=='COMPLETED' and summary['gpu_status']=='EXECUTED'
                ss,ll=json_lines(path/'steps.jsonl'),json_lines(path/'losses.jsonl')
                wall=sum(r['train_wall_s'] for r in summary['epochs'])
                samples=sum(r['samples'] for r in summary['epochs'])
                row=dict(method=method,mode=mode,repeat=repeat,train_wall_s=wall,samples=samples,
                    samples_per_sec=samples/wall,steps=len(ss),forward_ms=None,backward_ms=None,update_ms=None,
                    loss_first=ll[0]['loss'],loss_last=ll[-1]['loss'],dispatch=json.dumps(summary['dispatch_counts']),source=str(path))
                if mode=='phases':
                    for phase in ('forward','backward','update'):
                        row[phase+'_ms']=1000*np.mean([s[phase+'_s'] for s in ss])
                rows.append(row)
                steps.extend(dict(r,method=method,mode=mode,repeat=repeat) for r in ss)
                losses.extend(dict(r,method=method,mode=mode,repeat=repeat) for r in ll)
                check=dict(method=method,mode=mode,repeat=repeat,
                    initial_equal=identical(arrays(path/'initial_weights.npz'),initial),
                    inputs_equal=identical(arrays(path/'comparison_inputs.npz'),inputs))
                checks.append(check)
                assert check['initial_equal'] and check['inputs_equal']
                if method=='A':assert summary['dispatch_counts']=={}
                assert summary['dispatch_counts'].get('conv_dinput_fused',0)==(1024 if method=='F' else 0)
                actual=arrays(path/'final_weights.npz')
                actual_loss=np.array([r['loss'] for r in ll])
                references=[('A',weights,base_loss)]
                if method=='F':
                    d=raw/f'training-{mode}-D-r{repeat}'
                    references.append(('D',arrays(d/'final_weights.npz'),np.array([r['loss'] for r in json_lines(d/'losses.jsonl')])))
                for label,reference,ref_loss in references:
                    for key in actual:
                        errors.append(dict(method=method,mode=mode,repeat=repeat,reference=label,parameter=key,**error(actual[key],reference[key])))
                    errors.append(dict(method=method,mode=mode,repeat=repeat,reference=label,parameter='loss',**error(actual_loss,ref_loss)))
    stats=[]
    for mode in ('throughput','phases'):
        metrics=['train_wall_s','samples_per_sec']+(['forward_ms','backward_ms','update_ms'] if mode=='phases' else [])
        for method in METHODS:
            for metric in metrics:
                stats.append(dict(method=method,mode=mode,metric=metric,
                    **statistics([r[metric] for r in rows if r['method']==method and r['mode']==mode])))
    def mean(method,metric):
        return next(r['mean'] for r in stats if r['method']==method and r['mode']=='throughput' and r['metric']==metric)
    gains=[]
    for new,old in [('B','A'),('C','B'),('D','C'),('F','D'),('F','A'),('F','E')]:
        rate=mean(new,'samples_per_sec')/mean(old,'samples_per_sec')
        wall=mean(old,'train_wall_s')/mean(new,'train_wall_s')
        gains.append(dict(comparison=new+'/'+old,throughput_ratio=rate,throughput_gain_pct=(rate-1)*100,
            wall_speedup=wall,wall_reduction_pct=(1-1/wall)*100))
    progress=[dict(method=m,samples_per_sec=mean(m,'samples_per_sec'),versus_A=mean(m,'samples_per_sec')/mean('A','samples_per_sec'),
                   versus_E=mean(m,'samples_per_sec')/mean('E','samples_per_sec')) for m in METHODS]
    for name,data in [('training-raw',rows),('training-summary',stats),('training-steps',steps),('training-losses',losses),
                       ('training-errors',errors),('speedups',gains),('progress-table',progress)]:write_csv(out/(name+'.csv'),data)
    (out/'training-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    assert all(r['passed'] for r in errors), 'Training errors saved before assertion'


def micro(raw,out):
    timings,memory,errors,repeats=[],[],[],[]
    for repeat in range(1,4):
        base=raw/f'micro-original-r{repeat}'
        for method in ('original','fused','torch'):
            path=raw/f'micro-{method}-r{repeat}'
            assert not json_file(path/'config.json')['nvtx']
            local=csv_rows(path/'latencies.csv')
            for r in local:
                for key in ('event_us','host_submit_us','end_event_wait_us','sync_wall_us'):r[key]=float(r[key])
                timings.append(dict(r,repeat=repeat))
            memory.extend(dict(r,repeat=repeat) for r in csv_rows(path/'memory.csv'))
            for layer in ('conv2','conv3','conv4','conv5'):
                assert identical(arrays(path/(layer+'-inputs.npz')),arrays(base/(layer+'-inputs.npz')))
                for scope in ('dinput','full'):
                    av,bv=arrays(path/(layer+'-'+scope+'-outputs.npz')),arrays(base/(layer+'-'+scope+'-outputs.npz'))
                    for key in av:
                        delta=av[key].astype(np.float64)-bv[key]
                        norm=np.linalg.norm(bv[key].astype(np.float64))
                        errors.append(dict(method=method,repeat=repeat,layer=layer,scope=scope,tensor=key,
                            max_abs=float(np.abs(delta).max()),relative_l2=float(np.linalg.norm(delta)/norm),
                            atol=1e-5,rtol=1e-4,passed=bool(np.allclose(av[key],bv[key],atol=1e-5,rtol=1e-4))))
                    for metric in ('event_us','host_submit_us','end_event_wait_us','sync_wall_us'):
                        repeats.append(dict(method=method,repeat=repeat,layer=layer,scope=scope,metric=metric,
                            **statistics([r[metric] for r in local if r['layer']==layer and r['scope']==scope])))
    aggregate=[]
    for method in ('original','fused','torch'):
        for layer in ('conv2','conv3','conv4','conv5'):
            for scope in ('dinput','full'):
                for metric in ('event_us','host_submit_us','end_event_wait_us','sync_wall_us'):
                    values=[r['mean'] for r in repeats if (r['method'],r['layer'],r['scope'],r['metric'])==(method,layer,scope,metric)]
                    aggregate.append(dict(method=method,layer=layer,scope=scope,metric=metric,**statistics(values)))
    for name,data in [('micro-latencies',timings),('micro-memory',memory),('micro-errors',errors),
                       ('micro-repeats',repeats),('micro-summary',aggregate)]:write_csv(out/(name+'.csv'),data)
    assert all(r['passed'] for r in errors), 'Micro errors saved before assertion'


def interval_union(events):
    intervals=sorted((int(r['start']),int(r['end'])) for r in events)
    total=0;end=0
    for lo,hi in intervals:
        total+=max(0,hi-max(lo,end));end=max(end,hi)
    return total


def traces(raw,analysis,out):
    analysis.mkdir(parents=True,exist_ok=False)
    old=ROOT/'docs/experiments/donkeynet-backward-profiling/scripts/analyze_trace.py'
    phase_rows,categories,parts,windows,memory,micros=[],[],[],[],[],[]
    native=re.compile(r'\b(gemm_basic|gemm_blocked|gemm_shared|im2col_kernel|conv_direct|conv_pack|pool_kernel|multi_tensor_sgd_kernel|conv_dinput_gather)\b')
    names=['trace-training-'+m for m in METHODS]+['trace-micro-'+m for m in ('original','fused','torch')]
    for name in names:
        command=[sys.executable,str(old),'--input',str(raw/(name+'.sqlite')),'--output',str(analysis/name),'--label',name]
        with (analysis/(name+'.log')).open('x') as f:
            done=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        with (analysis/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=command,returncode=done.returncode))+'\n')
        done.check_returncode()
        folder=analysis/name
        summary=json_file(folder/'profiling-summary.json')
        assert summary['unmatched_kernel_api_count']==0
        kernels=csv_rows(folder/'kernel-events.csv');ranges=csv_rows(folder/'nvtx-events.csv')
        if name.startswith('trace-micro-'):
            for r in ranges:
                if not r['name'].startswith('OP/micro:'):continue
                scope,layer,step=re.match(r'OP/micro:(\w+)\|layer=(\w+)\|step=(\d+)',r['name']).groups()
                events=[k for k in kernels if int(r['sqlite_rowid']) in json.loads(k['nvtx_ids'])]
                micros.append(dict(method=name[len('trace-micro-'):],scope=scope,layer=layer,step=int(step),
                    kernels=len(events),kernel_us=sum(float(k['duration_ms']) for k in events)*1000))
            continue
        method=name[-1]
        phase_rows.extend(dict(r,method=method) for r in csv_rows(folder/'phase-summary.csv'))
        if method=='A':assert not any(native.search(k['name']) for k in kernels)
        if method!='E':
            instrumentation=json_file(raw/name/'instrumentation.json')
            memory.extend(dict(r,method=method) for r in instrumentation['memory'])
        groups=defaultdict(list)
        for k in kernels:
            if k['phase']!='backward':continue
            category=('fused_dInput' if 'conv_dinput_gather' in k['name'] else 'Copy' if 'cupy_copy' in k['name']
                      else 'Col2im' if k['name'].startswith('col2im') else 'GEMM' if 'gemm' in k['name'].lower() else 'other')
            groups[category].append(k)
        for category,events in groups.items():
            categories.append(dict(method=method,category=category,kernels=len(events),kernel_us=sum(float(k['duration_ms']) for k in events)*1000))
        for r in ranges:
            if r['phase']=='backward' and r['name'].startswith('OP/part:Conv2d.'):
                match=re.match(r'OP/part:Conv2d\.(\w+)\|layer=(\w+)',r['name'])
                events=[k for k in kernels if int(r['sqlite_rowid']) in json.loads(k['nvtx_ids'])]
                parts.append(dict(method=method,part=match[1],layer=match[2],nvtx_id=r['sqlite_rowid'],
                    host_us=float(r['duration_ms'])*1000,kernels=len(events),kernel_us=sum(float(k['duration_ms']) for k in events)*1000))
            if r['name']=='backward':
                events=[k for k in kernels if int(r['sqlite_rowid']) in json.loads(k['nvtx_ids'])]
                if not events:continue  # PyTorch may use the documented cross-thread API phase rule.
                span=max(int(k['end']) for k in events)-min(int(k['start']) for k in events)
                active=interval_union(events)
                windows.append(dict(method=method,nvtx_id=r['sqlite_rowid'],host_us=float(r['duration_ms'])*1000,
                    own_kernel_span_us=span/1000,own_kernel_union_us=active/1000,
                    gap_between_own_kernels_us=(span-active)/1000,scope='not whole-device idle/utilization'))
    for method in 'DF':
        for layer in ('conv2','conv3','conv4','conv5'):
            rr=[r for r in parts if r['method']==method and r['layer']==layer and r['part']=='dInput']
            assert len(rr)==16 and all(r['kernels']==(1 if method=='F' else 3) for r in rr)
        assert not any(r['method']==method and r['layer']=='conv1' and r['part']=='dInput' for r in parts)
    for name,data in [('trace-phases',phase_rows),('kernel-categories',categories),('gradient-components',parts),
                      ('backward-windows',windows),('allocation-requests',memory),('micro-kernel-launches',micros)]:write_csv(out/(name+'.csv'),data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert json_file(a.raw/'completion.json')['status']=='COMPLETED'
    training(a.raw,a.output);micro(a.raw,a.output);traces(a.raw,a.analysis,a.output)
    (a.output/'completion.json').write_text(json.dumps(dict(status='PASSED',raw=str(a.raw.resolve()),
        timing='Uninstrumented CUDA Events/host/wall, training throughput/phases, Nsight are separate scopes'),indent=2)+'\n')


if __name__=='__main__':main()
