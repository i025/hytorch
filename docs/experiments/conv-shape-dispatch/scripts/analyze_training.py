"""Five-way matched training, numeric arrays and dispatch counts, without hashes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/experiments/fused-sgd-cuda/scripts'))
from analyze_fused_results import arrays,identical,difference,statistics,write_csv,json_file,json_lines

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
assert json_file(a.raw/'completion.json')['status']=='COMPLETED'
rows,steps,losses,errors,checks=[],[],[],[],[]
methods='ADFGE';devices=set()
for model in ('donkey','spatial'):
    for mode in ('throughput','phases'):
        for repeat in range(1,4):
            base=a.raw/f'training-{model}-{mode}-A-r{repeat}'
            initial,inputs,weights=[arrays(base/n) for n in ('initial_weights.npz','comparison_inputs.npz','final_weights.npz')]
            ref_loss=np.array([r['loss'] for r in json_lines(base/'losses.jsonl')])
            for method in methods:
                folder=a.raw/f'training-{model}-{mode}-{method}-r{repeat}'
                config,summary=json_file(folder/'config.json'),json_file(folder/'summary.json')
                assert config['no_fingerprints'] and not config['nvtx'] and config['model']==model
                assert summary['status']=='COMPLETED' and summary['gpu_status']=='EXECUTED'
                devices.add(config['monitor_id'])
                ss,ll=json_lines(folder/'steps.jsonl'),json_lines(folder/'losses.jsonl')
                assert len(ss)==len(ll)==256 and all(np.isfinite(r['loss']) for r in ll)
                check=dict(model=model,method=method,mode=mode,repeat=repeat,
                    initial_equal=identical(arrays(folder/'initial_weights.npz'),initial),
                    inputs_equal=identical(arrays(folder/'comparison_inputs.npz'),inputs))
                checks.append(check);assert check['initial_equal'] and check['inputs_equal']
                if method=='A':assert summary['dispatch_counts']=={}
                if method=='G':assert summary['dispatch_counts'].get('conv_dinput_auto_original',0)+summary['dispatch_counts'].get('conv_dinput_auto_fused',0)>0
                wall=sum(r['train_wall_s'] for r in summary['epochs']);samples=sum(r['samples'] for r in summary['epochs'])
                row=dict(model=model,method=method,mode=mode,repeat=repeat,train_wall_s=wall,samples=samples,
                    samples_per_sec=samples/wall,steps=len(ss),complete_step_wall_ms=wall*1000/len(ss),
                    profiled_step_ms=np.mean([r['step_s'] for r in ss])*1000,
                    forward_ms=None,backward_ms=None,update_ms=None,
                    loss_first=ll[0]['loss'],loss_last=ll[-1]['loss'],dispatch=json.dumps(summary['dispatch_counts']),source=str(folder))
                if mode=='phases':
                    for phase in ('forward','backward','update'):row[phase+'_ms']=np.mean([r[phase+'_s'] for r in ss])*1000
                rows.append(row)
                for name,data,dest in [('steps',ss,steps),('losses',ll,losses)]:
                    dest.extend(dict(r,model=model,method=method,mode=mode,repeat=repeat) for r in data)
                references=[('A',weights,ref_loss)]
                for reference in (['D'] if method=='F' else ['D','F'] if method=='G' else []):
                    other=a.raw/f'training-{model}-{mode}-{reference}-r{repeat}'
                    references.append((reference,arrays(other/'final_weights.npz'),np.array([r['loss'] for r in json_lines(other/'losses.jsonl')])))
                actual=arrays(folder/'final_weights.npz')
                for reference,ww,lo in references:
                    for key in list(actual)+['loss']:
                        actual_value=np.array([r['loss'] for r in ll]) if key=='loss' else actual[key]
                        expected=lo if key=='loss' else ww[key]
                        delta=actual_value.astype(np.float64)-expected;norm=np.linalg.norm(expected.astype(np.float64))
                        errors.append(dict(model=model,method=method,mode=mode,repeat=repeat,reference=reference,parameter=key,
                            relative_l2=float(np.linalg.norm(delta)/norm) if norm else None,**difference(actual_value,expected)))
assert len(devices)==1,devices
stats=[]
for model in ('donkey','spatial'):
    for mode in ('throughput','phases'):
        metrics=['train_wall_s','samples_per_sec','complete_step_wall_ms','profiled_step_ms']+(['forward_ms','backward_ms','update_ms'] if mode=='phases' else [])
        for method in methods:
            for metric in metrics:
                stats.append(dict(model=model,method=method,mode=mode,metric=metric,
                    **statistics([r[metric] for r in rows if (r['model'],r['method'],r['mode'])==(model,method,mode)])))
def mean(model,method,metric):
    return next(r['mean'] for r in stats if (r['model'],r['method'],r['mode'],r['metric'])==(model,method,'throughput',metric))
gains,progress=[],[]
for model in ('donkey','spatial'):
    for new,old in [('D','A'),('F','D'),('F','A'),('G','D'),('G','F'),('G','A'),('G','E')]:
        rate=mean(model,new,'samples_per_sec')/mean(model,old,'samples_per_sec')
        speed=mean(model,old,'train_wall_s')/mean(model,new,'train_wall_s')
        gains.append(dict(model=model,comparison=new+'/'+old,throughput_ratio=rate,throughput_gain_pct=(rate-1)*100,
            wall_speedup=speed,wall_reduction_pct=(1-1/speed)*100))
    for method in methods:
        progress.append(dict(model=model,method=method,samples_per_sec=mean(model,method,'samples_per_sec'),
            std=next(r['std'] for r in stats if (r['model'],r['method'],r['mode'],r['metric'])==(model,method,'throughput','samples_per_sec')),
            complete_step_wall_ms=mean(model,method,'complete_step_wall_ms'),
            versus_A=mean(model,method,'samples_per_sec')/mean(model,'A','samples_per_sec'),
            versus_E=mean(model,method,'samples_per_sec')/mean(model,'E','samples_per_sec')))
for name,data in [('training-raw',rows),('training-steps',steps),('training-losses',losses),('training-errors',errors),
                  ('training-summary',stats),('speedups',gains),('progress-table',progress)]:write_csv(a.output/(name+'.csv'),data)
with (a.output/'checks.json').open('x') as f:json.dump(checks,f,indent=2);f.write('\n')
result=dict(status='PASSED' if all(r['passed'] for r in errors) else 'FAILED',
    gpu_uuid=list(devices)[0],training_processes=len(rows),steps=len(steps),comparisons=len(errors),
    max_abs=max(r['max_abs'] for r in errors))
with (a.output/'analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
assert result['status']=='PASSED','Numerical errors retained before failure'
print(json.dumps(dict(summary=result,progress=progress,gains=gains),indent=2))
