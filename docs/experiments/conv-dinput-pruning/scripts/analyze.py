"""CPU analysis of five matched configurations and correlation-attributed traces."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT/'docs/experiments/donkeynet-backward-profiling/scripts'
sys.path.insert(0,str(ROOT/'docs/experiments/fused-sgd-cuda/scripts'))
from analyze_fused_results import arrays, identical, statistics, difference, csv_rows, write_csv

METHODS = 'ABCDE'
LABELS = {'A':'CuPy / SGD','B':'Combined / SGD','C':'Combined / fused SGD',
          'D':'Combined / fused / prune','E':'PyTorch'}


def read_json(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def training(raw,out):
    runs,steps,losses,checks,errors = [],[],[],[],[]
    for mode in ['throughput','phases']:
        for repeat in range(1,4):
            base = raw/f'training-{mode}-A-r{repeat}'
            init,inputs,final = [arrays(base/f) for f in ['initial_weights.npz','comparison_inputs.npz','final_weights.npz']]
            reference_loss = np.array([r['loss'] for r in read_lines(base/'losses.jsonl')])
            for method in METHODS:
                path = raw/f'training-{mode}-{method}-r{repeat}'
                cfg,s = read_json(path/'config.json'),read_json(path/'summary.json')
                assert not cfg['nvtx'] and cfg['no_fingerprints']
                local_steps = read_lines(path/'steps.jsonl')
                local_loss = read_lines(path/'losses.jsonl')
                wall = sum(e['train_wall_s'] for e in s['epochs'])
                samples = sum(e['samples'] for e in s['epochs'])
                row = dict(method=method,mode=mode,repeat=repeat,train_wall_s=wall,
                           samples=samples,steps=len(local_steps),samples_per_sec=samples/wall,
                           step_ms=1000*np.mean([v['step_s'] for v in local_steps]),
                           loss_first=local_loss[0]['loss'],loss_last=local_loss[-1]['loss'],
                           loss_finite=bool(all(np.isfinite(v['loss']) for v in local_loss)),
                           forward_ms=None,backward_ms=None,update_ms=None,
                           dispatch_counts=json.dumps(s['dispatch_counts']),source=str(path))
                if mode == 'phases':
                    for phase in ['forward','backward','update']:
                        row[phase+'_ms'] = 1000*np.mean([v[phase+'_s'] for v in local_steps])
                runs.append(row)
                steps.extend(dict(v,method=method,mode=mode,repeat=repeat) for v in local_steps)
                losses.extend(dict(v,method=method,mode=mode,repeat=repeat) for v in local_loss)
                check = dict(method=method,mode=mode,repeat=repeat,
                             initial_equal=identical(arrays(path/'initial_weights.npz'),init),
                             inputs_equal=identical(arrays(path/'comparison_inputs.npz'),inputs),
                             dispatch=s['dispatch_counts'],status=s['status'])
                if method == 'A':
                    assert cfg['matmul']==cfg['conv']==cfg['pool']=='cupy'
                    assert not cfg['fused_sgd'] and not cfg['prune_input_grad']
                    assert s['dispatch_counts']=={}
                checks.append(check)
                actual = arrays(path/'final_weights.npz')
                for key in final:
                    errors.append(dict(method=method,mode=mode,repeat=repeat,reference='A',
                                       parameter=key,**difference(actual[key],final[key])))
                errors.append(dict(method=method,mode=mode,repeat=repeat,reference='A',parameter='loss',
                                   **difference(np.array([v['loss'] for v in local_loss]),reference_loss)))
                if method=='D':
                    c = raw/f'training-{mode}-C-r{repeat}'
                    cfinal=arrays(c/'final_weights.npz')
                    for key in final:
                        errors.append(dict(method=method,mode=mode,repeat=repeat,reference='C',
                                           parameter=key,**difference(actual[key],cfinal[key])))
                    errors.append(dict(method=method,mode=mode,repeat=repeat,reference='C',parameter='loss',
                        **difference(np.array([v['loss'] for v in local_loss]),np.array([v['loss'] for v in read_lines(c/'losses.jsonl')]))))
    summary=[]
    for mode in ['throughput','phases']:
        for method in METHODS:
            rr=[r for r in runs if r['mode']==mode and r['method']==method]
            metrics=['train_wall_s','samples_per_sec','step_ms']
            if mode=='phases':metrics+=['forward_ms','backward_ms','update_ms']
            for metric in metrics:
                summary.append(dict(method=method,mode=mode,metric=metric,**statistics([r[metric] for r in rr])))
    def mean(method,metric):
        return next(r['mean'] for r in summary if r['mode']=='throughput' and r['method']==method and r['metric']==metric)
    speedups=[]
    for new,old,label in [('B','A','native_increment'),('C','B','fused_increment'),('D','C','pruning_increment'),('D','A','cumulative'),('D','E','versus_pytorch')]:
        rate=mean(new,'samples_per_sec')/mean(old,'samples_per_sec')
        wall=mean(old,'train_wall_s')/mean(new,'train_wall_s')
        speedups.append(dict(comparison=new+'/'+old,label=label,throughput_ratio=rate,
                             throughput_gain_pct=(rate-1)*100,wall_speedup=wall,
                             wall_reduction_pct=(1-1/wall)*100))
    progress=[]
    for method in METHODS:
        previous={'B':'A','C':'B','D':'C'}.get(method)
        progress.append(dict(version=method,description=LABELS[method],samples_per_sec=mean(method,'samples_per_sec'),
                             versus_previous=mean(method,'samples_per_sec')/mean(previous,'samples_per_sec') if previous else None,
                             versus_A=mean(method,'samples_per_sec')/mean('A','samples_per_sec'),
                             versus_E=mean(method,'samples_per_sec')/mean('E','samples_per_sec')))
    for name,rows in [('training-raw',runs),('training-steps',steps),('training-losses',losses),
                      ('training-summary',summary),('numerical-errors',errors),('speedups',speedups),('progress-table',progress)]:
        write_csv(out/(name+'.csv'),rows)
    numerical=dict(status='PASSED' if all(r['passed'] for r in errors) else 'FAILED',
                   tolerance=dict(atol=1e-6,rtol=1e-5),checks=checks,
                   all_initial_inputs_equal=all(r['initial_equal'] and r['inputs_equal'] for r in checks),
                   loss_finite=all(r['loss_finite'] for r in runs),
                   max_abs=max(r['max_abs'] for r in errors),
                   pruning_exact=all(r['exact_equal'] for r in errors if r['reference']=='C'),
                   failures=[r for r in errors if not r['passed']])
    (out/'numerical-summary.json').write_text(json.dumps(numerical,indent=2)+'\n')
    assert numerical['status']=='PASSED' and numerical['all_initial_inputs_equal'] and numerical['loss_finite']
    return runs,summary,losses,errors,speedups


def traces(raw,analysis,out):
    analysis.mkdir(parents=True,exist_ok=False)
    phases,components,layouts,native,trace_summary=[],[],[],[],[]
    native_pattern=re.compile(r'\b(gemm_basic|gemm_blocked|gemm_shared|im2col_kernel|conv_direct|conv_pack|pool_kernel|multi_tensor_sgd_kernel)\b')
    for method in METHODS:
        name='trace-'+method
        target=analysis/name
        command=[sys.executable,str(OLD/'analyze_trace.py'),'--input',str(raw/(name+'.sqlite')),
                 '--output',str(target),'--label',name]
        with (analysis/(name+'.log')).open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        with (analysis/'commands.jsonl').open('a') as log:
            log.write(json.dumps(dict(command=command,returncode=result.returncode))+'\n')
        result.check_returncode()
        p=csv_rows(target/'phase-summary.csv')
        kernels=csv_rows(target/'kernel-events.csv')
        ops=csv_rows(target/'operator-hotspots.csv')
        ts=read_json(target/'profiling-summary.json')
        assert ts['unmatched_kernel_api_count']==0
        for row in p:phases.append(dict(row,method=method))
        for row in kernels:
            if native_pattern.search(row['name']):
                native.append(dict(method=method,phase=row['phase'],kernel=row['name'],duration_ms=float(row['duration_ms'])))
        if method=='A':assert not [r for r in native if r['method']=='A']
        for layer in ['conv'+str(i) for i in range(1,6)]:
            for part in ['dInput','dWeight','dBias']:
                selected=[r for r in ops if r['attribution']=='inclusive' and r['phase']=='backward'
                          and r['operator'].startswith('OP/part:Conv2d.'+part+'|layer='+layer)]
                if method=='E':continue
                count=sum(int(r['operator_nvtx_call_count']) for r in selected)
                components.append(dict(method=method,layer=layer,part=part,ranges=count,
                    gpu_kernel_count=sum(int(r['gpu_kernel_count']) for r in selected),
                    gpu_kernel_ms=sum(float(r['gpu_kernel_sum_ms']) for r in selected),
                    host_ms=sum(float(r['host_nvtx_inclusive_ms']) for r in selected)))
                assert count==(0 if method=='D' and layer=='conv1' and part=='dInput' else 16),(method,layer,part,count)
        groups=defaultdict(list)
        for row in kernels:
            if row['phase']!='backward':continue
            path=json.loads(row['operator_path'])
            layer=next((re.search(r'\|layer=([^|]+)',s).group(1) for s in reversed(path) if '|layer='in s),'unlabelled')
            part=next((s.split('|')[0].split('Conv2d.')[1] for s in path if s.startswith('OP/part:Conv2d.')),'other')
            for category in ['col2im','im2col','tensordot','copy']:
                matched=(row['name'].startswith('cupy_copy') if category=='copy' else
                         any(('OP/internal:'+('cupy.' if category=='tensordot' else '')+category) in s for s in path))
                if matched:groups[(layer,part,category)].append(row)
        for (layer,part,category),rows in groups.items():
            layouts.append(dict(method=method,layer=layer,part=part,category=category,
                                kernels=len(rows),gpu_ms=sum(float(r['duration_ms']) for r in rows)))
        trace_summary.append(dict(method=method,whole_trace_kernels=ts['whole_trace_kernel_count'],
            measured_kernels=ts['training_phase_submitted_kernel_count'],envelope_ms=ts['training_envelope_ms'],
            coverage=ts['activity_coverage_fraction'],unmatched=ts['unmatched_kernel_api_count']))
    for name,rows in [('trace-phases',phases),('gradient-components',components),('layout-kernels',layouts),
                      ('native-kernel-events',native),('trace-summary',trace_summary)]:write_csv(out/(name+'.csv'),rows)
    (out/'trace-checks.json').write_text(json.dumps(dict(status='PASSED',A_native_kernel_count=0,
        first_dInput_D_ranges=0,other_convolution_components_each=16,unmatched=0,
        attribution='CUDA correlation -> submitting API -> NVTX; no asynchronous GPU-overlap attribution',
        warning='Inclusive layout/gradient components overlap; do not add categories as disjoint GPU work'),indent=2)+'\n')
    return phases,components,layouts


def plots(figures,runs,summary,losses,errors,speedups,phases,components,layouts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures.mkdir(parents=True,exist_ok=True)
    colors=['#718096','#4477aa','#ee9933','#228866','#aa5599']
    def save(fig,name):
        fig.text(.01,.01,'Source: measured CSV / Nsight export visualization; not a screenshot.',fontsize=8)
        fig.text(.01,.04,'A: CuPy; B: Combined; C: + fused SGD; D: + dInput pruning; E: PyTorch',fontsize=8)
        fig.tight_layout(rect=(0,.09,1,1))
        with (figures/name).open('xb') as file:
            fig.savefig(file,format='png',dpi=170)
        plt.close(fig)
    def metric(m,mode,key):return next(r for r in summary if r['method']==m and r['mode']==mode and r['metric']==key)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(list(METHODS),[metric(m,'throughput','samples_per_sec')['mean'] for m in METHODS],
           yerr=[metric(m,'throughput','samples_per_sec')['std'] for m in METHODS],color=colors,capsize=4)
    for m in METHODS:ax.scatter([m]*3,[r['samples_per_sec'] for r in runs if r['method']==m and r['mode']=='throughput'],color='black',s=15)
    ax.set(title='DonkeyNet: 3 independent runs x 256 steps',ylabel='Samples/s (mean +/- process sample SD)')
    save(fig,'five-version-throughput.png')
    fig,ax=plt.subplots(figsize=(9,5))
    base=metric('A','throughput','samples_per_sec')['mean']
    bars=ax.bar(list(METHODS),[metric(m,'throughput','samples_per_sec')['mean']/base for m in METHODS],color=colors)
    ax.bar_label(bars,fmt='%.3fx');ax.axhline(1,color='black',linestyle='--',lw=1)
    ax.set(title='Same-window throughput relative to current-code CuPy baseline A',ylabel='Throughput ratio vs A (PyTorch E is a control)')
    save(fig,'cumulative-speedup.png')
    fig,ax=plt.subplots(figsize=(9,5))
    rr=speedups[:3];bars=ax.bar([r['comparison'] for r in rr],[r['throughput_gain_pct'] for r in rr],color=colors[1:4])
    ax.bar_label(bars,fmt='%.2f%%');ax.axhline(0,color='black',lw=1)
    ax.set(title='Measured increments: native / fused SGD / dInput pruning',ylabel='Throughput improvement (%)')
    save(fig,'incremental-gains.png')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,trace,title in [(axes[0],False,'Uninstrumented synchronized phase runs'),(axes[1],True,'Instrumented Nsight GPU kernel sums')]:
        values=[next(float(r['gpu_kernel_sum_ms'])/int(r['nvtx_count']) for r in phases if r['method']==m and r['phase']=='backward') if trace else metric(m,'phases','backward_ms')['mean'] for m in 'CD']
        ax.bar(list('CD'),values,color=colors[2:4]);ax.set(title=title,ylabel='Backward ms / step')
    save(fig,'backward-time.png')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    values=[sum(float(r['gpu_kernel_count']) for r in phases if r['method']==m and r['phase'] in ['data','transfer','forward','backward','update'])/16 for m in METHODS]
    bars=axes[0].bar(list(METHODS),values,color=colors);axes[0].bar_label(bars,fmt='%.0f');axes[0].set(title='Measured training kernels / step',ylabel='Kernel count')
    xx=np.arange(5)
    for m,offset,color in [('C',-.18,colors[2]),('D',.18,colors[3])]:
        vals=[sum(r['gpu_ms'] for r in layouts if r['method']==m and r['layer']=='conv'+str(i) and r['category']=='col2im')*1000/16 for i in range(1,6)]
        axes[1].bar(xx+offset,vals,width=.36,color=color,label=m)
    axes[1].set_xticks(xx,['conv'+str(i) for i in range(1,6)]);axes[1].set(title='Col2im GPU kernel time by layer',ylabel='Microseconds / step');axes[1].legend()
    save(fig,'kernel-and-col2im.png')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for m,color in zip(METHODS,colors):
        loss=[r['loss'] for r in losses if r['method']==m and r['mode']=='throughput' and r['repeat']==1]
        axes[0].plot(np.arange(1,len(loss)+1),loss,color=color,label=m,alpha=.8)
        err=[r['max_abs'] for r in errors if r['method']==m and r['reference']=='A']
        axes[1].bar(m,max(err),color=color)
    axes[0].set(title='Matched synthetic training, repeat 1',xlabel='Step',ylabel='MSE Loss');axes[0].legend()
    axes[1].set(title='All modes/repeats: maximum absolute difference vs A',ylabel='Absolute difference (weights or loss)')
    save(fig,'loss-and-correctness.png')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--figures',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    runs,summary,losses,errors,speedups=training(a.raw,a.output)
    phases,components,layouts=traces(a.raw,a.analysis,a.output)
    plots(a.figures,runs,summary,losses,errors,speedups,phases,components,layouts)
    (a.output/'completion.json').write_text(json.dumps(dict(status='PASSED',training_runs=len(runs),
        trace_count=5,raw_root=str(a.raw.resolve()),trace_analysis_root=str(a.analysis.resolve())),indent=2)+'\n')
    print(json.dumps(speedups,indent=2))


if __name__=='__main__':main()
