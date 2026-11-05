"""CPU-only analysis of actual records, including partial/NOT_RUN GPU work."""
import argparse
import csv
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    if not rows:return
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def tests(path):
    if not path.exists():return dict(status='NOT_RUN')
    root=ET.parse(path).getroot();suites=list(root.iter('testsuite'))
    values={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    return dict(status='FAIL' if values['failures'] or values['errors'] else 'SKIPPED' if values['skipped'] else 'PASS',
        passed=values['tests']-values['failures']-values['errors']-values['skipped'],**values)


def gpu_tests(path,window,name):
    value=tests(path)
    if value['status']=='NOT_RUN' and any(v['name']==name for v in window.get('commands',[])):
        return dict(status='PARTIAL',reason='Task started but no complete JUnit XML was saved; inspect its original log')
    return value


def mean_sd(values):
    return statistics.mean(values),statistics.stdev(values) if len(values)>1 else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--final',action='store_true')
    a=p.parse_args();config=json.loads(a.schedule.read_text());raw=Path(config['raw']);report=Path(config['report'])
    out=report/'benchmarks'/a.label;out.mkdir(exist_ok=False)
    figures=report/'figures'/a.label;figures.mkdir(exist_ok=False)
    rows=list(csv.DictReader((report/'gpu-monitor.csv').open()))
    first=datetime.fromisoformat(rows[0]['utc']);last=datetime.fromisoformat(rows[-1]['utc'])
    gaps=[(datetime.fromisoformat(y['utc'])-datetime.fromisoformat(x['utc'])).total_seconds() for x,y in zip(rows,rows[1:])]
    monitor=dict(samples=len(rows),first_utc=first.isoformat(),last_utc=last.isoformat(),
        observed_seconds=(last-first).total_seconds(),max_gap_s=max(gaps) if gaps else None,
        max_idle_streak=max(int(r['idle_streak']) for r in rows),
        idle_samples=sum(r['idle']=='True' for r in rows),
        qualified_samples=sum(int(r['idle_streak'])>=config['idle_samples'] for r in rows),
        qualified_before_gpu_cutoff=sum(int(r['idle_streak'])>=config['idle_samples'] and
            datetime.fromisoformat(r['utc'])<datetime.fromisoformat(config['gpu_cutoff_utc']) for r in rows),
        occupied_zero_utilization_samples=sum(int(r['utilization_pct'])==0 and bool(json.loads(r['foreign_processes'])) for r in rows),
        distinct_compute_pids=sorted({v['pid'] for r in rows for v in json.loads(r['processes'])}),
        uuid=config['uuid'],physical_gpu=0)
    times=[datetime.fromisoformat(r['utc']).astimezone(timezone(timedelta(hours=8))) for r in rows]
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    axes[0].plot(times,[int(r['memory_used_mib'])/1024 for r in rows],'.-',ms=3)
    axes[0].set_ylabel('Whole-device memory GiB')
    axes[1].plot(times,[int(r['utilization_pct']) for r in rows],'.-',ms=3,color='#cc8844')
    axes[1].set_ylabel('Whole-device sampled GPU %');axes[1].set_ylim(-3,103)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=timezone(timedelta(hours=8))))
    axes[1].set_xlabel('Beijing time, '+times[0].strftime('%Y-%m-%d'))
    fig.suptitle('GPU0 monitoring: device-wide measurements include other users; no ownership inferred')
    fig.tight_layout();fig.savefig(figures/'gpu-monitor.png',dpi=150);plt.close(fig)
    cpu=list(csv.DictReader((raw/'schedule-bench-02/timings.csv').open()))
    fig,ax=plt.subplots(figsize=(10,4));workloads=list(dict.fromkeys(r['workload'] for r in cpu))
    for i,mode in enumerate(('list','heap')):
        values=[[float(r['ms_per_step']) for r in cpu if r['workload']==w and r['mode']==mode] for w in workloads]
        ax.bar(np.arange(len(workloads))+(i-.5)*.36,[statistics.mean(v) for v in values],.36,
            yerr=[statistics.stdev(v) for v in values],label=mode,capsize=3)
    ax.set_xticks(range(len(workloads)),workloads);ax.set_yscale('log');ax.set_ylabel('CPU complete step ms (log scale)')
    ax.legend();ax.set_title('CPU schedule prototype, 5 alternating repeats; NOT GPU throughput')
    fig.tight_layout();fig.savefig(figures/'cpu-schedule.png',dpi=150);plt.close(fig)
    window_path=raw/'gpu-window-01/window.json'
    window=json.loads(window_path.read_text()) if window_path.exists() else dict(status='NOT_RUN',reason='No qualified idle GPU0 window launched')
    gpu=raw/'gpu-window-01'
    suites={name:gpu_tests(gpu/(name+'.xml'),window,name) for name in ('sum-tests','relu-tests','gpu-regression','schedule-gpu-tests')}
    training=[];initials={};protocols={};initial_errors=[]
    if gpu.exists():
        for path in sorted(gpu.glob('*/summary.json')):
            directory=path.parent
            cfg=directory/'config.json'
            if not cfg.exists():continue
            conf=json.loads(cfg.read_text());value=json.loads(path.read_text())
            if not all(k in conf for k in ('model','method','task')):continue
            if 'wall_s' not in value:continue
            instrumented=directory.name.startswith('trace-')
            phase=value.get('phases',{}).get('phase_fraction',{})
            training.append(dict(run=directory.name,model=conf['model'],method=conf['method'],task=conf['task'],
                schedule=conf.get('backward_schedule','list'),instrumented=instrumented,workers=conf['workers'],
                sum_native=conf['sum_native'],relu_native=conf['relu_native'],wall_s=value['wall_s'],
                samples_per_sec=value['samples_per_sec'],final_loss=value['final_loss'],
                forward_ms=phase.get('forward',0)*value.get('phases',{}).get('mean_step_s',0)*1000 if phase else None,
                backward_ms=phase.get('backward',0)*value.get('phases',{}).get('mean_step_s',0)*1000 if phase else None,
                update_ms=phase.get('update',0)*value.get('phases',{}).get('mean_step_s',0)*1000 if phase else None,
                source=str(directory)))
            if not instrumented:
                arrays=np.load(directory/'initial_weights.npz');model=conf['model']
                if model not in initials:
                    initials[model]={k:arrays[k].copy() for k in arrays.files}
                    protocols[model]=conf
                else:
                    ref=initials[model]
                    exact=set(arrays.files)==set(ref) and all(arrays[k].dtype==ref[k].dtype==np.float32 and np.array_equal(arrays[k],ref[k]) for k in arrays.files)
                    same=all(conf[k]==protocols[model][k] for k in ('indices','batch','dtype','seed','lr','dropout','tf32'))
                    if not exact or not same:initial_errors.append(dict(run=directory.name,weights_equal=exact,protocol_equal=same))
    write_csv(out/'training-runs.csv',training)
    groups=[]
    def group_key(r):
        return (r['model'],r['task'],r['method'],r['schedule'],r['run'].startswith('schedule-'),r['workers'],r['sum_native'],r['relu_native'])
    keys=sorted({group_key(r) for r in training if not r['instrumented']})
    for key in keys:
        model,task,method,schedule,optional,workers,sum_native,relu_native=key
        rr=[r for r in training if not r['instrumented'] and group_key(r)==key]
        wall,wall_sd=mean_sd([r['wall_s'] for r in rr]);speed,speed_sd=mean_sd([r['samples_per_sec'] for r in rr])
        groups.append(dict(model=model,task=task,method=method,schedule=schedule,optional=optional,workers=workers,
            sum_native=sum_native,relu_native=relu_native,repeats=len(rr),
            status='COMPLETE' if len(rr)>=3 else 'PARTIAL',wall_s_mean=wall,wall_s_sd=wall_sd,
            samples_per_sec_mean=speed,samples_per_sec_sd=speed_sd))
    write_csv(out/'training-summary.csv',groups)
    ratios=[]
    for model in ('donkey','spatial'):
        for task in ('compute','io','phases'):
            complete=[r for r in groups if r['model']==model and r['task']==task and not r['optional'] and r['status']=='COMPLETE']
            assert len({r['method'] for r in complete})==len(complete),'Do not pool different variants under one method label'
            selected={r['method']:r for r in groups if r['model']==model and r['task']==task and not r['optional'] and r['status']=='COMPLETE'}
            for new,base in (('I','A'),('P','I'),('R','I'),('S','I'),('J','I'),('J','A'),('J','E'),('J','E-prefetch'),('E-prefetch','E')):
                if new not in selected or base not in selected:continue
                x,y=selected[new],selected[base]
                ratios.append(dict(model=model,task=task,new=new,baseline=base,
                    throughput_ratio=x['samples_per_sec_mean']/y['samples_per_sec_mean'],
                    throughput_change_pct=(x['samples_per_sec_mean']/y['samples_per_sec_mean']-1)*100,
                    wall_speedup=y['wall_s_mean']/x['wall_s_mean'],
                    wall_reduction_pct=(1-x['wall_s_mean']/y['wall_s_mean'])*100))
            if selected:
                fig,ax=plt.subplots(figsize=(10,4));methods=list(selected)
                ax.bar(methods,[selected[m]['samples_per_sec_mean'] for m in methods],
                    yerr=[selected[m]['samples_per_sec_sd'] for m in methods],capsize=3)
                ax.set_ylabel('Samples/s, mean ± SD of independent runs');ax.set_title(model+' / '+task+' / uninstrumented matched GPU0')
                fig.tight_layout();fig.savefig(figures/f'training-{model}-{task}.png',dpi=150);plt.close(fig)
    write_csv(out/'training-ratios.csv',ratios)
    operator=[]
    for name,filename,method_key,wall_key in [('sum','timings.csv','method','full_sync_wall_us'),
                                            ('relu','raw.csv','mode','synchronized_host_us')]:
        path=gpu/(name+'-micro')/filename
        if not path.exists():continue
        records=list(csv.DictReader(path.open()))
        for case,method in sorted({(r['case'],r[method_key]) for r in records}):
            selected=[r for r in records if r['case']==case and r[method_key]==method]
            repeat_means=[];event_means=[]
            for repeat in sorted({r['repeat'] for r in selected}):
                rr=[r for r in selected if r['repeat']==repeat]
                repeat_means.append(statistics.mean(float(r[wall_key]) for r in rr))
                event_means.append(statistics.mean(float(r['event_span_us']) for r in rr))
            wall,sd=mean_sd(repeat_means);event,event_sd=mean_sd(event_means)
            operator.append(dict(operator=name,case=case,method=method,repeats=len(repeat_means),
                samples=len(selected),full_sync_wall_us=wall,full_sync_wall_sd_us=sd,
                event_span_us=event,event_span_sd_us=event_sd,
                scope='Uninstrumented full operation; CUDA Event span is not pure kernel time'))
    write_csv(out/'operator-timing-summary.csv',operator)
    allocations=[]
    for name in ('sum','relu'):
        path=gpu/(name+'-memory/allocations.csv')
        if path.exists():
            allocations.extend(dict(operator=name,**r) for r in csv.DictReader(path.open()))
    write_csv(out/'operator-allocation-records.csv',allocations)
    for name in ('sum','relu'):
        records=[r for r in operator if r['operator']==name]
        if not records:continue
        cases=sorted({r['case'] for r in records},key=int);methods=sorted({r['method'] for r in records})
        fig,ax=plt.subplots(figsize=(12,4));width=.8/len(methods)
        for i,method in enumerate(methods):
            selected={r['case']:r for r in records if r['method']==method}
            present=[(index,c) for index,c in enumerate(cases) if c in selected]
            ax.bar(np.array([index for index,c in present])+(i-(len(methods)-1)/2)*width,
                [selected[c]['full_sync_wall_us'] for index,c in present],width,
                yerr=[selected[c]['full_sync_wall_sd_us'] or 0 for index,c in present],label=method)
        ax.set_xticks(range(len(cases)),cases);ax.set_yscale('log');ax.set_ylabel('Full synchronized operator wall µs (log)')
        ax.set_xlabel('Predeclared case index');ax.legend(fontsize=8)
        ax.set_title(name+' / uninstrumented 3 repeats; no claim from kernel time alone')
        fig.tight_layout();fig.savefig(figures/(name+'-full-operation.png'),dpi=150);plt.close(fig)
    profiled=[]
    for name in ('sum','relu'):
        path=gpu/'offline-traces-01'/('trace-'+name+'-analysis')/'micro-kernels.csv'
        if not path.exists():continue
        records=list(csv.DictReader(path.open()))
        keys=set()
        for r in records:
            case=re.match(r'c(\d+)_r\d+_',r['case'])[1] if name=='sum' else r['case']
            r['actual_case']=case;keys.add((case,r['scope']))
        for case,scope in sorted(keys):
            selected=[r for r in records if r['actual_case']==case and r['scope']==scope]
            profiled.append(dict(operator=name,case=case,scope=scope,calls=len(selected),
                gpu_kernel_us_mean=statistics.mean(float(r['kernel_us']) for r in selected),
                kernels_per_call=statistics.mean(int(r['kernels']) for r in selected),
                launch_api_host_us_mean=statistics.mean(float(r['launch_api_us']) for r in selected),
                scope_note='Separate Nsight-instrumented run, not normal throughput'))
    write_csv(out/'profiled-kernel-summary.csv',profiled)
    # Strict trajectory comparisons are retained separately from same-state acceptance.
    trajectory=[]
    for row in training:
        if row['instrumented']:continue
        match=re.search(r'-r(\d+)$',row['run'])
        if not match:continue
        repeat=match[1]
        for reference in ('I','E'):
            target=gpu/(row['task']+'-'+row['model']+'-'+reference+'-r'+repeat)
            if not (target/'losses.jsonl').exists():continue
            actual=[json.loads(v)['loss'] for v in (Path(row['source'])/'losses.jsonl').read_text().splitlines()]
            old=[json.loads(v)['loss'] for v in (target/'losses.jsonl').read_text().splitlines()]
            if len(actual)!=len(old):continue
            error=np.abs(np.asarray(actual)-np.asarray(old));mask=error>1e-6+1e-5*np.abs(old)
            trajectory.append(dict(run=row['run'],reference=reference,steps=len(actual),
                loss_max_abs=float(error.max()),loss_failed_steps=int(mask.sum()),
                first_loss_failed_step=int(np.flatnonzero(mask)[0]+1) if mask.any() else None,
                atol=1e-6,rtol=1e-5,scope='Independent training trajectory, not common-state single step'))
    write_csv(out/'training-trajectory-errors.csv',trajectory)
    cpu_xml=sorted((report/'correctness').glob('regression-final-*.xml'))
    cpu_path=cpu_xml[-1] if cpu_xml else report/'correctness/regression-01.xml'
    summary=dict(status='FINAL' if a.final else 'INTERMEDIATE',monitor=monitor,window=window,gpu_tests=suites,
        cpu_regression=tests(cpu_path),cpu_regression_file=str(cpu_path),
        cpu_report_status_checks=tests(report/'correctness/report-status-01.xml'),
        cpu_allocation_hook_checks=tests(report/'correctness/operator-memory-host-02.xml'),
        cpu_schedule=json.loads((raw/'schedule-bench-02/summary.json').read_text()),
        cpu_sum_model=json.loads((raw/'cpu-sum-training-01/summary.json').read_text()) if (raw/'cpu-sum-training-01/summary.json').exists() else dict(status='NOT_RUN'),
        cpu_sum_trajectory=json.loads((raw/'cpu-sum-trajectory-01/summary.json').read_text()) if (raw/'cpu-sum-trajectory-01/summary.json').exists()
            else dict(status='PARTIAL' if (raw/'cpu-sum-trajectory-01/losses.jsonl').exists() else 'NOT_RUN'),
        protocol_check=dict(status='NOT_RUN' if not training else 'PASS' if not initial_errors else 'FAIL',errors=initial_errors,
            scope='Actual initial arrays and config equality, no hashes',runs=len(training)),
        training_completed_runs=len([r for r in training if not r['instrumented']]),
        operator_groups=len(operator),profiled_kernel_groups=len(profiled),
        operator_allocation_records=len(allocations),
        strict_training_trajectory_status='NOT_RUN' if not trajectory else 'FAIL' if any(r['loss_failed_steps'] for r in trajectory) else 'PASS',
        strict_training_trajectory_failures=sum(r['loss_failed_steps'] for r in trajectory),
        source_config=config,finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'summary.json',summary)
    status=('GPU窗口未执行' if window['status']=='NOT_RUN' else 'GPU窗口状态：'+window['status'])
    md=f'''# GPU0七小时研究：{'最终报告' if a.final else '阶段性汇总'}

{status}。实际监控样本{len(rows)}条，首尾跨度{monitor['observed_seconds']/3600:.4f}小时；
不是把任务预算直接当实测时长。最大采样间隔{monitor['max_gap_s']}秒，最高连续空闲计数{monitor['max_idle_streak']}。
物理GPU0 / `{config['uuid']}`，未切卡。

CPU最终回归：{summary['cpu_regression']}；heap调度CPU188项逐元素一致；CPU Sum算术模型训练结果见JSON。
CPU合成图收益不当作GPU或自动驾驶训练收益。原始失败和修正记录保留。

GPU Sum：{suites['sum-tests']}。
GPU ReLU：{suites['relu-tests']}。
GPU基线回归：{suites['gpu-regression']}。
新调度GPU：{suites['schedule-gpu-tests']}。

正式训练完成{summary['training_completed_runs']}次，只有至少3次独立重复的组才计算正式比值。
GPU未执行或组不完整时，不填造Kernel时间、吞吐、累计收益、与PyTorch差距或Nsight结果。
统计口径、协议复核和窗口每项完整命令见本目录summary.json、training CSV和服务器window.json。
Sum强消去超差即使符合CPU舍入模型也保留FAIL；不改容差，不能声称通用数值验收通过。

原始目录：`{raw}`。本次图像为原始CSV数据可视化，不是终端截图。
'''
    (out/'report.md').write_text(md)
    print(json.dumps(dict(status=summary['status'],monitor_samples=len(rows),window=window['status'],output=str(out))))


if __name__=='__main__':main()
