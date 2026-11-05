"""Durable CPU closeout before the seven-hour deadline; never starts GPU work."""
import argparse
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    a=p.parse_args();cfg=json.loads(a.schedule.read_text());raw=Path(cfg['raw']);report=Path(cfg['report'])
    at=datetime.fromisoformat(cfg['deadline_utc']).timestamp()-300
    state_path=raw/'publication-state.json'
    if a.wait:
        while time.time()<at:time.sleep(min(30,at-time.time()))
        # Load the final reviewed report code at closeout, after ongoing CPU analysis.
        os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),'--schedule',str(a.schedule.resolve())])
    if state_path.exists() and json.loads(state_path.read_text()).get('status')=='PUSHED':return
    status=dict(status='CLOSING',started_utc=datetime.now(timezone.utc).isoformat())
    def save():state_path.write_text(json.dumps(status,indent=2)+'\n')
    save()
    window=raw/'gpu-window-01/window.json'
    stopped=[]
    # Only terminate a controller whose recorded PID still identifies this experiment.
    if window.exists():
        recorded=json.loads(window.read_text())
        pid=recorded.get('controller_pid')
        if pid and psutil.pid_exists(pid):
            process=psutil.Process(pid)
            matches=any((Path(process.cwd())/v).resolve()==HERE/'run_window.py'
                for v in process.cmdline()[1:] if v.endswith('run_window.py'))
            if matches:
                children=process.children(recursive=True)
                for child in reversed(children):child.terminate();stopped.append(child.pid)
                process.terminate();stopped.append(pid)
                _,alive=psutil.wait_procs(children+[process],timeout=20)
                for child in alive:child.kill()
    preparation=json.loads((report/'environment/preparation.json').read_text())
    pid=preparation['watch_pid']
    if psutil.pid_exists(pid):
        process=psutil.Process(pid)
        if 'watch_gpu0.py' in ' '.join(process.cmdline()):
            process.terminate();_,alive=psutil.wait_procs([process],timeout=10)
            for value in alive:value.kill()
            stopped.append(pid)
    stop=dict(stopped_utc=datetime.now(timezone.utc).isoformat(),our_stopped_pids=stopped,
        reason='Reserve final five minutes for CPU archival and Git publication; GPU work cutoff was thirty minutes before deadline',
        gpu_context='No context held by monitor/finalizer; GPU children closed before CPU reporting')
    (raw/'monitor-end.json').write_text(json.dumps(stop,indent=2)+'\n')
    controls=report/'environment'/('final-control-'+datetime.now(timezone.utc).strftime('%H%M%S'))
    controls.mkdir(exist_ok=False)
    for name in ('schedule.json','gpu-plan.json','optional-gpu-tasks.json','monitor-end.json','monitor-state.json','trace-analysis-status.json'):
        if (raw/name).exists():shutil.copyfile(raw/name,controls/name)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        MPLCONFIGDIR='/tmp/hytorch-gpu0-7h-mpl',GIT_TERMINAL_PROMPT='0',GIT_ASKPASS='/bin/false')
    label='final-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    subprocess.run([sys.executable,str(HERE/'summarize.py'),'--schedule',str(a.schedule.resolve()),
        '--label',label,'--final'],cwd=ROOT,env=env,check=True)
    sources=[p for p in raw.iterdir() if p.is_dir() and not p.name.startswith('.')]
    destination=report/'correctness'/('final-raw-'+datetime.now(timezone.utc).strftime('%H%M%S'))
    if sources:
        subprocess.run([sys.executable,str(ROOT/'docs/experiments/fused-sgd-cuda/scripts/archive_results.py'),
            '--destination',str(destination),*[str(p) for p in sources]],cwd=ROOT,env=env,check=True)
    summary=json.loads((report/'benchmarks'/label/'summary.json').read_text())
    monitor=summary['monitor'];gpu=summary['window'];suites=summary['gpu_tests']
    start=datetime.fromisoformat(monitor['first_utc']).astimezone(timezone(timedelta(hours=8)))
    end=datetime.fromisoformat(monitor['last_utc']).astimezone(timezone(timedelta(hours=8)))
    def test_text(name):
        value=suites[name]
        return value['status'] if value['status'] in ('NOT_RUN','PARTIAL') else f"{value['passed']} passed / {value['failures']} failed / {value['errors']} errors / {value['skipped']} skipped"
    readme=f'''# GPU0自主性能研究：最终报告

任务预算北京时间2026-11-06 00:18:50–07:18:50。GPU计算截止06:48:50；
最后5分钟停止监控、归档和提交。实际监控{monitor['observed_seconds']/3600:.4f}小时、{monitor['samples']}条，
从{start.isoformat()}到{end.isoformat()}，最大采样间隔{monitor['max_gap_s']}秒。
没有将七小时预算写成七小时实测监控，准备时段也没有冒充持续监测。

## GPU生命周期与实验状态

物理GPU0，UUID `{cfg['uuid']}`。最高连续空闲计数{monitor['max_idle_streak']}；
满足连续空闲要求的采样{monitor['qualified_samples']}条，其中GPU截止前{monitor['qualified_before_gpu_cutoff']}条。
利用率为0但外部计算进程仍驻留的采样{monitor['occupied_zero_utilization_samples']}条，未据此抢占。
窗口状态 **{gpu['status']}**。所有GPU实验只允许这一个UUID，不换卡，不使用其他七张GPU。
没有独占租约，外部任务/资源冲突即停。窗口未执行时，不适用“完成同窗口实验/释放Context”，
因为本轮没有创建GPU计算Context；若已执行，窗口起止、子进程退出和release记录在汇总JSON与原始window.json。
本轮监控及GPU控制进程已按预算停止，停止记录见服务器monitor-end.json。

| 项目 | 实际状态 |
|---|---|
| Native Sum GPU测试 | {test_text('sum-tests')} |
| Native ReLU Backward GPU测试 | {test_text('relu-tests')} |
| 已有原生GPU回归 | {test_text('gpu-regression')} |
| 新heap调度GPU | {test_text('schedule-gpu-tests')} |
| 未插桩训练运行记录 | {summary['training_completed_runs']}次；不足三次的组标PARTIAL，不计算正式收益 |

Sum超差样本即使能由FP32顺序解释，也保留FAIL；完整CuPy/PyTorch/FP64/CPU模型误差若执行则见原始sum-diagnosis。
只对通过配置测性能，ReLU独立验收，S/J还需真实数据共同状态检查。没有放宽数值阈值。
纯Kernel时间来自API correlationId与同线程NVTX，不把Event span、提交API时间或覆盖率称为Kernel核时/GPU利用率。

## CPU研究与唯一新原型

最终主机集合：{summary['cpu_regression']}。此前210项及定向集合各次记录均保留，不能相加重复计数。
另有CPU归档状态测试：{summary['cpu_report_status_checks']}，验证未启动/被中断/失败与跳过的区分。
算子内存跟踪回调/注册生命周期CPU测试：{summary['cpu_allocation_hook_checks']}；不代表GPU已执行。
heap调度在CNN/Transformer的188项比较逐元素一致，保持generation优先级和相同LIFO平局顺序；默认list。
五次交替CPU计时中，256分支合成图吞吐均值约提高52.17%，CNN只变化0.3%–1.3%，不宣称稳定CNN收益。
该批前段与回归共享CPU，微小差异不能作为收益证据。初始测试参考dtype和CSV脚本失败保留，生产数学没有因此改变。
独立拓扑补测12组、24项Loss/输入梯度比较逐元素一致：宽64/256的树形与扇出图CPU吞吐提高约10%–55%，
ready frontier仅1–2的链式/残差图变化约-1.53%到+1.22%，不能宣称稳定收益。
这说明收益取决于待调度队列宽度；图节点数本身不足以预测收益。详见[拓扑研究](graph-topologies.md)。
独立CPU tracemalloc预热后36次记录中，宽度256的树形/扇出图峰值增加约22.3–22.8KiB；不是GPU显存。
原生Sum的CPU算术模型真实图像三步训练156项、逐归约192项比较通过；仍不是GPU验收。
256step CPU Sum算术诊断：{summary['cpu_sum_trajectory']}。
其CPU长轨迹和共同状态比较不作为CUDA执行或加速证据。
另有[固定种子条件数诊断](sum-conditioning.md)：320项FP64比较，普通输入160项全通过，
强消去输入68项超差原样保留（CPU模型31项、NumPy37项）；不能据此豁免GPU数值失败。
主机Sum轴规划约4–14µs，配置上下文约3.48µs；已避免基准给native每次重复计入配置成本。
纯主机ABI版本调用约0.135µs，不等于CUDA提交耗时。

没有新增CUDA Kernel。本轮只有一个新的通用heap调度原型，保持已有GPU Kernel/优化器/数据语义。
继续依赖CuPy的MatMul梯度、BMM、Norm/Softmax、布局物化等已列候选；没有同时开发多个未经验证的算子。

## 端到端、累计收益与PyTorch差距

若窗口执行且组完整，本轮匹配的I/A、P/I、R/I、S/I、J/I、J/A、J/E、J/E-prefetch结果见
[逐次训练与统计](benchmarks/{label}/)。所有原始值、SD和不同计时口径分开；
吞吐变化、耗时下降与加速比各列报告，不跨历史窗口拼接。
若GPU未执行，则异步预取的真实GPU训练收益、Sum/ReLU端到端收益、与PyTorch差距变化均 **NOT_RUN**。
CPU数据准备或合成图加速不能替代上述结论。Loss下降不证明自动驾驶能力。

## 负结果、未完成项和下一轮

Sum长归约强消去的已有数值负样本保留；原型不能因CPU模型一致而宣称通用精度全部通过。
GPU没有空闲或时间不足的任务保持NOT_RUN/PARTIAL，其他任务重新占用则保留已完成部分并停止。
heap对普通短链CNN缺乏稳定收益证据，不改默认；额外显存/Kernel数量不因此减少。
下一轮优先：已有Sum/ReLU未完成的真实GPU验收；再依据新Trace决定MSE/激活链融合；
若I/O占主导才研究pinned H2D双缓冲。仅靠CPU不能确定应重写哪个CUDA Kernel。

## 交付

[时间线](timeline.md)、[GPU监控CSV](gpu-monitor.csv)、[源码分析](source-analysis.md)、
[候选依据](optimization-candidates.md)、[环境](environment.md)、[全部命令](commands.md)、
[本轮汇总](benchmarks/{label}/report.md)、[完整数据索引](data-index.md)、[全部图](figures/{label}/)。
原始数据目录`{raw}`；Nsight/SQLite、完整数组和动态库留服务器，适合Git的CSV/JSON/XML/日志和PNG上传。
图像均为实际记录可视化，不冒充截图。
实验分支`experiment/gpu0-autonomous-7h`。本轮停止，不开始下一阶段。
'''
    (report/'README.md').write_text(readme)
    index=f'''# 最终数据索引

服务器完整根目录：`{raw}`。

- [完整逐文件索引]({destination.relative_to(report)}/file-index.csv)：全部CPU/GPU原始目录、服务器二进制路径、Git可移植副本。
- [最终分析](benchmarks/{label}/)、[最终图](figures/{label}/)、[实时监控完整CSV](gpu-monitor.csv)。
- [CPU初始原始索引](correctness/cpu-raw-01/file-index.csv)、[日志](logs)、[环境](environment)、[测试XML](correctness)。
- GPU若执行，原始`gpu-window-01`含完整window.json、每次命令和日志、测试数组、训练权重、逐step Loss/耗时、Nsight。
- Nsight的`.nsys-rep`/SQLite及所有NPZ/动态库只留服务器，索引不计算哈希。
- `monitor-end.json`和`publication-state.json`在服务器根目录，后者记录最终commit及推送状态。

未执行的文件不提前宣称已生成。原始失败和成功记录均保留，GPU未执行不填造性能CSV或图。
'''
    (report/'data-index.md').write_text(index)
    with (report/'timeline.md').open('a') as f:
        f.write('\n- '+datetime.now(timezone(timedelta(hours=8))).isoformat()+': 按总预算完成自动收尾，停止本轮监控/计算并发布已完成记录。\n')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch=='experiment/gpu0-autonomous-7h','Do not change or publish a different branch'
    subprocess.run(['git','add','--',str(report)],cwd=ROOT,env=env,check=True)
    subprocess.run(['git','commit','-m','Archive measured GPU0 window and seven-hour CPU research results'],cwd=ROOT,env=env,check=True)
    status.update(status='COMMITTED',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());save()
    remaining=max(1,int(datetime.fromisoformat(cfg['deadline_utc']).timestamp()-time.time())-5)
    try:
        result=subprocess.run(['git','-c','pack.threads=2','-c','credential.interactive=false','push','origin',branch],
            cwd=ROOT,env=env,capture_output=True,text=True,timeout=min(120,remaining))
        output=result.stdout+result.stderr;code=result.returncode
    except subprocess.TimeoutExpired as error:
        parts=[v.decode(errors='replace') if isinstance(v,bytes) else v or '' for v in (error.stdout,error.stderr)]
        output=''.join(parts)+'\nPush stopped at task publication time limit.\n';code=124
    (raw/'final-push.log').write_text(output)
    status.update(status='PUSHED' if code==0 else 'PUSH_FAILED',push_returncode=code,
        finished_utc=datetime.now(timezone.utc).isoformat(),report_label=label);save()
    print(json.dumps(status),flush=True)
    sys.exit(code)


if __name__=='__main__':main()
