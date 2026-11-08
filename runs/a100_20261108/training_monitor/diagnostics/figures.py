from pathlib import Path
import csv,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parents[1]
def read(n):return list(csv.DictReader((O/n).open()))
gpu=[r for r in read('GPU_SAMPLES.csv') if r['tag']=='trend' and r['inside_train']=='True']
cpu=[r for r in read('CPU_SAMPLES.csv') if r['tag']=='trend' and r['inside_train']=='True']
steps=read('STEP_TIMINGS.csv');over=read('MONITOR_OVERHEAD.csv');summary=read('TIMING_SUMMARY.csv');profile=read('PROFILE_PHASES.csv')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
foot='A100-PCIE-40GB GPU 1 | DonkeyNet FP32, batch=4, 256 synthetic samples | F1+O1, legacy SGD | NVML/psutil period=0.2s'
def save(fig,name,note):
 fig.text(.01,.012,foot+'\n'+note,fontsize=8);fig.tight_layout(rect=(0,.065,1,1));fig.savefig(O/'figures'/f'{name}.png',dpi=190);fig.savefig(O/'figures'/f'{name}.svg');plt.close(fig)
def timeaxis(rows):return [(int(r['monotonic_ns'])-int(r['train_start_ns']))/1e9 for r in rows]
def values(rows,k,div=1):return [float(r[k])/div if r.get(k) else np.nan for r in rows]
x=timeaxis(gpu)
fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
ax[0].plot(x,values(gpu,'gpu_utilization_pct'),'o-',ms=3,label='NVML GPU busy (%)');ax[0].set_ylim(0,100);ax[0].set_ylabel('%');ax[0].legend(loc='upper right')
for k,label in [('gpu_used_bytes','Whole GPU used'),('process_gpu_used_bytes','Current PID reported by NVML')]:ax[1].plot(x,values(gpu,k,2**20),label=label)
ax[1].set_ylim(bottom=0);ax[1].set_ylabel('MiB');ax[1].legend(loc='upper right')
for k,label in [('cupy_active_bytes','CuPy active (sampled)'),('cupy_reserved_bytes','CuPy pool reserved (sampled)')]:ax[2].plot(x,values(gpu,k,2**20),'o-',ms=3,label=label)
ax[2].set_ylim(bottom=0);ax[2].set_ylabel('MiB');ax[2].set_xlabel('Seconds since training start');ax[2].legend(loc='upper right')
fig.suptitle('GPU utilization and distinct memory accounting scopes')
save(fig,'01_gpu_utilization_memory','30 epochs / 1920 steps, 38 in-training samples. Whole-device telemetry is not per-process utilization. Sampled maxima are not exact peaks.')
fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True)
ax[0].plot(x,values(gpu,'gpu_temperature_c'),'o-',ms=3,color='#d55e00');ax[0].set_ylim(0,60);ax[0].set_ylabel('Temperature (C)')
ax[1].plot(x,values(gpu,'gpu_power_w'),'o-',ms=3,color='#0072b2');ax[1].set_ylim(0,60);ax[1].set_ylabel('Power draw (W)');ax[1].set_xlabel('Seconds since training start')
fig.suptitle('Whole-GPU temperature and power; clocks not locked')
save(fig,'02_gpu_temperature_power','30 epochs / 1920 steps; 0.2s samples plus device-defined sensor windows. A short cool run does not establish long-run thermal behavior.')
fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True);xx=timeaxis(cpu)
ax[0].plot(xx,values(cpu,'system_cpu_pct'),'o-',ms=3,label='System CPU utilization');ax[0].set_ylim(0,100);ax[0].set_ylabel('System %');ax[0].legend()
ax[1].plot(xx,values(cpu,'process_cpu_pct'),'o-',ms=3,label='Process CPU: 100% = one core');ax[1].axhline(100,color='gray',ls='--');ax[1].set_ylim(0,310);ax[1].set_ylabel('Process %');ax[1].set_xlabel('Seconds since training start');ax[1].legend()
fig.suptitle('CPU scope matters: shared host vs current training process')
save(fig,'03_cpu_utilization','30 epochs / 1920 steps; first CPU-percent sample is null (priming), 37 valid samples. System CPU includes unrelated workloads.')
full=[r for r in steps if r['tag']=='formal' and r['monitor_components']=='all'];lat=[r for r in steps if r['tag']=='latency']
fig,ax=plt.subplots(1,2,figsize=(13,5))
y=np.array(values(full,'host_iteration_s',.001));z=np.array(values(full,'step_gpu_interval_s',.001));bins=np.geomspace(min(y.min(),z.min())*.95,max(y.max(),z.max())*1.05,45)
ax[0].hist(y,bins=bins,histtype='step',label='Host iteration (submission)');ax[0].hist(z,bins=bins,histtype='step',label='CUDA step stream interval');ax[0].set_xscale('log');ax[0].set_yscale('log');ax[0].set_xlabel('ms (log)');ax[0].set_ylabel('Count (log)');ax[0].legend(fontsize=8);ax[0].set_title('12 x 192 throughput iterations; all outliers retained')
latency=values(lat,'step_wall_s',.001);ax[1].plot(range(len(latency)),latency,'o-',ms=2);ax[1].axhline(np.mean(latency),ls='--',color='#d55e00',label=f'Mean {np.mean(latency):.3f} ms');ax[1].set_ylim(bottom=0);ax[1].set_xlabel('Step ID');ax[1].set_ylabel('Synchronized completed step (ms)');ax[1].legend();ax[1].set_title('Separate latency diagnostic: one sync at step end')
save(fig,'04_step_time_distribution','3 epochs / 192 steps per run. Throughput host iteration is not completion latency; CUDA events are not pure kernel time. n=2304 / 192.')
phases=['data_load','data_prepare','h2d','forward','loss','backward','optimizer'];colors=['#999999','#bbbbbb','#e69f00','#0072b2','#56b4e9','#d55e00','#009e73','#cc79a7']
def mean(metric):return float(next(r['mean_s'] for r in summary if r['tag']=='formal_all' and r['metric']==metric))*1000
cpuvals=[mean(p+'_cpu_s') for p in phases];evvals=[mean(p+'_gpu_event_s') if p not in ['data_load','data_prepare'] else 0 for p in phases]
kvals=[float(next(r['kernel_us_per_step'] for r in profile if r['phase']==p))/1000 for p in phases]
cpuvals.append(max(0,mean('host_iteration_s')-sum(cpuvals)));evvals.append(max(0,mean('step_gpu_interval_s')-sum(evvals)));kvals.append(0)
fig,ax=plt.subplots(figsize=(12,5));left=np.zeros(3)
for j,p in enumerate(phases+['unassigned interval']):
 v=[cpuvals[j],evvals[j],kvals[j]];ax.barh([2,1,0],v,left=left,label=p,color=colors[j]);left+=v
ax.set_yticks([2,1,0],['CPU host iteration (n=2304)','GPU event interval (n=2304)','CUPTI kernel sum (n=20)']);ax.set_xlabel('Mean ms per step (different measurements, NOT additive)');ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.3),fontsize=9);ax.set_xlim(left=0)
save(fig,'05_phase_breakdown','Normal full monitor vs separate monitor-disabled Nsight diagnostic. Data-load/prepare have no GPU events; event gaps include CPU idle/submission time.\nCUPTI row excludes H2D memcpy and profiler timing cannot replace formal throughput. Backward is largest in all three views.')
fig,ax=plt.subplots(1,2,figsize=(13,5));modes=['disabled','timing','resources','all'];labels=['Disabled','Timing only','NVML/CPU only','All monitoring']
for j,metric in enumerate(['training_window_s','wall_s']):
 rr=[next(r for r in over if r['mode']==m and r['metric']==metric) for m in modes];vv=[float(r['mean_s']) for r in rr];sd=[float(r['std_s']) for r in rr]
 ax[0].bar(np.arange(4)+(j-.5)*.33,vv,.33,yerr=sd,capsize=3,label=['Training loop + final sync','Including setup/finalize/export'][j],color=['#56b4e9','#0072b2'][j])
 dd=np.array([float(r['paired_delta_mean_s'])*1000 for r in rr]);lo=np.array([float(r['paired_delta_ci95_low_s'])*1000 for r in rr]);hi=np.array([float(r['paired_delta_ci95_high_s'])*1000 for r in rr])
 ax[1].errorbar(np.arange(4)+(j-.5)*.15,dd,yerr=np.array([dd-lo,hi-dd]),fmt='o',capsize=4,label=['Training window','Whole monitor use'][j])
ax[0].set_ylabel('Complete training run (s)');ax[0].set_ylim(bottom=0);ax[0].legend(fontsize=8);ax[1].axhline(0,color='black',ls='--');ax[1].set_ylabel('Paired difference vs disabled (ms)');ax[1].legend(fontsize=8)
for a in ax:a.set_xticks(range(4),labels,rotation=15)
fig.suptitle('Monitoring overhead: 12 balanced paired runs, identical data/state/boundaries')
save(fig,'06_monitor_overhead','3 epochs / 192 steps per run. Left error bars: sample SD. Right: paired t 95% CI (df=11), sequential shared-server data.\nResource-only CI crosses zero; do not claim it accelerates training. Optional allocator hook is OFF in these formal comparisons.')
print('Saved 6 PNG + 6 SVG figures')
