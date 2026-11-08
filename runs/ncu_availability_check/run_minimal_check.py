"""Explicit, one-kernel probes; refuses an occupied physical GPU1."""
from pathlib import Path
import sys,subprocess,datetime,json,os
R=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
mode=sys.argv[1]
if mode=='basic':
 O=BASE/('recheck_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));O.mkdir(exist_ok=False)
 (BASE/'LATEST_RUN.txt').write_text(str(O)+'\n')
else:O=Path((BASE/'LATEST_RUN.txt').read_text().strip())
if (O/(mode+'.json')).exists():raise SystemExit('Refuse overwrite')
def capture(args):return subprocess.check_output(args,text=True)
uuid=capture(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader']).strip()
processes=capture(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'])
gpus=capture(['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,driver_version','--format=csv'])
resource=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpus=gpus,processes=processes)
(O/(mode+'_resources.json')).write_text(json.dumps(resource,indent=2))
if any(line.split(',')[0].strip()==uuid for line in processes.splitlines()):raise SystemExit('GPU1 occupied; stop without touching processes')
mem=capture(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).strip()
if any(int(v.strip())!=0 for v in mem.split(',')):raise SystemExit('GPU1 not idle: '+mem)
ncu='/opt/nvidia/nsight-compute/2025.2.1/ncu';py='/home/xieminhui_pubuse/hytorch-env/bin/python'
cmd=[ncu,'--clock-control','none','--pipeline-boost-state','dynamic','--cache-control','none','--launch-count','1','--export',str(O/mode)]
if mode in ('basic','gemm'):cmd+=['--set','basic']
else:
 for section in ['SpeedOfLight','LaunchStats','Occupancy','SchedulerStats','WarpStateStats']:cmd+=['--section',section]
if mode=='basic':cmd +=[py,'-c','import torch; a=torch.ones((512,512),device="cuda"); b=a+a; torch.cuda.synchronize(); print(b[0,0].item())']
else:cmd+=['--kernel-name-base','demangled','--kernel-name','regex:.*gemm_register.*',py,str(BASE/'gemm_minimal.py')]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',NVIDIA_TF32_OVERRIDE='0',CUDA_PATH='/usr/local/cuda',LD_LIBRARY_PATH='/usr/local/cuda/lib64'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
print('RUN',mode,str(O),flush=True)
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
try:
 result=subprocess.run(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120)
 data=dict(argv=cmd,started_utc=start,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=result.returncode)
 (O/('ncu_'+mode+'.stdout')).write_text(result.stdout);(O/('ncu_'+mode+'.stderr')).write_text(result.stderr)
 (O/('ncu_'+mode+'.log')).write_text('COMMAND: '+json.dumps(cmd)+'\nEXIT CODE: '+str(result.returncode)+'\nSTDOUT:\n'+result.stdout+'\nSTDERR:\n'+result.stderr)
except subprocess.TimeoutExpired as exc:
 (O/(mode+'.json')).write_text(json.dumps(dict(argv=cmd,status='timeout',seconds=120),indent=2));raise
(O/(mode+'.json')).write_text(json.dumps(data,indent=2));print(data);print(result.stdout[-15000:]);print(result.stderr)
if result.returncode==0 and (O/(mode+'.ncu-rep')).exists():
 exp=subprocess.run([ncu,'--import',str(O/(mode+'.ncu-rep')),'--page','raw','--csv'],cwd=R,env=env,text=True,capture_output=True)
 (O/(mode+'_metrics.csv')).write_text(exp.stdout);(O/(mode+'_export.stderr')).write_text(exp.stderr)
 (O/(mode+'_export.json')).write_text(json.dumps(dict(exit_code=exp.returncode),indent=2));print('EXPORT',exp.returncode)
