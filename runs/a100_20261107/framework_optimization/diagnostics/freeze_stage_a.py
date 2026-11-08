from pathlib import Path
import shutil,json,hashlib,datetime
O=Path(__file__).resolve().parents[1]
if (O/'stage_a/FROZEN_SHA.json').exists():raise SystemExit('Refuse to overwrite frozen stage A')
for name in ['OPERATOR_COMPARISON.csv','CONV_FORWARD.csv','CONV_BACKWARD.csv','GEMM_COMPARISON.csv','OTHER_OPERATORS.csv','TRAINING_HOTSPOTS.csv','PROFILE_KERNELS.csv','OPERATOR_KERNELS.csv']:
 shutil.copy2(O/name,O/'stage_a'/name)
for name in ['cpu_B0.csv','cpu_B0.prof','cpu_B0.txt','tests.log']:shutil.copy2(O/'raw'/name,O/'stage_a'/name)
for src,dst in [('WORKLOAD_COMPLETE.json','WORKLOAD.json'),('ENVIRONMENT.json','ENVIRONMENT.json')]:shutil.copy2(O/src,O/'stage_a'/dst)
files=list((O/'stage_a').glob('*'))+list((O/'traces').glob('*'))
(O/'stage_a/FROZEN_SHA.json').write_text(json.dumps({'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}},indent=2))
