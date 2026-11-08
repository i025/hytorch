from pathlib import Path
import csv,shutil
O=Path(__file__).resolve().parents[1]
def read(n):return list(csv.DictReader((O/n).open()))
def write(n,rows):
 with (O/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
for n in ['RAW_TIMINGS.csv','ACCURACY.csv','RESULTS_ALL.csv','ALLOCATIONS.csv','DWEIGHT_RESULTS.csv','POOL_RESULTS.csv']:
 dst=O/'raw/formal_firstpass'/n;dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():shutil.copy2(O/n,dst)
 keep=[r for r in read('raw/formal_firstpass/'+n) if r['op'] not in ('dweight','pool_backward')]
 corrected=read('raw/interface_corrected_'+n);write(n,keep+corrected)
for n in ['PROFILE_SUMMARY.csv','KERNEL_DETAILS.csv','raw/PROFILE_KERNEL_SAMPLES.csv','raw/PROFILE_OPERATOR_SAMPLES.csv']:
 dst=O/'raw/formal_firstpass'/n;dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():shutil.copy2(O/n,dst)
 keep=[r for r in read('raw/formal_firstpass/'+n) if r['op'] not in ('dweight','pool_backward')];write(n,keep+read('raw/corrected_profile/'+n))
for name in ['GEMM_REAL_SHAPES.csv','raw/previous_gemm_baseline_summary.json','raw/previous_gemm_paired_comparison.csv']:
 src=O.parents[1]/'a100_20261106/fp32_fix_and_profile'/name;dst=O/'raw/historical_strict_gate'/Path(name).name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
