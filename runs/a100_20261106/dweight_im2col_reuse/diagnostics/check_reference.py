from pathlib import Path
import csv,json,sys
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from hytorch.functions_conv import im2col_array
O=Path('runs/a100_20261106/dweight_im2col_reuse');examples=json.loads((O/'raw/failure_examples.json').read_text());rows=[]
specs={'conv1':((4,3,66,200),24,(5,5),(2,2),(0,0)),'conv2':((4,24,31,98),36,(5,5),(2,2),(0,0))}
for e in examples:
 if e['layout']!='contiguous':continue
 shape,oc,kernel,stride,pad=specs[e['case']];n,c,h,w=shape;kh,kw=kernel;sh,sw=stride;ph,pw=pad;oh=(h+2*ph-kh)//sh+1;ow=(w+2*pw-kw)//sw+1
 rng=np.random.default_rng(e['seed']);x=rng.normal(size=shape).astype(np.float32);g=rng.normal(size=(n,oc,oh,ow)).astype(np.float32)
 col=im2col_array(x,kernel,stride,pad,to_matrix=False);o,i,ky,kx=e['index']
 # Independent CPU scatter expansion + long-double reduction at the failing coordinate.
 a=g[:,o,:,:].ravel().astype(np.longdouble);b=col[:,i,ky,kx,:,:].ravel().astype(np.longdouble);ref=float(np.sum(a*b,dtype=np.longdouble));error=abs(ref-e['reference64'])
 assert error<1e-10
 rows.append(dict(case=e['case'],seed=e['seed'],index=e['index'],comparison=e['comparison'],cpu_longdouble_reference=ref,gpu_fp64_reference=e['reference64'],reference_abs_diff=error,old_abs_error=abs(e['A']-ref),new_abs_error=abs(e['B']-ref)))
with (O/'raw/independent_reference.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(rows)
