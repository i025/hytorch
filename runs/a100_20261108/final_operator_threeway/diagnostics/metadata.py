from common import *
work=json.loads((O/'raw/workload.json').read_text());rows=[]
for case in cases():
 r=dict(case,spec=json.dumps(case.get('spec',{})),dtype='float32',benchmark_input_values='synthetic fixed seed; actual training shapes only',output_preallocated_array=(case['op']=='gemm'),stream='current CUDA stream (default ptr=0 in this run)')
 if case['op']=='gemm':
  m,n,k=case['mnk'];r.update(a_shape=str((m,k)),a_strides_bytes=str((k*4,4)),b_shape=str((k,n)),b_strides_bytes=str((n*4,4)),actual_training_calls_per_step=sum((v['M'],v['N'],v['K'])==(m,n,k) for v in work['gemms']),actual_training_paths=';'.join(v['path'] for v in work['gemms'] if (v['M'],v['N'],v['K'])==(m,n,k)))
 else:
  s=case['spec'];n,c,h,w,kh,kw,sh,sw,ph,pw,oh,ow=dims(s)
  shape=(n,c,kh,kw,oh,ow) if case['op']=='col2im' else (n,c,h,w);x=layout(np.empty(shape,'f'),np,case['layout']);r.update(input_shape=str(x.shape),input_strides_bytes=str(x.strides),col_matrix_shape=str((n*oh*ow,c*kh*kw)),kernel=str((kh,kw)),stride=str((sh,sw)),padding=str((ph,pw)),output_elements=n*c*h*w if case['op']=='col2im' else '')
 rows.append(r)
write('WORKLOAD.csv',rows)
# independent timing decomposition summaries
samples=read('raw/IM2COL_LAYOUT_SAMPLES.csv');groups={}
for r in samples:groups.setdefault((r['case'],r['mode']),[]).append(r)
rows=[]
for (case,mode),v in groups.items():
 for metric in ('sync_us','submit_us','event_us'):
  z=np.array([float(x[metric]) for x in v]);rows.append(dict(case=case,mode=mode,metric=metric,mean=z.mean(),median=np.median(z),std=z.std(ddof=1),min=z.min(),max=z.max(),repeats=len(z),loops=5))
write('IM2COL_LAYOUT_RESULTS.csv',rows)
