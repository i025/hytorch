from common import *
rows=[]
for s in specs():
 rng=np.random.default_rng(SEED);x=rng.normal(size=s['x']).astype('f');a=FC.im2col_array(x,s['k'],s['s'],s['p'],True);b=matrix(x,s);assert np.array_equal(a,b)
 col=window(x,s).copy();a=FC.col2im_array(col,s['x'],s['k'],s['s'],s['p'],False);ref=fold_ref(col,s);e=accuracy(a,ref);assert e['mixed_fail_count']==0
 rows.append(dict(id=s['id'],**e))
write('raw/cpu_reference_checks.csv',rows)
np.show_config()
print('CPU independent reference checks: 12 geometries PASS; GPU not initialized')
