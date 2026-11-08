from common import *
rows=[];passed=True
for group,seeds,ss in [('legacy',[42,123,501],specs),('holdout',[2026,31415],specs+holdout)]:
 for spec in ss:
  for seed in seeds:
   for layout in ['contiguous','negative','slice']:
    x,g=arrays(spec,seed,layout);passed &= validate(x,g,*spec[3:],f'{group}/{spec[0]}/{seed}/{layout}',rows)
  print(group,spec[0],flush=True)
write(O/'NUMERICS.csv',rows);(O/'raw/longdouble.json').write_text(json.dumps(dict(nmant=int(np.finfo(np.longdouble).nmant),eps=float(np.finfo(np.longdouble).eps),passed=bool(passed)),indent=2))
assert passed,'D candidate fails frozen FP64 gate'
