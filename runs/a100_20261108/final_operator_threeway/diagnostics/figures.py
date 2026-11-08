from pathlib import Path
import csv,collections,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
O=Path(__file__).resolve().parents[1]
def read(n):return list(csv.DictReader((O/n).open()))
rs=read('RESULTS_ALL.csv');agg=read('OPERATOR_AGGREGATE.csv');raw=read('RAW_TIMINGS.csv');acc=read('ACCURACY.csv');prof=read('PROFILE_SUMMARY.csv')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','figure.dpi':150})
colors={'numpy':'#6c7b8b','cupy':'#e69f00','native':'#0072b2','shared_old':'#b05a7a','register_old':'#009e73'}
labels={'gemm':'GEMM','im2col':'Im2col','col2im':'Col2im','conv':'Conv forward','dweight':'dWeight*','pool_forward':'MaxPool forward','pool_backward':'MaxPool backward*'}
ops=['gemm','im2col','col2im','conv','dweight','pool_forward','pool_backward'];modes=['numpy','cupy','native']
idx={(r['op'],r['case'],r['scope'],r['mode'],r['metric']):r for r in rs}
def get(op,case,mode,metric='sync_us',scope='array'):return idx[(op,case,scope,mode,metric)]
def save(fig,n,foot):
 fig.text(.01,.012,foot,fontsize=8);fig.tight_layout(rect=(0,.055,1,1));fig.savefig(O/'figures'/f'{n}.png',dpi=190);fig.savefig(O/'figures'/f'{n}.svg');plt.close(fig)
def agg_sd(op,mode,scope='array'):
 rr=[r for r in raw if r['op']==op and r['mode']==mode and r['scope']==scope];v=collections.defaultdict(list)
 for r in rr:v[int(r['repeat'])].append(float(r['sync_us']))
 return np.std([np.mean(vv) for vv in v.values()],ddof=1)
fig,axs=plt.subplots(1,2,figsize=(15,6))
for ax,scope in zip(axs,['array','function']):
 for j,mode in enumerate(modes):
  vals=[next(float(r[mode+'_us']) for r in agg if r['op']==op and r['scope']==scope) for op in ops];sd=[agg_sd(op,mode,scope) for op in ops]
  ax.bar(np.arange(7)+(j-1)*.24,vals,.24,label=mode,color=colors[mode],yerr=sd,capsize=2)
 ax.set_xticks(range(7),[labels[o] for o in ops],rotation=35,ha='right');ax.set_yscale('log');ax.set_ylabel('Mean sync-complete time (us), log');ax.set_title(scope+' interface | resident inputs');ax.legend()
fig.suptitle('NumPy CPU (4 BLAS threads) vs CuPy / Native on A100-PCIE-40GB')
save(fig,'01_threeway_latency','Equal weight per case; 36 GEMM / 14 im2col-conv-dW / 36 col2im / 4 pool cases. n=18 (GEMM array n=30).\nError bars: SD of per-repeat case means. * Hybrid cuBLAS dWeight / CuPy scatter pool backward. No H2D/D2H.')
fig,ax=plt.subplots(figsize=(11,5))
for j,scope in enumerate(['array','function']):
 vals=[next(float(r['native_vs_cupy']) for r in agg if r['op']==op and r['scope']==scope) for op in ops]
 # descriptive propagation of repeat variability, not a formal independent CI
 sd=[]
 for op,v in zip(ops,vals):
  rr=next(r for r in agg if r['op']==op and r['scope']==scope);sd.append(v*np.hypot(agg_sd(op,'cupy',scope)/float(rr['cupy_us']),agg_sd(op,'native',scope)/float(rr['native_us'])))
 ax.bar(np.arange(7)+(j-.5)*.34,vals,.34,yerr=sd,capsize=3,label=scope,color=['#0072b2','#56b4e9'][j])
ax.axhline(1,color='black',ls='--');ax.set_ylim(bottom=0);ax.set_xticks(range(7),[labels[o] for o in ops],rotation=15);ax.set_ylabel('CuPy time / Native time (>1 favors Native)');ax.legend();ax.set_title('Matched sync-complete time: A100 vs A100')
save(fig,'02_native_vs_cupy','Ratio of case-weighted means; same scopes / n as Fig.1. Error bars describe repeat variability, not cross-case heterogeneity.\nAll per-shape regressions remain in SPEEDUP_SUMMARY.csv; dWeight / pool backward are hybrid implementations.')
fig,axs=plt.subplots(1,2,figsize=(13,5));sizes=[256,512,1024,2048,4096]
for mode in ['numpy','cupy','shared_old','register_old','native']:
 vals=[get('gemm',f'ext_{n}_{n}_{n}',mode) for n in sizes]
 axs[0].errorbar(sizes,[float(v['mean']) for v in vals],yerr=[float(v['std']) for v in vals],marker='o',label=mode,color=colors[mode],capsize=2)
 if mode!='numpy':
  vals=[next(r for r in prof if r['op']=='gemm' and r['case']==f'ext_{n}_{n}_{n}' and r['scope']=='array' and r['mode']==mode) for n in sizes]
  axs[1].errorbar(sizes,[float(v['kernel_us_mean']) for v in vals],yerr=[float(v['kernel_us_std']) for v in vals],marker='o',label=mode,color=colors[mode],capsize=2)
for ax,title in zip(axs,['Normal calls: sync-complete, n=30','Separate Nsight Systems: kernel sum, n=15']):
 ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(sizes,[str(n) for n in sizes]);ax.set_xlabel('M=N=K (FP32)');ax.set_ylabel('Mean time (us), log');ax.set_title(title);ax.legend(fontsize=8)
save(fig,'03_gemm_size_curve','NumPy: CPU / OpenBLAS 4 threads. GPU: A100-PCIE-40GB, TF32 off. Preallocated output. Error bars = sample SD.\nOld register dispatch can fall back to Shared; final dispatch requires >=128 register CTAs. These are newly measured frozen libraries.')
import sys
sys.path.insert(0,str(Path(__file__).parent));from common import specs,dims
fig,axs=plt.subplots(1,2,figsize=(13,5))
for ax,op in zip(axs,['im2col','col2im']):
 ss=specs();xx=[]
 for s in ss:
  n,c,h,w,kh,kw,sh,sw,ph,pw,oh,ow=dims(s);xx.append(n*c*kh*kw*oh*ow)
 order=np.argsort(xx)
 for mode in modes:
  vv=[get(op,s['id']+('_contiguous' if op=='im2col' else '_training'),mode) for s in ss]
  ax.errorbar(np.array(xx)[order],np.array([float(v['mean']) for v in vv])[order],yerr=np.array([float(v['std']) for v in vv])[order],marker='o',label=mode,color=colors[mode],capsize=2)
 for i,s in enumerate(ss):ax.annotate(s['id'],(xx[i],float(get(op,s['id']+('_contiguous' if op=='im2col' else '_training'),'native')['mean'])),fontsize=7,rotation=25)
 ax.set_xscale('log');ax.set_yscale('log');ax.set_title(op+' | resident array, sync-complete');ax.set_xlabel('Col tensor elements');ax.set_ylabel('Mean time (us), log');ax.legend()
save(fig,'04_im2col_col2im_scale','CPU 4 BLAS threads / GPU A100. n=18, SD error bars. Im2col: contiguous NCHW -> contiguous (T,D); col2im: training-strided 6D -> NCHW.\nGeometry changes along x; this is not a single-variable scaling fit. All contiguous/negative/sliced cases remain in CSV.')
fig,axs=plt.subplots(1,2,figsize=(14,5));shapes=[(240,64,432),(72,64,576),(1024,1024,1024),(2048,2048,2048),(4096,4096,4096)]
for j,mode in enumerate(['shared_old','register_old','native']):
 vs=[get('gemm',('real_' if m!=n else 'ext_')+f'{m}_{n}_{k}',mode) for m,n,k in shapes]
 axs[0].bar(np.arange(5)+(j-1)*.25,[float(v['mean']) for v in vs],.25,yerr=[float(v['std']) for v in vs],capsize=2,label=mode,color=colors[mode])
axs[0].set_xticks(range(5),['240x64x432','72x64x576','1024^3','2048^3','4096^3'],rotation=20);axs[0].set_yscale('log');axs[0].set_ylabel('Sync-complete mean (us), log');axs[0].legend();axs[0].set_title('GEMM frozen-library A/B, n=30')
for j,mode in enumerate(['cupy','native']):
 oo=['im2col','col2im','dweight'];vs=[next(float(r[mode+'_us']) for r in agg if r['op']==op and r['scope']=='array') for op in oo]
 axs[1].bar(np.arange(3)+(j-.5)*.35,vs,.35,yerr=[agg_sd(o,mode) for o in oo],capsize=2,color=colors[mode],label=mode)
axs[1].set_xticks(range(3),['Im2col','Col2im','dWeight (hybrid)']);axs[1].set_ylabel('Sync-complete mean (us)');axs[1].set_ylim(bottom=0);axs[1].legend();axs[1].set_title('Original CuPy vs final paths, n=18')
save(fig,'05_optimization_ab','All bars: newly measured on the same A100; resident array interfaces. Error bars = repeat SD.\nGEMM includes identical modern Python wrapper around each frozen C ABI. Right panel: equal case weighting, not training-frequency weighting.')
fig,axs=plt.subplots(1,2,figsize=(14,5))
for ax,metric,title in zip(axs,['max_abs','relative_l2'],['Maximum absolute error','Maximum relative L2']):
 for j,mode in enumerate(modes):
  vs=[max(float(r[metric]) for r in acc if r['op']==op and r['mode']==mode) for op in ops];ax.bar(np.arange(7)+(j-1)*.24,np.maximum(vs,1e-16),.24,color=colors[mode],label=mode)
 ax.set_xticks(range(7),[labels[o] for o in ops],rotation=30,ha='right');ax.set_yscale('log');ax.set_ylabel(title+' (log)');ax.set_title(title+' vs independent FP64');ax.legend()
save(fig,'06_accuracy','Maximum over all cases and both interfaces; full FP64 4096^3 reference included. Exact zero is displayed at 1e-16 (plot floor only).\nDiagnostic tolerance atol=5e-5, rtol=2e-5, 0 failed elements. Existing unit-test gates unchanged. No claim of bitwise equivalence.')
fig,axs=plt.subplots(3,1,figsize=(17,11),gridspec_kw={'height_ratios':[1.7,3,1.5]});norm=TwoSlopeNorm(vmin=-2,vcenter=0,vmax=2)
# heat map shows log2 speedup with native wins blue, losses red.
gcases=[r['case'] for r in rs if r['op']=='gemm' and r['mode']=='native' and r['scope']=='array' and r['metric']=='sync_us']
mat=np.array([[float(get('gemm',c,'cupy')['mean'])/float(get('gemm',c,'native')['mean']) for c in gcases]])
axs[0].imshow(np.log2(mat),aspect='auto',cmap='RdBu',norm=norm);axs[0].set_xticks(range(len(gcases)),[c.replace('real_','').replace('ext_','').replace('_','x') for c in gcases],rotation=65,ha='right',fontsize=7);axs[0].set_yticks([0],['GEMM']);axs[0].set_title('Native advantage regions: log2(CuPy sync time / Native sync time)')
ss=specs();oo=['im2col','col2im','conv','dweight'];mat=[]
for op in oo:
 row=[]
 for s in ss:
  c=s['id']+('_training' if op=='col2im' else '_contiguous');row.append(float(get(op,c,'cupy')['mean'])/float(get(op,c,'native')['mean']))
 mat.append(row)
im=axs[1].imshow(np.log2(mat),aspect='auto',cmap='RdBu',norm=norm);axs[1].set_xticks(range(len(ss)),[s['id'] for s in ss],rotation=25,ha='right');axs[1].set_yticks(range(4),[labels[o] for o in oo])
for i,row in enumerate(mat):
 for j,v in enumerate(row):axs[1].text(j,i,f'{v:.2f}x',ha='center',va='center',fontsize=8)
pnames=['tiny_net_pool','overlap_pool','asym_pool','negative_pool'];mat=[[float(get(op,c,'cupy')['mean'])/float(get(op,c,'native')['mean']) for c in pnames] for op in ['pool_forward','pool_backward']]
axs[2].imshow(np.log2(mat),aspect='auto',cmap='RdBu',norm=norm);axs[2].set_xticks(range(4),pnames);axs[2].set_yticks([0,1],['Pool forward','Pool backward*'])
for i,row in enumerate(mat):
 for j,v in enumerate(row):axs[2].text(j,i,f'{v:.2f}x',ha='center',va='center')
save(fig,'07_advantage_regions','A100, resident arrays, sync-complete means; n=18/30. Red <1x; white=1x; blue >1x. Colors saturate at 0.25x / 4x.\nAll GEMM shapes; primary layouts for convolution ops (other layouts in CSV). No inference of significance from color alone; paired CI in SPEEDUP_SUMMARY.csv.')
print('14 figures generated')
