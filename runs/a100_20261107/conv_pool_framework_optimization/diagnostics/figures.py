from pathlib import Path
import csv,json,statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
O=Path(__file__).resolve().parents[1];F=O/'figures';F.mkdir(exist_ok=True)
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font);family=font_manager.FontProperties(fname=font).get_name()
plt.rcParams.update({'font.family':family,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'path','axes.unicode_minus':False})
def read(n):return list(csv.DictReader((O/n).open()))
rows=read('LAYER_COMPARISON.csv');idx={(r['case'],r['implementation']):r for r in rows};S=json.loads((O/'SUMMARY.json').read_text());modes=['B0','O1','torch'];colors=['#7b8794','#14866d','#d66b30'];labels=['B0：F1基线','O1：单输出包装优化','PyTorch FP32']
def save(fig,name,note):
 fig.text(.5,.025,note,ha='center',va='bottom',fontsize=9,color='#455160');fig.tight_layout(rect=(0,.10,1,.90));fig.savefig(F/(name+'.png'),dpi=180,facecolor='white');fig.savefig(F/(name+'.svg'),facecolor='white');plt.close(fig)
def bars(ax,cases,xlabels,metric,title,chosen=modes):
 x=np.arange(len(cases));width=.24 if len(chosen)==3 else .32
 for j,m in enumerate(chosen):
  ax.bar(x+(j-(len(chosen)-1)/2)*width,[float(idx[c,m][metric]) for c in cases],width,color=colors[modes.index(m)],label=labels[modes.index(m)])
 ax.set_xticks(x,xlabels);ax.set_ylabel('μs / call');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
conv=[f'conv{i}/backward' for i in range(1,6)]
fig,axes=plt.subplots(1,2,figsize=(14,5.8));fig.suptitle('卷积完整反向：kernel 接近，完整调用仍有差距',fontsize=17,y=.98)
bars(axes[0],conv,[f'Conv{i}' for i in range(1,6)],'gpu_kernel_us_median','GPU kernel 累计执行（CUPTI，中位数）')
bars(axes[1],conv,[f'Conv{i}' for i in range(1,6)],'sync_us_median','完整 autograd + 最终同步（中位数）')
axes[0].legend(fontsize=9);axes[1].text(.98,.95,'五层完整调用降幅中位数：3.16%\n部分单层配对 CI 包含零',transform=axes[1].transAxes,ha='right',va='top',fontsize=10)
save(fig,'01_conv_backward_gpu_full','A100 GPU1 | DonkeyNet 五层真实形状，batch=4，同求 x/W/b 梯度 | 无 profiler n=18；CUPTI n=15\nConv1 的 Torch 输入梯度局部耗时较高；真实训练首层输入不 requires_grad，不能累加本图预测训练。')
pools=['tiny_actual_pool','tiny_pool','irregular_pool','overlap_pool','wide_pool'];pc=[p+'/forward' for p in pools];pl=['真实 Tiny\nN4 C4','通用\nN4 C3','不规则\nN2 C7','重叠 pad\nN2 C5','宽矩形\nN1 C3']
fig,axes=plt.subplots(1,2,figsize=(14,5.8));fig.suptitle('Pooling 前向：GPU 时间不变，Python 包装略有改善',fontsize=17,y=.98)
bars(axes[0],pc,pl,'gpu_kernel_us_median','GPU kernel（CUPTI，中位数）');bars(axes[1],pc,pl,'sync_us_median','完整 Function + 最终同步（中位数）');axes[0].legend(fontsize=9)
save(fig,'02_pooling_gpu_full','A100 GPU1 | batch / channel 如横轴 | 无 profiler n=18；CUPTI n=15 | 所有输入为正，padding winner 与 Torch 一致\n真实 Tiny 为 11-08 独立补测：同步延迟 −1.88%，配对 CI 含零；其余为 11-07 主窗口。DonkeyNet 无 Pooling。')
fig,axes=plt.subplots(1,2,figsize=(14,5.8));fig.suptitle('CPU 提交时间：保留检查与图语义，减少临时包装',fontsize=17,y=.98)
bars(axes[0],conv,[f'Conv{i}' for i in range(1,6)],'submit_us_median','完整 Conv Backward 提交',modes[:2]);bars(axes[1],pc,pl,'submit_us_median','完整 Pooling Function 提交',modes[:2]);axes[0].legend(fontsize=9);axes[0].set_ylim(0,360)
save(fig,'03_cpu_submission','A100 GPU1 | Conv batch=4，Pool batch 如横轴 | 同进程无 profiler，18轮平衡配对，中位数\nCPU 提交包含 Python / C / CUDA API 与潜在背压，不代表 GPU 已完成；不能与 GPU 时间相加。真实 Tiny 为独立补测。')
fig,axes=plt.subplots(1,2,figsize=(13,5.8));fig.suptitle('正式训练：wall time 降低 4.20%，12 / 12 轮胜出',fontsize=18,y=.98)
x=np.arange(3);means=[S[m]['mean_s'] for m in modes];sd=[S[m]['sd_s'] for m in modes]
axes[0].bar(x,means,color=colors,yerr=sd,capsize=5);axes[0].set_xticks(x,['B0（F1）','O1','PyTorch']);axes[0].set_ylabel('wall time (s / 192 steps)');axes[0].set_ylim(0,.88);axes[0].set_title('均值 ± 样本标准差')
for i,v in enumerate(means):axes[0].text(i,v+sd[i]+.013,f'{v:.6f}s',ha='center')
pr=read('PAIRED.csv');ds=[1000*float(r['saved_s']) for r in pr]
axes[1].bar(np.arange(1,13),ds,color=colors[1]);axes[1].axhline(0,color='#555',lw=1);axes[1].axhline(S['paired']['delta_mean_s']*1000,color='#124b40',linestyle='--');axes[1].set_xticks(np.arange(1,13));axes[1].set_xlabel('配对轮次');axes[1].set_ylabel('B0 − O1 (ms)');axes[1].set_title('逐对节省；均值 32.10ms\n95% 配对 CI [25.62, 38.57]ms')
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
save(fig,'04_training_paired','A100 GPU1 | DonkeyNet，256样本，batch=4，192steps，seed42，SGD 0.001，FP32，TF32 off\n同进程12轮（6种顺序各2次），无 profiler；含 DataLoader / H2D / forward / backward / SGD / 最终同步。')
fig,axes=plt.subplots(1,2,figsize=(13,5.8));fig.suptitle('与 PyTorch 的差距缩小，但尚未消除',fontsize=18,y=.98)
cats=['Conv Backward\n五层中位数','Pool Forward\n4种通用形状','完整训练\n192 steps']
base=[S['local']['conv_real']['sync_us_median']['B0_over_torch_median'],S['local']['pool_main_window']['sync_us_median']['B0_over_torch_median'],S['paired']['before_vs_torch']]
opt=[S['local']['conv_real']['sync_us_median']['O1_over_torch_median'],S['local']['pool_main_window']['sync_us_median']['O1_over_torch_median'],S['paired']['after_vs_torch']]
x=np.arange(3)
for j,(vals,lab,col) in enumerate([(base,labels[0],colors[0]),(opt,labels[1],colors[1])]):
 pos=x+(j-.5)*.32;axes[0].bar(pos,vals,.32,color=col,label=lab)
 for xx,v in zip(pos,vals):axes[0].text(xx,v+.035,f'{v:.3f}×',ha='center',fontsize=10)
axes[0].axhline(1,color='#d66b30',linestyle='--');axes[0].set_xticks(x,cats);axes[0].set_ylabel('HyTorch / PyTorch 耗时比（越低越好）');axes[0].set_ylim(0,3.2);axes[0].legend(fontsize=9)
gaps=[1000*(S[m]['mean_s']-S['torch']['mean_s']) for m in ['B0','O1']];axes[1].bar([0,1],gaps,color=colors[:2]);axes[1].set_xticks([0,1],['B0','O1']);axes[1].set_ylabel('与 PyTorch 的训练 wall 差距 (ms)');axes[1].set_ylim(0,530)
for i,v in enumerate(gaps):axes[1].text(i,v+9,f'{v:.2f}ms',ha='center')
axes[1].set_title('绝对训练差距减少 7.13%\nO1 仍需 PyTorch 的 2.332× 时间')
save(fig,'05_gap_to_pytorch','A100 GPU1 | 主窗口独立统计：Conv batch4；通用Pool batch4/2/2/1；训练 batch4\n局部完整同步调用 n=18；训练无 profiler 配对 n=12。比值不是各阶段可相加的贡献，不使用历史窗口累计收益。')
print('Created 5 PNG + 5 SVG')
