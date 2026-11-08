# 本阶段源码变化

## 生产：hytorch/core.py（新增11行）

- Config：增加compact_function=False。
- Function.__call__：仅非tuple单输出，减少临时tuple、outputs list、listcomp和单元素循环；已知ndarray直接传给Variable，跳过np.isscalar。
- 保留Variable验证、generation、set_creator、inputs强引用、outputs弱引用；无梯度模式依旧保存原有Function状态。
- 未改Variable.backward、计算算法、autograd排序/累加顺序、原返回类型、异常行为、dtype/设备/alias检查。

开关：

```python
from hytorch.core import using_config
with using_config('compact_function', True):
    # forward / backward / SGD，正常用法
    ...
```

默认关闭，独立于runtime.backend(metadata_cache=True)。本轮B0/O1都开启已有F1，新开关不是F1的新名字。

## 专项测试：tests/test_compact_function.py

28个参数化测试，覆盖CPU/GPU、标量/多输出、自定义Function、卷积/池化多阶导数、视图、突变、stream、生命周期及接口拒绝；精确比较数据字节。

## 诊断与交付脚本

- save_baseline.py：冻结之前的dirty工作区、源码和动态库。
- common.py / training.py / run_tasks.py：统一配置、资源检查、12轮训练、补充/内存/profiler任务。
- layers_main18.py：主窗口7卷积+4通用pool，保持该次固定seed生成顺序。
- layers.py / tiny_workload.py：真实TinyNet 4通道pool补测与运行时shape证据。前者的预验证直调只供诊断，不是生产API。
- analyze_profiles.py / aggregate.py / summarize.py：从原始计时/CUPTI计算表格及配对CI。
- report.py / figures.py：报告与PPT图；package.py：清单与打包。

完整本阶段生产/测试diff见git_diff.patch；之前轮次的全部已存在修改位于baseline/working_tree.patch，不混入本轮增量。source/提供最终必要源码，baseline/source/提供原版。SOURCE_INTEGRITY.json对83个基线文件逐个SHA比较，仅core.py改变，kernel、runtime、functions_conv、optimizers、动态库均不变。没有Commit/Push/PR。
