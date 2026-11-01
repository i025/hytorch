# Udacity 阶段 A：共享预处理、Center Dataset 与连续时间划分

本阶段仅实现数据链和 CPU 验证，未运行模型前向/反向或 GPU 训练，未接入模拟器闭环驾驶。

## 基线与工作区

- 开始检查时，共享仓库 `/home/xymo/projects/hytorch` 位于 `experiment/conv-backward-shape-dispatch`，HEAD 为 `e2d60c26cdb28facd4a41dd3b443c3148e397b66`，工作区干净。
- 按要求从此前已审查的 `d26667f160bf3af772ad24685f73049f6267a11a` 新建 `feat/udacity-dataset`；创建前本地及远程均无同名分支。
- 开发期间，共享检出目录被外部工作切到 `experiment/conv-dinput-kernel-optimization`。为避免混入改动，建立独立 Git worktree，将本阶段新文件逐字节核对后移入；其他工作的文件与分支保持原样。
- 最终验收工作目录为 `/srv/lab-disks/disk2/users/xymo/Documents/Codex/2026-10-30/hytorch-a100-udacity-behavioral-cloning-hytorch/work/hytorch-udacity`，HEAD 基线明确为 `d26667f`。
- 独立工作树与主仓库共用 Git 对象和分支；不需要另行复制补丁。应用内 worktree 工具未能识别服务器检出路径，因此使用 `git worktree add` 创建。

## 文件与设计

| 文件 | 内容 |
| --- | --- |
| `donkeycar_hy/preprocess.py` | `load_rgb_image()` 关闭源文件后返回 RGB 图片；`preprocess_rgb()` 接收 RGB Pillow 图像或 HWC uint8 NumPy 数组，顶部裁 40、底部不裁、Pillow 明确双线性 resize 至宽200高66、`x/127.5-1`、转 CHW，返回连续 float32 数组。文件和未来 Telemetry 解码均可调用。 |
| `donkeycar_hy/udacity_dataset.py` | 标准库 CSV 解析，仅 Center/Steering；支持 UTF-8 BOM、字段与表头两侧空格、相对/本机绝对路径、`~` 展开；相对路径相对于 CSV。初始化检查所有 Center 路径，图像按需解码。 |
| 同上：`UdacityRecord` | 保留原始零基记录索引、一基物理 CSV 起始行号、解析后的图片路径、原始 steering 和文件名时间戳。标签不裁剪、不缩放。 |
| 同上：`split_temporal()` / `TemporalSplit` | 保持 CSV 顺序，要求时间戳严格递增；末尾 `ceil(N*val_ratio)` 作为验证集；训练只保留 `timestamp < validation_start-gap_seconds`；其余边界样本单独列为 excluded。默认比例0.1、隔离窗口2秒。 |
| 同上：`to_dict()` / `save_manifest()` | 输出三组完整源索引、CSV 行号、时间和标签统计；JSON 保存拒绝覆盖已有文件。划分不使用随机数，不修改原 Dataset。 |
| `tests/test_udacity_dataset.py` | CPU 合同测试和真实官方数据集测试，包括所有8,036张Center图片解码、标签、异常输入、时间边界、可复现性、无重叠及完整覆盖、同步/异步 DataLoader 顺序与尾批。 |
| 本目录 | 环境、完整测试命令、日志、JUnit XML、样本/标签统计和完整划分索引。 |

旧 DonkeyDataset、DonkeyNet、Optimizer、DataLoader、训练入口与所有 C/CUDA 文件均未修改。新增两个数据模块本身不导入 HyTorch 或 PyTorch；检查脚本确认导入现有 CPU DataLoader 后也未导入 PyTorch。

## 完整测试命令与结果

最终验收在上述独立工作树中执行，复用原仓库现有虚拟环境，未安装依赖：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/home/xymo/projects/hytorch/.matplotlib_cache \
  HYTORCH_UDACITY_DATA=/home/xymo/datasets/udacity/data \
  /home/xymo/projects/hytorch/.venv/bin/python \
  docs/experiments/a100-stage1/run_logged.py \
  --log docs/experiments/udacity-stage-a/tests-isolated.log -- \
  /home/xymo/projects/hytorch/.venv/bin/python -m pytest -q \
  tests/test_udacity_dataset.py \
  --junitxml=docs/experiments/udacity-stage-a/tests-isolated.xml
```

结果：**44 passed，0 failures，0 errors，0 skipped；pytest 用时9.49秒**。测试未执行模型计算。日志包装器及其 JSON 元数据记录完整 argv、cwd、环境和退出码；复现时应使用新的日志文件名以避免覆盖。

`tests.log` / `tests.xml` 是共享检出目录中的首轮结果（44 passed / 11.57秒）。由于该目录随后发现外部分支切换，此结果仅保留过程记录，固定基线验收以 `tests-isolated.*` 为准。

额外统计检查使用 Python 读取真实数据并保存 `environment.json`、`inspection.json`、`split-indices.json`。完整 Python 命令保存在 `inspection.log.command.json`，输出在 `inspection.log`；检查过程没有模型计算。

环境：Python 3.8.10、NumPy 1.24.4、Pillow 10.4.0、pytest 8.3.5；测试进程 `CUDA_VISIBLE_DEVICES` 为空。

## 数据、标签和划分统计

数据集实际目录 `/srv/lab-disks/disk2/users/xymo/datasets/udacity/data`，通过 `/home/xymo/datasets/udacity/data` 访问。

- 8,036条记录、8,036个不同的有效Center图片路径；全部Center图片通过解码、shape、dtype、连续内存、有限值及范围检查。
- 单样本图片 `(3,66,200)` float32，标签 `(1,)` float32；Batch 4 得到图片 `(4,3,66,200)`、标签 `(4,1)`。
- steering 范围 `[-0.9426954,1.0]`，均值 `0.0040696440648332515`，总体标准差 `0.12883205581793244`。
- 零转向4,361条，占54.26829268292683%。
- 原始文件名时间范围为 `2016-12-01 13:30:48.287` 至 `13:46:38.947`；时区未知，不推断采集时区。
- CSV SHA256：`df567887f7dc8567f9f3580a56245e4d26a3635b3c1a5aa5744621e83d1690b5`。

默认 `val_ratio=0.1, gap_seconds=2.0`：

| 组 | 条数 | 原始零基索引（含端点） | 时间范围（2016-12-01，采集时区未知） |
| --- | ---: | --- | --- |
| train | 7,213 | 0–7212 | 13:30:48.287–13:45:18.267 |
| excluded | 19 | 7213–7231 | 13:45:18.369–13:45:20.191 |
| validation | 804 | 7232–8035 | 13:45:20.292–13:46:38.947 |

训练末帧与验证首帧实际相隔 **2.025秒**。三组无交集，合并完整覆盖全部8,036条记录，顺序保持不变。零隔离窗口时为训练7,232、验证804、排除0。完整索引和原始CSV行号见 `split-indices.json`。

训练标签均值约0.009116、验证约-0.038062；时间划分没有强制匹配标签分布，后续评估应保留这种分布差异，而不为匹配分布改用逐帧随机划分。

## 兼容性与边界

- NumPy输入仅接受HWC uint8原始RGB，拒绝float数组以避免重复归一化；Pillow输入必须已为RGB，文件解码辅助函数负责转换。
- 当前使用Pillow，未引入OpenCV或YUV；相同JPEG字节从文件和内存解码后的预处理逐元素相同；也已核对与旧Pillow预处理在真实样本上的数值兼容性。
- 文件名需包含Udacity格式时间戳；时间倒序/相同、重复图片、缺图、无效/非有限标签、无训练样本的划分均明确报错，不静默修复。
- 本机绝对路径和Linux相对路径已测试；不将其他机器的Windows绝对路径自动映射到本机，也不使用basename兜底。
- 无数据的其他环境默认跳过真实数据测试；显式设置 `HYTORCH_UDACITY_DATA` 却不存在CSV时测试失败。本次真实数据验收没有任何skip。
- 原有HyTorch顶层导入会提示缺少SciPy、无法加载SVHN；这是现有可选分类数据集提示，本阶段无需SciPy且测试通过，没有为此安装依赖。
- 2秒隔离只降低邻帧泄漏风险，不保证重复路线/圈次完全独立；CSV没有session/lap信息。本阶段不进行准确率或训练性能结论。
- 此处CPU耗时是单元测试耗时，不是训练性能基准。后续训练、Telemetry模型服务、Dropout推理模式修复均不在阶段A范围内。

## Git 检查与交付

提交前对阶段A文件执行 `git diff --cached --check`，检查仅新增预处理、Dataset、测试和本目录记录；核对既有模型、数据读取器、训练入口、Optimizer和CUDA源码均无改动。提交与远端验证的最终Hash、Push状态由本次聊天交付报告给出，避免在提交内容中记录自身Hash。
