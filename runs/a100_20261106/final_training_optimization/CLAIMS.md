# 可用于汇报的结论与证据

| ID | 结论 | 性质 | 证据 | 边界 |
|---|---|---|---|---|
| C1 | D使本固定训练wall减少3.552%，12/12胜出 | 实测 | TRAINING.csv; PAIRED.csv; raw/training_summary.json | 限A100/该合成配置，非全workload |
| C2 | D每步少5个列布局copy，147→142kernel；im2col仍10 | 实测 | PROFILE.csv; PROFILE_SUMMARY.csv; traces/train_*.sqlite | 不称前向im2col复用，不称消除backward展开 |
| C3 | D通过168固定/留出+15真实梯度FP64准入及完整训练对齐 | 实测 | NUMERICS.csv; TRAINING_DWEIGHT_NUMERICS.csv; TRAINING_ERRORS.csv | 不保证每个元素更准确或全数值范围 |
| C4 | batch4池活跃峰值下降，连续3轮未持续增长 | 实测 | MEMORY.csv | 仅CuPy allocator，不是整设备所有库精确峰值 |
| C5 | SGD原型16tensor从32kernel降至1，但因FTZ/subnormal不一致拒绝 | 实测且未通过验收 | OPERATOR_PROFILE.csv; raw/special_initial.log; raw/cupy_ftz_source.txt | 不能用于生产训练加速声明 |
| C6 | D的wall收益与减少packing/提交工作一致 | 推断，机制由计数支持 | FULL_WRAPPERS.csv; PROFILE.csv; TRAINING.csv | 不能把CPU/GPU重叠时间相加定量分解 |
| C7 | 真实驾驶任务精度、全形状泛化、早期累计收益 | 未验证/NOT_RUN | REPORT.md | 不得用历史报告补成新实测 |
