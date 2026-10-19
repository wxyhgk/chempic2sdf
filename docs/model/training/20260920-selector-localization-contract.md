# 上游定位最小契约：目标格监督不等于错误候选格监督

## 结论

现有 `spatial_atom_object_loss` 的 offset 项只在 **GT 所在网格** 取值。
如果某目标的最近候选来自另一格，这个目标不会直接监督那一格的 offset。
因此上一轮“10 个缺口中，9 个存在几何可达的已有格”不能改写成“训练 offset 就能救回 9 个”。

这不是宣布其他格完全没有学习信号：共享卷积参数、其他 GT 或下游坐标相关损失可能影响它们。
本轮只确认既有直接监督的采样支持域，没有真实模型参数梯度或准确率结论。

依据的旧报告 SHA 已核对为
`834bcf6eeb2e119bc8e0bfd64db1fbe20259bcd6275c384dd72ff0c5bbc77e62`。
10 个缺口、9/1 几何可行性及目标格均不在 raw 的结论沿用
[坐标契约报告](20260920-selector-coordinate-contract-audit.md)，没有重跑图片或改写旧报告。

## 实际监督与解码

- [`spatial_atom_object.py:114`](../../../chemvector/components/atoms/objects/spatial_atom_object.py#L114)：
  由 `q * width/height` 得到 GT 格；offset 的 SmoothL1 只索引这些格。不是在当前候选格上做目标吸附。
- [`vertex.py:57`](../../../chemvector/components/vision/vertex.py#L57)：
  heatmap 的中心格为 1，默认半径 1 的相邻格是小于 1 的 Gaussian 值。
- [`vertex.py:133`](../../../chemvector/components/vision/vertex.py#L133)：
  只有 target==1 是 focal 正例；Gaussian 邻格是减小权重的负例，并非额外正中心或额外 offset 监督点。
- [`predicted_incidence.py:37`](../../../chemvector/components/graph/joint/predicted_incidence.py#L37)：
  sigmoid → 3×3 局部最大值 → 全图 top-k → threshold mask，顺序不变。
  并列平台都可成为局部峰；top-k 的并列选择必须看实际返回索引，不能自行规定低 grid ID 优先。
- [`predicted_incidence.py:79`](../../../chemvector/components/graph/joint/predicted_incidence.py#L79)：
  decoder 从入选网格直接取 anchor，仍为 `(cell + sigmoid(offset))/192`，没有吸附到 GT。

## 合成验证，不运行模型

新增测试直接构造 CPU 输出 tensor，调用现有 loss 和 decoder；不构造任何 `nn.Module`。

1. GT 格的 x/y offset 误差获得相应梯度；相邻或更远的错误候选格，从这项直接监督得到的梯度为零。
2. 半径 1 的 Gaussian 邻格有负例 heatmap 方向，但不会因此获得该 GT 的 offset 梯度。
3. 人工构造一个距离 GT 7 px 的候选，其格内另一个有限 offset 可到 5.4 px：
   数学上可进入原 6 px 半径，当前 GT 格 offset loss 对错误候选格仍无直接梯度。
4. decoder 精确保留选中格坐标；对解码坐标另构造损失则可以回传到该候选格，
   防止把“直接监督为零”误称为“任何路径的梯度必为零”。
5. 分别复现局部峰抑制、峰低于阈值、超过 top-k 容量、并列平台、阈值包含等号和 padding mask。
   合成数据里的小尺寸/容量只用于测试算法契约，没有修改真实模型配置。

新数值帮助函数 [`inspect_decoder_cell`](../../../tools/selector_localization_contract.py)
只接受已提供的单张 CPU dense heatmap，返回局部峰、top-k、threshold 的实际判定及并列计数。
它没有文件读取、模型加载或生产入口，不替换原 decoder，也没有接入训练。

验证命令：

```text
.venv-train/bin/python -m pytest -q \
  tests/chemvector/test_selector_localization_contract.py \
  tests/chemvector/test_selector_coordinate_contract.py
```

结果 **34 passed / 1.03 秒**，其中新增 20 项；不是全仓测试。
真实图片读取、模型前向、checkpoint 加载、GPU、参数更新、身份查询均为零。
未读取开发64或封存数据，未修改历史 loss、阈值、匹配、坐标或原始产物。

## 缺什么证据

仅凭保存的 raw 坐标和 grid ID，无法区分一个目标格为什么没进入 raw。
需要同一次 image-only 前向的以下记录：

| 问题 | 必需 dense 证据 |
| --- | --- |
| GT 格是不是局部峰 | GT 格及其 3×3 邻域概率、局部最大值与相等关系 |
| 是不是阈值拒绝 | 原始概率、原 0.01 threshold 判定；不能只保存 top-k 后分数 |
| 是不是 top-k 排除 | 全图局部峰分数、原 128 容量、实际 top-k 索引、cutoff 及并列情况 |
| 正确格坐标若保留是否准确 | GT 格 offset logits/sigmoid 及解码坐标 |
| 邻近候选是否仅需格内偏移 | 其实际 grid ID/offset/坐标、格内距离下界；保持 q1023 与原 6 px 契约 |

首个排除环节只是解码机制的观察，不自动证明该环节对应的训练原因。
特别是低分格可能同时不在 top-k 内，应保留所有判定，不能只给互斥标签隐去其他条件。

## 唯一优先的未来观测：固定 7 图 dense 快照

若主任务另行确认执行范围，建议只对上述 **10 个目标所属的 7 张旧训练/回放图**，
从原 initial 做一次固定上游 dense 输出观察：仅视觉与 atom 输出，不跑 selector、pair、OCR 或训练。
先冻结图片键、输入 SHA、模式与数值设置；GT 只在 image-only 输出保存后用于解释。
保存完整 heatmap 与 offset，以及实际 decoder 索引/mask，使用原配置，不扫描阈值、容量或匹配半径。
建议 CPU 硬上限 120 秒、GPU=0、每图至多一次前向、无自动重试。

先核对是否复现旧候选格和缺口。CPU/GPU 数值差异或候选格不一致必须单列为未复现，
不能把新输出强行解释成历史输出的原因。完成或超时即停止，完整与未完成分母分开记录。

这次观察用于决定究竟缺的是正确峰、排名/阈值保留还是格内定位证据，
不是通过条件就自动训练。当前没有批准或执行这次真实推理，更没有批准上游解冻、
增大 top-k、改半径或新一轮试训；也不再重提已经完成且 NO-GO 的 full-loss 对照。
