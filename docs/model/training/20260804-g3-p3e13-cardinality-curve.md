# G3 P3-E13 Cardinality Curve Refiner

> 日期：2026-08-04
>
> 状态：完成，NO-GO
>
> 基础 Count checkpoint：P3-E10 expanded3316 h64
>
> 输出目录：`outputs/smoke/chemvector-g3-p3e13-cardinality-curve-expanded3316-h64`

## 目标

P3-E10 的 coarse Count 在 confirm256 有 108/256 exact，但误差大多靠近真实值。本轮不再
调整 DeepSets 的普通分类 loss，而是验证一个结构化假设：真实 Count 应当能从不同固定基数
`K` 下的最大一一匹配分数曲线中识别出来。

预先固定每张图只计算 `coarse Count-4 ... coarse Count+4` 九个 K。训练一个小 refiner 预测
`-4...+4` 偏移，不重新训练 base、expert 或 P3-E10。

## 数据与隔离

- train：829 个 molecule、3,316 张四风格图片；
- dev：64 个 molecule、256 张图；
- confirm：64 个 molecule、256 张图；
- historical route100：100 张，只做 fixed-final 历史诊断；
- train/dev/confirm 按 molecule 隔离，沿用 P3-E9/P3-E10 的固定划分；
- MR-TADF 未参与训练、阈值、epoch、checkpoint 或决策选择。

训练前的 coarse Count 覆盖审计显示，真实 Count 落在 `±4` 窗口内的比例为：train
3270/3316、dev 241/256、confirm 243/256、route100 92/100。超窗训练标签固定截到边界类，
但指标中单列 window oracle，不能把超窗错误算成 refiner 失败。

## 模型与冻结范围

冻结组件：

- P2-W base：`step-004800.pt`；
- P3-E1 expert：`step-001920.pt`；
- P3-E10 h64 coarse Count：`step-001560.pt`。

对每个 K 分别在 base/expert 的 128×128 query-candidate 分数矩阵上执行精确 fixed-cardinality
Hungarian assignment，分数固定为：

```text
logsigmoid(existence) + logsoftmax(pointer)
```

每个 K 的 13 维输入包含绝对 K、相对 coarse 偏移、有效位、base/expert 总分与均分、两模型
差值、相邻 K 增量，以及 P3-E10 在该 K 的局部 probability/log-probability。九点分别经过
共享 64 维编码器，再由 77,091 参数 MLP 输出九个偏移类别。0/128 边界外类别被显式屏蔽。

## 训练配置

```bash
.venv-train/bin/python -m \
  chemvector.training.atoms.train_g3_p3e13_cardinality_curve
```

- batch size：64；
- epoch：30；
- fixed-final steps：1,560；
- exposure：99,480 张；
- AdamW：`lr=1e-3`、`weight_decay=0.01`；
- gradient clip：1.0；
- BF16；随机种子 `20260713`；
- 不选择中间 checkpoint。

## 预注册门

- dev Count exact ≥ 50%；
- confirm Count exact ≥ 50%；
- confirm MAE ≤ 1.0；
- historical route100 Count exact ≥ 40%；
- route100 不参与训练或 checkpoint 选择。

## 结果

| 集合 | P3-E10 coarse exact / MAE | P3-E13 exact / MAE | `±4` window oracle |
|---|---:|---:|---:|
| train3316 | 2797/3316 / 0.322 | **2939/3316 / 0.223** | 3270/3316 |
| dev256 | 103/256 / 1.406 | **110/256 / 1.352** | 241/256 |
| confirm256 | **108/256 / 1.270** | 101/256 / 1.301 | 243/256 |
| historical route100 | 37/100 / 1.570 | 38/100 / 1.470 | 92/100 |

四项正式门全部未通过。确定性重复评估逐项一致，fixed steps 与 exposure 均通过协议检查。

## 观察

搜索窗口不是主要限制：confirm 有 94.92% 的真实 Count 位于窗口内，但 refiner 在这些样本中
仅选对 101/243。相反，train 窗口内达到 2939/3270，形成明显泛化差距。说明当前 base/expert
匹配分数曲线没有提供跨 molecule 稳定的 Count 拐点，小 MLP 主要拟合了训练分布；不能因为
window oracle 很高就把 94.92% 当成可达到的部署准确率。

P3-E13 也没有证明固定基数匹配本身无效。它只否定了“冻结现有 selector 的 scalar matching
curve，再用 3,316 张图训练局部 offset 头”这一具体方案。P3-E7 已证明真实 Count 条件下
base/expert selector 可用，因此 Count 仍是独立首要瓶颈。

## 决策

`NO-GO`。不接 router、不接前端，当前 Count winner 仍为 P3-E10 expanded3316 h64。

下一轮不继续调九点曲线的 MLP、窗口或 loss。优先建立冻结视觉主干在
`style_train_100k` 上的大规模 Count evidence cache，再用相同 P3-E10 与 curve 协议做严格
数据量 A/B；这能直接区分当前泛化差距究竟来自 829 个 molecule 太少，还是冻结分数根本
缺少 Count 信息。

## 产物校验

- checkpoint：`step-001560.pt`；
- SHA256：`b94b1421ed4d95d5291dc096bf3ed7b65ab28c5830ce1362e7344ea78fc9445f`；
- evidence SHA256：`d86d53dc84a92c39d8206008ce92055b1c1defae481f88e337ad8b77a431cfd3`；
- 单元测试：5 项通过；
- fixed-final protocol：全部通过；
- 文档复核日期：2026-08-04。
