# G3 P3-B Joint Atom Query Train4937

> 日期：2026-08-04
>
> 状态：跨分子能力提高 / 容量身份灾难性回退 / NO-GO

## 实验

P3-B从P3-A加载完整301,206参数Atom query模块并全部解冻。train5k中4,937张候选可完整覆盖
GT的图训练5 epochs、6,175 steps、24,685 exposures；使用pointer-compatible Hungarian、唯一
candidate pointer、existence/coordinate/pointer/Count联合损失。63张不可达图不参与pointer训练，
但未从数据或后续评测删除。route100只在step0/final读取。

## Route100

| 指标 | P3-A step0 | P3-B final |
|---|---:|---:|
| Count exact | 8/100 | **27/100** |
| Count MAE | 4.59 | **1.63** |
| predicted-count matched | 2674/3080 | **2930/3080** |
| predicted whole-set exact | 0/100 | **11/100** |
| known-count matched | 2827/3080 | **2922/3080** |
| known-count whole-set exact | 4/100 | **30/100** |

扩量联合训练显著获得跨分子能力，但仍未通过预设route门。

## Capacity64 回退

| 域 | P2-X/P2-W joint | P3-B final predicted |
|---|---:|---:|
| easy32 | 967/967，32/32 | **946/967，5/32** |
| hard32 | 807/807，32/32 | **685/807，0/32** |

即使输入真实Count，P3-B也仅easy 14/32、hard 0/32。训练loss持续下降，但动态Hungarian的
`boundary_exact_images`始终为0；模型通过不断变化的等价query身份降低局部loss，与固定基数joint
MAP推理解码不一致。容量样本本身在训练集中，故这不是普通未回放遗忘，而是目标协议冲突。

## 决策

- `joint_atom_query_train4937_generalization_no_go`。
- 证据同时否定“只缺数据”和“容量路线无效”：数据确实提高route能力，joint decoder也已过容量门；
  真正断点是扩量时动态身份目标破坏稳定对象语言。
- 停止相同动态Hungarian协议、epoch或数据量扩张。
- P3-C应保留P2-R capacity64固定identity replay作为不变量，并让一般样本的训练目标直接对齐
  fixed-cardinality joint MAP，而不是再对每个query独立生成动态正负标签。

```text
model = chemvector-g3-p3b-joint-atom-query-train4937
checkpoint = outputs/checkpoints/chemvector-g3-p3b-joint-atom-query-train4937/step-006175.pt
SHA256 = 6cfe359ce9ae7904665719957c8a7bb50388d3fb74e68ef03887ec419d7eb9ad
training images = 4,937
samples seen = 24,685
trainable parameters = 301,206
decision = NO-GO
```
