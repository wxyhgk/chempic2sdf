# G3 P3-E30 Bond Pair Consistency Micro160

> 日期：2026-08-07
>
> 状态：已完成，全局双端点关系 NO-GO
>
> 审计：`chemvector-g3-p3e30-bond-pair-consistency-micro160`

## 假设

P3-E29把每个candidate附近的预测键端点压成独立局部统计，总体AUC达到0.9608，但tail-required
图片仍为0/27。这可能不是BondObject没有信息，而是压缩方式丢失了一根键最重要的关系：两个端点
必须由两个不同节点共同解释。

本轮不训练模型，直接测试成对关系的零训练上界。若仍不能救回tail样本，就停止使用当前冻结
BondObject指导top192 Atom selector；若通过，才允许把该目标改造成可训练的graph-conditioned set
selector。

## 固定协议

- 只读取P3-E29冻结预测cache，SHA256为
  `003a6fc0384a6c5a5a16cf5be8387d10b5ee8a4cd6957792a00623b289265c29`；
- 不再次读取PNG，不读取GT bond、Render JSON、bond layer或semantic layer；
- 对每根预测键构造candidate-pair affinity，同时解释endpoint0和endpoint1，并对正反方向对称化；
- 距离核不使用固定像素宽度。每张图取score最高64根预测键的中位键长作为reference，测试固定
  比例0.08/0.16/0.32；
- selector使用known Count做信号上界，从raw candidate score开始，逐步加入与已选节点pair affinity
  最大的候选；alpha固定为0、0.03125、0.0625、0.125、0.25、0.5、1、2；
- 同集oracle只用于判断是否存在信号，不作为部署模型结果。

## GO/NO-GO门

- 27张tail-required图片至少5张Atom集合完整；
- tail exact严格高于alpha=0原始score基线；
- 旧top128可达图片的Atom recall回退不超过0.5个百分点；
- 160张及27张tail输入完整；
- 不读取dev、confirm、fresh、USPTO、MR-TADF或sealed_test。

任一项失败即`bond_pair_consistency_signal_no_go`，不得追加alpha/scale搜索，也不得训练同类模型。
本轮零训练，不创建tag。

## 实际结果

| 指标 | 原始score | 最佳pair consistency |
|---|---:|---:|
| 整体exact | 53/160 | 55/160 |
| 整体Atom recall | 95.92% | 96.18% |
| top128可达exact | 53/133 | 55/133 |
| top128可达recall | 96.89% | 97.14% |
| tail exact | 0/27 | 0/27 |
| tail recall | 92.27% | 92.61% |

最佳固定组合为`ratio=0.08, alpha=0.03125`。它只多匹配22个Atom、增加2张普通完整图，未救回
任何tail完整图；提高alpha后整体和tail recall均快速下降。三个自适应ratio的平均实际尺度约为
4.30、8.60、17.19像素，但没有任一尺度形成tail exact。

最终决策为`bond_pair_consistency_signal_no_go`。P3-E29局部端点统计和本轮全局双端点关系都只
强化已有容易候选，因此不再训练当前冻结BondObject驱动的selector。下一步审计33个tail-only
GT Atom的角色、degree、variant和candidate rank，判断断点属于candidate generator、文字对象还是
视觉backbone，而不是继续构造第三个selector目标。

audit SHA256：`3ed3f58866b122a923341708367080a41cd6c8203534d4bf325edec23a5ec1ce`。
本轮只读取P3-E29预测cache和训练集GT Atom坐标；未读取GT bond、PNG、dev、confirm、fresh、外部
数据或sealed_test。零训练，不创建tag。
