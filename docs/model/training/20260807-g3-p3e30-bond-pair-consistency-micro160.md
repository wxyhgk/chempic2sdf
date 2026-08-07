# G3 P3-E30 Bond Pair Consistency Micro160

> 日期：2026-08-07
>
> 状态：预注册，待执行零训练审计
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
