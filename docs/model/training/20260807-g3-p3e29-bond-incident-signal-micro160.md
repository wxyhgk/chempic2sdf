# G3 P3-E29 Bond Incident Signal Micro160

> 日期：2026-08-07
>
> 状态：已完成，局部键端点证据 NO-GO
>
> 审计：`chemvector-g3-p3e29-bond-incident-signal-micro160`

## 目标

P3-E28证明旧五路线周期已经无法达到至少两路通过的Micro门，但冻结BondObject仍可能为下一代
selector提供新图像证据。本轮先做零训练审计，回答一个更基础的问题：冻结预测的键端点，能否把
top192中真正的原子候选与假候选分开，尤其能否救回top128遗漏、top192才可达的27张图片。

若零训练上界也不能救回tail样本，就不训练局部`bond-conditioned occupancy`，避免再次只在已有
容易候选上提高平均指标。

## 冻结输入协议

- Micro160、五个Count桶和四variant完全复用P3-E25；sample SHA256为
  `06f44d30a88013643be6041fe55741fd75f6d58bcf1d875a22224972a7944501`；
- candidate输入为true-global192，tensor SHA256为
  `274222ec409a6f90999976cbb7898af6ca4d8dca9c5597fcea28c515353d8a9a`；
- Bond source为`chemvector-v0101-set-first-style10k/selected.pt`，SHA256为
  `e982057d57dd4c03864eae890df12e1d8710fce60dfc452b94511d86a5f967ac`；
- vision SHA256为
  `1eede2e814efc0c440c26eba88bc8e9ed86e0911d7727b770a992c1ea2671c6a`；
- 实际前向只读取原PNG，输出最多192个预测bond score、两个端点和键型分布。

每个candidate对所有预测端点计算连续高斯支持，尺度为4/8/12/20像素，同时记录max、sum、最近
端点距离、对应score和键型加权支持，共18维。这里没有硬最近邻分配、候选删除或GT bond输入。
GT Atom只在前向完成后用于审计标签。

## 预注册门

- candidate正负分离AUC不低于0.65；
- 对原始candidate score与endpoint support做固定alpha网格的同集oracle融合上界；
- 27张tail-required图片至少救回5张完整Atom集合；
- 必须优于原始candidate score的tail exact；
- 旧top128可达图片的Atom recall回退不超过0.5个百分点；
- 160张证据完整，且不得读取dev、confirm、fresh、外部集或sealed_test。

oracle融合仅用于判断证据中是否存在可学习信号，不是部署selector，也不能作为模型结果。

## 实际结果

| 指标 | 结果 | 判断 |
|---|---:|---|
| candidate正负AUC | 0.9608 | 通过 |
| 原始score exact | 53/160 | 基线 |
| 最佳融合exact | 55/160 | 仅增加2张 |
| 原始tail exact | 0/27 | 基线 |
| 最佳融合tail exact | 0/27 | 未救回 |
| tail-only Atom数量 | 33 | 诊断 |
| tail-only支持分位均值/最小值 | 61.62% / 8.49% | 不稳定 |
| 可达图片recall | 96.89% -> 97.03% | 无回退 |

总体AUC高，说明预测键端点会强化普通原子候选；但它对真正决定top192价值的tail-only候选没有
形成足够强的排序信号。固定alpha从0.125增加到8时，tail exact始终为0，继续增强局部支持反而
持续降低整体recall。

最终决策为`bond_incident_signal_no_go`。这否定的是“把每个候选独立压成局部endpoint统计后直接
融合”的路线，不是否定BondObject本身。下一步只审计成对键端点与候选集合的全局一致性：一根键
的两个端点必须由两个节点共同解释，不能再把两个端点拆成互相独立的局部presence特征。

## 产物

- audit SHA256：`dff767d8a61308e566c01c67a7ae96906b3bc3e2bb854337fda88be7b172fc23`；
- evidence cache SHA256：`003a6fc0384a6c5a5a16cf5be8387d10b5ee8a4cd6957792a00623b289265c29`；
- evidence tensor SHA256：`29ccbda38d82aed7c574ad6a2564ce99c4f199aa1d2aab210a9a20fed3e2e63f`；
- 实际耗时：23.76秒；
- dev、旧confirm、fresh、USPTO、MR-TADF和sealed_test均未读取。

本轮没有训练模型，不创建tag。
