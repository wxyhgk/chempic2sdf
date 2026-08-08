# G3 P5-E6 Refine Error Decomposition

> 日期：2026-08-08
>
> 状态：已预注册，等待审计

## 目标

P5-E5把动作扩到1,378后，calibration/holdout恢复率提高到77.42%/53.73%，证明数据量有效但
仍未过门。本轮不训练，只对冻结P5-E5的98个跨scaffold Oracle动作拆解剩余错误，判断当前标签
是否混合了不同视觉机制。

## 固定分桶

- source距离：`(6,8]`、`(8,10]`、`(10,12]`px；
- 目标语义：ChemScene `IMPLICIT/TEXT`、元素、incident degree；
- 文字干扰：任一GT文字bbox是否与source中心32x32 crop相交；
- 局部拥挤：source 16px内GT数量，目标最近邻`<16`、`16--32`、`>=32`px；
- center-field证据：目标cell在4--13px环带内的rank，以及3px内是否存在局部峰；
- 预测误差：方向角`<=30`、`31--90`、`>90`度，预测半径、heatmap margin和entropy。

每个split动作数必须保持31/67；单桶至少5个动作才作解释。只读calibration和holdout，强制正确
Oracle source；不读dev、`confirm_v2`、外部或封存集，不训练、不创建tag。

下一实验只能针对两个域方向一致、样本量达门的错误机制预注册；不得从单个小桶直接提出修复。
