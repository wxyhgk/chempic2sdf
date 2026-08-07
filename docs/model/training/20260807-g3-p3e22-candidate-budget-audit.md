# G3 P3-E22 Candidate Budget 审计

> 日期：2026-08-07
>
> 状态：预注册，待运行
>
> 输出：`outputs/audits/chemvector-g3-p3e22-candidate-budget-audit`

## 目标

P3-E21因top128在train/dev未达到99% candidate recall而停止。本轮冻结同一center field、同一
train/dev/confirm manifest和同一regional-maxima排序，一次提取top256，对前128、192、256
分别计算6px最大一一匹配可达率，选择三个split都达到99%的最小预算。

## 协议

- 零训练，不运行任何selector、SDF或下游阈值；
- batch24、BF16、确定性算法；
- 预算只允许128、192、256，不根据分桶增加临时候选；
- 三个split必须同时达到99%，否则NO-GO；
- fresh_dev、sealed_test、USPTO和MR-TADF均不读取；
- 保存top256 cache，若审计GO则供下一周期复用。

## 结果

待运行后填写。
