# G3 P6-E5 Peak-Blind Balanced

> 日期：2026-08-08
>
> 状态：已预注册，等待训练

模型名`chemvector-g3-p6e5-peak-blind-balanced-hard294-easy294`。从随机初始化训练115,581参数
结构selector，训练和推理均在模型内部永久忽略候选峰强度。训练集固定为169 generic hard、
125 real hard及各域等量easy，共294 hard+294 easy；每batch严格32+32，D4、AdamW `3e-4`，
固定5,000步。

能力门：hard fit恢复率至少90%；calibration/holdout相对第一峰都至少提高5pp，且MAE均不回退。
不读dev、`confirm_v2`、MR-TADF、USPTO或封存集。无论GO/NO-GO都保存最终checkpoint、提交结果
并创建同名annotated tag。
