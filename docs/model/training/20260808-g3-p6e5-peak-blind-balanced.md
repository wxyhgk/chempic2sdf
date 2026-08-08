# G3 P6-E5 Peak-Blind Balanced

> 日期：2026-08-08
>
> 状态：已完成，NO-GO

模型名`chemvector-g3-p6e5-peak-blind-balanced-hard294-easy294`。从随机初始化训练115,581参数
结构selector，训练和推理均在模型内部永久忽略候选峰强度。训练集固定为169 generic hard、
125 real hard及各域等量easy，共294 hard+294 easy；每batch严格32+32，D4、AdamW `3e-4`，
固定5,000步。

能力门：hard fit恢复率至少90%；calibration/holdout相对第一峰都至少提高5pp，且MAE均不回退。
不读dev、`confirm_v2`、MR-TADF、USPTO或封存集。无论GO/NO-GO都保存最终checkpoint、提交结果
并创建同名annotated tag。

## 结果

- hard fit：285/294，96.94%，MAE 2.42px；
- easy fit：291/294，98.98%，MAE 1.22px；
- calibration：12/31，38.71%，MAE 8.76px；
- holdout：32/67，47.76%，MAE 7.14px。

只有hard fit容量门通过，其余四项泛化与MAE门全部失败，决策为
`peak_blind_balanced_capacity_no_go`。这证明115,581参数结构头可以拟合hard选择，但294个
hard动作不足以形成跨scaffold规则；继续增加step、参数或重复曝光没有依据。下一步数据扩充
必须增加“第一峰错误但top-8可达”的hard-choice事件，而不是增加top-1已经正确的普通easy图。

checkpoint：

```text
outputs/checkpoints/chemvector-g3-p6e5-peak-blind-balanced-hard294-easy294/step-005000.pt
SHA256 c6056b020a6eced43c2a63ecd85d4dc1515421f97c70644fda12e862016d9015
```

audit SHA256：`c235937eb2c44a4467e7e90a34281c4569f186788dc1fc09d8e82efd449f77ea`；
`confirm_v2_read=false`，`dev_and_external_read=false`。
