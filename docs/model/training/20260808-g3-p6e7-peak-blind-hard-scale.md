# G3 P6-E7 Peak-Blind Hard Scale

> 日期：2026-08-08
>
> 状态：已预注册，等待训练

使用P6-E6的1,339个新hard-choice，构建500与1,339两臂。500臂是完整臂的确定性子集并覆盖
500个scaffold；easy从既有generic top-1正确池中取等量嵌套子集。两个115,581参数peak-blind
模型使用相同随机初始化、32 hard+32 easy、D4和AdamW `3e-4`。

每条hard固定约192次曝光，因此两臂分别训练3,000与8,034步。大臂须hard fit至少90%，在
calibration/holdout均比小臂和P6-E1第一峰至少提高5pp，且两域MAE均不比小臂差，才判定
`hard_scale_go`。不读取dev、`confirm_v2`、MR-TADF、USPTO或封存集。两臂均保存checkpoint并
创建各自同名annotated tag。
