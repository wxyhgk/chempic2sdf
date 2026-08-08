# G3 P6-E4 Selector Shortcut

> 日期：2026-08-08
>
> 状态：已完成，shortcut确认

## 目的与协议

冻结P6-E3，在calibration 31与holdout 67动作比较四臂：P6-E1第一峰、完整selector、清零
原图与center field的selector、抹平8个峰强度的selector。记录选择变化、rescue和harmful。

若完整selector在两域都至少90%复制第一峰，且清零视觉后与完整selector仍至少90%选择一致，
则确认`peak_rank_shortcut`。审计不训练，不读取dev、`confirm_v2`、外部或封存集。若确认，
下一模型必须删除候选峰强度输入，并以294个已知hard-reachable动作和等量easy动作做平衡容量实验，
而不是继续让50k简单样本淹没结构监督。

## 结果

- 完整selector复制第一峰：calibration 96.77%，holdout 98.51%；
- 完整与清零视觉的选择一致率：93.55%与98.51%；
- 完整selector相对第一峰只各改选1次，合计0 rescue、1 harmful；
- 抹平峰强度后改选12/22次，但只有2/2 rescue，同时有7/9 harmful；
- 决策：`peak_rank_shortcut_confirmed`，四个预注册检查全部通过。

这证明P6-E3没有学成可用的结构视觉选择器。下一容量实验必须从随机初始化开始，训练和推理均
永久把峰强度置零，并平衡294个hard-reachable与294个easy动作；不能从P6-E3微调。

audit SHA256：`dd13f7661d87fe0005e3a2edc0a32ecec03e880d6847611c516a7a88555327df`；
records SHA256：`20dc7ab9f808519371a52779b159d53bdb48cc3549d8cf376268d215b3a0312c`。
