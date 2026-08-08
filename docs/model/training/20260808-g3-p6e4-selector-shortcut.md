# G3 P6-E4 Selector Shortcut

> 日期：2026-08-08
>
> 状态：已预注册，等待审计

## 目的与协议

冻结P6-E3，在calibration 31与holdout 67动作比较四臂：P6-E1第一峰、完整selector、清零
原图与center field的selector、抹平8个峰强度的selector。记录选择变化、rescue和harmful。

若完整selector在两域都至少90%复制第一峰，且清零视觉后与完整selector仍至少90%选择一致，
则确认`peak_rank_shortcut`。审计不训练，不读取dev、`confirm_v2`、外部或封存集。若确认，
下一模型必须删除候选峰强度输入，并以294个已知hard-reachable动作和等量easy动作做平衡容量实验，
而不是继续让50k简单样本淹没结构监督。
