# G3 P6-E6 Unselected Hard Pool

> 日期：2026-08-08
>
> 状态：已预注册，等待审计

从P5-E8的421,122个合法generic方向对中排除已用于P6的50,000对，冻结P6-E1扫描剩余方向。
只保存第一峰距离GT大于6px、但target-free top-8至少一峰在6px内的hard-choice记录。

容量门为至少1,000动作、500分子、500 scaffold；不训练、不读取dev、`confirm_v2`、外部或
封存集。通过后用严格按分子/scaffold隔离的规模曲线验证hard数据量，不通过则需要新数据生成协议。
