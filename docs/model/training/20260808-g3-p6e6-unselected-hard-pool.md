# G3 P6-E6 Unselected Hard Pool

> 日期：2026-08-08
>
> 状态：已完成，GO

从P5-E8的421,122个合法generic方向对中排除已用于P6的50,000对，冻结P6-E1扫描剩余方向。
只保存第一峰距离GT大于6px、但target-free top-8至少一峰在6px内的hard-choice记录。

容量门为至少1,000动作、500分子、500 scaffold；不训练、不读取dev、`confirm_v2`、外部或
封存集。通过后用严格按分子/scaffold隔离的规模曲线验证hard数据量，不通过则需要新数据生成协议。

## 结果

- 扫描未选合法方向：371,122；
- 第一峰正确：369,472；top-8仍不可达：311；
- hard-choice：1,339，覆盖833分子、813 scaffold；
- variant：stress 831、original 184、rdkit 176、coordgen 148；
- 四个容量门全部通过，决策为`unselected_hard_pool_capacity_go`。

下一周期不请求新数据。把本池与原294 hard严格去重后，按分子/scaffold固定拆出训练规模曲线，
优先比较约500与完整1,300级hard；easy只做等量稳定器，peak logits继续永久禁用。

audit SHA256：`7850931bada748050e1d14945468bf94ad16578bd3dd46f92bb3fed81f8d4ac0`；
hard actions SHA256：`1a92dd1f6ed7549d1404dd081d3d412bb65db868353da1157920776bdd5f08ea`。
