# Observation Closure v0.5 P1 正式结果

本目录绑定源码提交 `fa796de982e9de63d4cff3015fbfa45317365e38` 和 source freeze
`148dc8b7a3bcb6c846b102dd1b0c3826fc60d3a562fcc4a898f182767ac8cba5`。

三路完整 census 均遍历 36,088,416 个 tuple，A-forward、A-reverse 与独立 B-shuffled 的
`records_root` 和 `coverage_root` 完全一致。Raster pilot 对 `tuple_id mod 100 == 0` 的
360,885 个 tuple 完成 padded clipping certificate、完整 canonical frame 保存和独立重渲染
逐字节 replay。

最终状态为 `NO-GO`。正确性、内存、OOM 和临时空间门均通过；失败门只有：

- pilot wall 上界 16,876 秒，大于 14,400 秒；
- 全量 wall 双倍投影 3,375,193 秒，大于 345,600 秒。

因此本结果不授权训练，也不能事后缩小冻结域来改写 P1 结论。1.4 GB 权威 raster artifact
保存在 root-owned 只读目录 `/opt/chemgraph/results/p1-raster-pilot-fa796de`，其摘要见
`pilot_receipt.json`。

Raster pilot 观察到 351,759 个 distinct raster class；360,885 个样本中有 9,126 个超出
每类首个成员的重复成员。重复类涉及 16,699 个样本，最大类大小为 7。这证明冻结渲染域内
存在大量 exact-pixel collision，但本 P1 只签发 sampled raster 统计，不把 sampled Graph
结论越权推广为 full Graph 结论。
