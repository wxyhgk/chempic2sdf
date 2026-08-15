# Fragment Support Partition v0.4 微探针

本目录在已经冻结的 geometry-only fragment root 上，为每个 base stroke 的 parent raster support 做确定性分区。
每个像素中心对 parent segment 做精确有理投影；cut 使用半开区间，精确落在 cut 上的像素归右侧 child；端帽
像素的投影钳制到 `[0,1]`。

当前只证明每个 parent fiber 内：

```text
child support 两两不交
union(child supports) = parent support
每个 parent pixel 恰好属于一个 child
```

不同 base stroke 在交叉区域可以引用同一前景像素；跨 carrier 的唯一 ownership 必须由下一层 named intersection
support 处理。当前输出逐像素列出所有 cross-carrier overlap 及 carrier fiber，不能用 count 隐藏位置或成员。
输入 parent support 仍是研究 fixture，不证明 PNG、Boundary 或 matching extractor 完备。
本探针也不验证 parent pixel 是否落在 stroke 的冻结 tube 内；它只验证给定 parent set 的确定性分区。

任一 child support 为空、RLE 非 canonical、类型不严格或容量超限时 closed failure，不输出部分 partition。
RLE hash 复用 v0.3 冻结的 `u32be(run_count)||u32be(row,start,length)...` framing；parent 与 child 的总 run
数量分别受独立容量门约束。

```text
support_partition_research_only=true
cross_carrier_ownership_complete=false
training_authorized=false
```
