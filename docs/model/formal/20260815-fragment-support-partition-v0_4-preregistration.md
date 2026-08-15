# Fragment Support Partition v0.4 预注册

日期：2026-08-15

## 合同

给定 geometry root 与每个 base stroke 的 canonical parent RLE，把 parent pixel 按精确投影参数分配给 atomic
fragment。区间规则冻结为：

```text
[t_i, t_{i+1})
最后一个区间包含 t=1
projection < 0 -> first child
projection > 1 -> last child
projection = cut -> right child
```

必须验证每个 parent fiber 内 child support 两两不交、并集精确等于 parent support，且任一 child 非空。

## 不授权范围

- parent support 是否由 PNG 完整提取；
- 不同 carrier 在 intersection 区域的像素 ownership；
- intersection support 的半径和形状；
- primitive、文字/键、port、lane、Bond；
- public proposer、SDF 和训练。

跨 carrier 重叠不是本周期的错误掩盖项，而是必须显式报告的下一层输入。
