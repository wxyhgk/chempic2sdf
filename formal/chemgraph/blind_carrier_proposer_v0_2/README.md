# Blind Carrier Proposer v0.2 纯适配器

本目录实现一个不含模型、不读图片、不懂化学的确定性适配器。它把上游已经给出的可见直线段和
语义 Bond 分组编译为 Blind Carrier v0.2 prediction。它不访问文件系统、网络、blind inbox、sealed、
GT 或其他 Data，也不加载 checkpoint。

这一步只验证：**如果上游已经正确给出物理线段、support、Bond 分组和端点 hint，能否无歧义地产生
通过冻结 v0.2 validator 的完整中间表示。** 它不证明这些输入可以从真实 PNG 稳定预测出来。

## 输入类型

`VisibleSegment` 是一条两点直线：

```python
VisibleSegment(
    id=7,
    begin_q4=(1608, 6152),
    end_q4=(8008, 6152),
    width_q4=16,
    support_runs=((384, 100, 401),),
)
```

`TerminalHint(id, point_q4)` 的 `id` 是上游给出的不透明端点身份；同一个 hint ID 在所有 Bond 中
必须给出相同坐标，不同 hint ID 不得占用同一坐标。hint ID 只定义输入内的端点等价类，不参与输出
编号或排序。

`SemanticBondGroup` 把 1 至 3 条物理 segment 声明为一个语义 Bond：

```python
SemanticBondGroup(
    segment_ids=(7, 9),
    begin_hint=TerminalHint(10, (1608, 6152)),
    end_hint=TerminalHint(20, (8008, 6152)),
)
```

## 确定性规则

1. segment 按输入 ID 排序并重编号为连续输出 ID；每个 segment 生成且只生成一条 lane。
2. Bond 是无向对象。适配器先按 hint 坐标字典序规范两端，再比较 segment 正反向到两端 hint 的
   总平方距离；唯一较小者确定 endpoint index，平局拒绝。
3. 相同 hint ID 的全部物理 endpoint 聚为一个 terminal；terminal 输出 ID 由
   `(center_q4, endpoint refs)` 排序派生。交换 begin/end 或整体重命名 opaque hint ID 不改变输出字节。
4. terminal center 每轴固定为 `(sum + n//2) // n`，与 v0.2 half-up mean 完全一致。
5. 每个 terminal 和每个 Bond 两侧的 endpoint carrier-disk 相交图都必须连通，闭边界相切通过。
6. 同一 Bond 的 lane 必须平行同向、support 两两不交；整个 selected interpretation 的任意平行
   segment 对都执行 near-overlay 门，不能把一条线跨 Bond 拆成两个假单键。
7. 每条 segment 必须恰好属于一个 Bond group，并在完整 interpretation 中由对应 selected lane 所有。
8. 每条 segment 的 support 必须与两点几何双向覆盖，并保留相对其他 segment 的独占前景。
9. 每个 terminal 的 `endpoint_alignment` 和 support union 均硬限制为 64；超限 closed reject。
10. 不同 terminal 的 endpoint carrier regions 必须两两不相交；相交表示单解释不足，适配器 closed reject，
    不能靠微小 hint 偏移把一个可见 junction 拆成两个图节点。
11. 适配器固定 one-segment-per-lane，因此物理 segment 和输出 lane 均最多 512；冻结 schema 虽允许
    1024 个 segment，但那部分容量只保留给未来的 multi-segment lane 表示。
12. 输入合同内的预期失败输出 schema 允许的 closed reject；输出不携带错误细节、hint 或化学字段。
    非预期运行时异常不伪装成 `internal_error`，而是传播给 runner，由外层执行合同记录失败。

`propose_prediction(...)` 返回带单个结尾换行的 canonical JSON bytes。合法结果固定
`training_authorized=false`。

## 验证

测试包含 single、double、标准平行 triple、chain、fan、不同 Bond 的 X crossing，以及 bad support、
重复归属、lane support 重叠、Bond 内及跨 Bond 的 1 Q4 near-overlay、disk 不连通、漏归属、bool
伪整数、terminal 超限、不同 terminal region 相交、512/513 lane 容量边界和异常传播。测试还验证交换
无向端点、整体重命名 hint ID 时输出逐字节相同。
每个 `status=ok` 均送入冻结 Blind Carrier v0.2 validator 二次验收；closed reject 也验证 schema/规范字节。

```bash
cd /home/zhenghe/GK/Project/chempic2sdf
.venv-train/bin/python -m unittest \
  formal.chemgraph.blind_carrier_proposer_v0_2.test_adapter -v
```

当前目录不授权 blind 选择、proposer 真实执行或训练：

```text
blind_sample_selection_authorized=false
proposer_execution_authorized=false
training_authorized=false
```
