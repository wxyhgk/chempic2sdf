# Blind Carrier v0.2 数据无关验证器

本目录只验证匿名 Blind Carrier v0.2 inbox/prediction 的 closed schema、规范字节和结构合同。不选择
数据、不读取 Blind inbox 目录、sealed 或 GT，不运行 proposer/模型，不评分，也不授权训练。

固定解释器：

```bash
/home/zhenghe/GK/Project/chempic2sdf/.venv-train/bin/python
```

## 冻结 schema

```text
Data/benchmarks/blind_carrier_v0_2/schemas/input.schema.json
0e1478985f29c1c78f0d88c1c4149972a8a01c60986569dfdd27f286951a6a96

Data/benchmarks/blind_carrier_v0_2/schemas/prediction.schema.json
8970b3317b0b6c0c3d9479bc260ee36858e22b446a990e00d3e057d1faaf7d40
```

JSON 继续拒绝 duplicate key、非有限数、非 UTF-8、非规范序列化及 schema 外字段。PNG 门完整继承
v0.1：仅 768×768 8-bit grayscale，chunk 只能为 `IHDR, IDAT, IEND`，固定 filter 0、zlib 9、
单 IDAT，并要求完整解码后规范重编码逐字节相等。

## Terminal region v0.2

每个 terminal 新增：

```json
"endpoint_alignment": [
  {"segment_id": 0, "endpoint_index": 0}
]
```

固定规则：

- alignment 按 `(segment_id, endpoint_index)` 排序且唯一；
- `support_segment_ids` 必须严格等于 alignment 的 segment ID union；
- `endpoint_index=0/1` 分别绑定 polyline 首/末端点；
- center 每个坐标严格为 `(sum + n//2) // n`，即非负整数 half-up rounded mean；
- 每个 aligned endpoint 的 disk radius 仍为 `ceil(width_q4/2)+16 Q4`；
- disk 相交图使用闭边界 `distance <= left_radius + right_radius`，必须连通；不再要求全部 disk 有
  单一公共交点，也不要求 center 落在每个 disk 内；
- 当前 v0.2 **只支持 one-segment lane**；其 begin/end terminal 必须分别且唯一对齐该 segment 的
  两个不同 endpoint；begin/end center 必须不同；
- 对每个 BondBundle 的 begin/end 两侧，按当前 Bond 的 lane segment 过滤 terminal alignment 后，
  必须精确得到全部 lane 的对应 endpoint refs，且这些 refs 自身的 disk 图连通；
- 对每个 interpretation，连通检查只使用 selected Bond lane segments，未选中 lane 不能充当 bridge；
- 同一 selected endpoint ref 在一个 interpretation 中只能属于一个 terminal；
- fan/chain 的共享 terminal 可以包含来自多个 selected Bond 的 refs；核对某个 Bond 时只过滤当前
  Bond 的 segment IDs，不把其他 Bond refs 当作多余项。

## 保留的 v0.1 门

RLE 排序/计数、真实前景引用、support↔geometry 双向覆盖、每 segment 独占像素、同 BondBundle lane
support 两两不交、parallel near-overlay、Bond order/lane 数、共享有序 terminal、完整 ownership、
selected lane 不复用、禁止化学字段和所有 canonical CLI 错误合同全部保留。

标准平行 triple fixture 使用三条等宽等长 lane；相邻 endpoint disks 恰好相切形成 path，闭边界通过，
外移 1 Q4 后断开并拒绝。测试还覆盖 chain、fan、X crossing、duplicate support、odd/even 拆线、
1 Q4 near-overlay、伪造 center/index、未选中 bridge 和 endpoint 重复归属。

合法输出固定为：

```text
STRUCTURALLY_VALID_NOT_SCORED
training_authorized=false
```

测试：

```bash
.venv-train/bin/python -m unittest \
  formal.chemgraph.audit.blind_carrier_v0_2.test_validators -v
```
