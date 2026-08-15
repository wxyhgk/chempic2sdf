# Blind Carrier v0.1 数据无关验证器

本目录只验证 Blind Carrier inbox 与 prediction 的结构完整性。它不选择数据、不读取 sealed 目录或 GT、
不运行 proposer/模型、不生成分数，也不授权训练。

固定解释器：

```bash
/home/zhenghe/GK/Project/chempic2sdf/.venv-train/bin/python
```

依赖使用该环境已有的 `jsonschema 4.26.0` 与 `Pillow 12.3.0`，未安装新包。

## 冻结 schema

验证器直接读取已提交的 Data schema，并先检查完整文件 SHA256：

```text
input.schema.json
2a6e3ac1d3553b86989bb63ac705e1ba41c754accb16ab7d834f26b0d8fbf751

prediction.schema.json
7bab948af313a43cf38b6c23e863501869798c51f33c40ebfe3dc9606963c5e2
```

JSON 解析额外拒绝 duplicate key、非有限数、非 UTF-8、非规范序列化和 schema 外字段。规范 JSON 定义为
`ensure_ascii=True`、key 排序、紧凑分隔符并以一个换行结束。

## Inbox 验证

```bash
cd /home/zhenghe/GK/Project/chempic2sdf
.venv-train/bin/python formal/chemgraph/audit/blind_carrier_v0_1/validate_inbox.py \
  formal/chemgraph/audit/blind_carrier_v0_1/fixtures/valid_inbox.json
```

检查内容：

- closed Draft 2020-12 schema；
- 严格 Base64；
- PNG signature、首个且唯一的 13-byte IHDR、全部 chunk CRC、空 IEND、无尾随字节；
- chunk 序列只能是 `IHDR, IDAT+, IEND`，IDAT 必须连续；PLTE、tEXt、iTXt、zTXt、tRNS 及全部
  ancillary/unknown chunk 一律拒绝；
- IHDR 只接受 8-bit、non-interlaced、compression/filter method 0、color type 0；RGB 即使每个像素
  都满足 `R=G=B` 也拒绝，alpha、透明隐藏颜色和彩色旁路均无效；
- 解码灰度必须按固定 filter 0、zlib level 9、单个 IDAT 重新编码，原始 PNG 必须与该结果逐字节相等；
  因此相同像素的不同压缩级别、IDAT 分片或 filter 选择也拒绝；
- IHDR 与完整解码后的尺寸均为 768×768；
- 实际解码全部像素，非白前景必须非空。

它不读取路径以外的样本文件，不接收文件名映射、SDF、annotation 或 GT。

## Prediction 验证

Prediction 的 `support_runs` 必须与同一匿名 inbox 的解码前景比较，因此 inbox 是必需参数：

```bash
.venv-train/bin/python formal/chemgraph/audit/blind_carrier_v0_1/validate_predictions.py \
  formal/chemgraph/audit/blind_carrier_v0_1/fixtures/valid_prediction.json \
  --inbox formal/chemgraph/audit/blind_carrier_v0_1/fixtures/valid_inbox.json
```

除 JSON Schema 外还检查：

- segment、terminal、lane、bond、interpretation ID 按数组顺序严格为 `0..n-1`；
- Q4 坐标与容量边界由冻结 schema 检查，相邻 polyline 点不得重复；
- 每个 RLE 非空、排序、不重叠，长度和等于 `support_pixel_count`；
- RLE 引用的每个像素都必须是 inbox 的真实解码前景；
- 不同 segment 可以在 crossing 或共同 terminal 共享少量像素，但任意两个 support set 不得相同，且每个
  segment 至少保留一个相对其他全部 segment 的 exclusive foreground pixel；
- support→geometry：每个 support pixel center 必须在对应 polyline carrier 半径内；
- geometry→support：每个 polyline vertex 及每隔 16 Q4 的 segment sample 必须有邻近 support pixel；
- terminal/lane 的 segment 引用存在、排序且唯一；
- terminal center 必须靠近每个 referenced segment 的某个 polyline endpoint，lane begin/end 必须分别匹配
  lane geometry 的端部；
- lane 两端存在且不同，并分别有引用 segment 支撑；
- Bond 的 lane 数量严格等于 order；`lane_ids` 与 alignment lane union 完全相等且排序唯一；
- 同一 Bond 的全部 lane 共享有序 semantic terminal；
- 同一 BondBundle 内各 lane 的 segment-union support 必须两两完全不相交；该限制不跨 Bond 应用，
  因此不同 Bond 的真实 X crossing 可以共享交点像素；
- 同一 Bond 的平行 lane carrier tube 不得在正长度投影上重叠；即使把同一线按奇偶像素拆成两个
  exclusive support set，或把一条 centerline 平移 1 Q4，也不能伪造双键；
- 每个 interpretation 对所有 segment 恰好给出一次、按 segment ID 排序的 ownership；
- 同一 interpretation 的 selected Bond 不得复用 lane；
- 同一 interpretation 中每条 selected lane 的 segment union 必须有相对其他 selected lane 的 exclusive
  support；
- selected Bond 的每个 lane segment 必须由对应 `bond_lane/lane_id` 拥有；
- 禁止元素、原子、SMILES、InChI、Molfile、SDF、label 和 GT 字段。

Bond hypotheses 之间可以复用 lane 作为互斥候选；只有同一个 interpretation 同时选择的 Bond 禁止复用。
这是保留视觉歧义所必需的，不等于允许同一输出图重复消费 lane。

### 固定距离常数

所有几何运算均使用整数 Q4，`1 Q4 = 1/16 pixel`，pixel center 为 `(16*x+8,16*y+8)`：

```text
AA_MARGIN_Q4 = 16
GEOMETRY_SAMPLE_STEP_Q4 = 16
carrier_radius_q4 = ceil(width_q4 / 2) + AA_MARGIN_Q4
parallel_lane_min_separation_q4 =
  ceil(left_width_q4 / 2) + ceil(right_width_q4 / 2) + AA_MARGIN_Q4
```

额外的 16 Q4 是在任何真实 inbox 出现前固定的一像素栅格量化/抗锯齿余量；采样步长同样固定为一像素，
用于阻止长 polyline 只有端点 support。比较使用整数 point-to-segment squared distance 和闭边界 `<=`，
不使用浮点数、图片风格阈值或针对 fixture 的例外。测试包含距离恰好等于 radius 的正例。
平行 lane 使用严格 `< parallel_lane_min_separation_q4` 判定 tube overlap，恰好等于边界时允许；合法双键
fixture 用两条 16 Q4 lane 和 32 Q4 centerline separation 覆盖该边界。

### Trust boundary：本验证器尚未证明的性质

本验证器只证明上述 JSON、栅格 support、局部 geometry、引用、BondBundle 和 ownership 合同。冻结 schema
中的 `segment_ids` 只能证明排序、唯一和引用存在；当一条 lane 由多个 segment 组成时，验证器尚未证明这些
segment 按几何路径连续、方向一致或具有唯一的端到端顺序。它也没有证明 `support_score` 的统计含义、
calibration certificate、coverage certificate、像素候选相对于未知 GT 的 recall/precision，或输出图的化学正确性。
这些性质必须由独立的评分/证书层处理，结构通过不得被解释为它们已经成立。

## 输出与退出码

合法文档退出 `0`，状态固定为：

```text
STRUCTURALLY_VALID_NOT_SCORED
training_authorized=false
```

任何解析、schema、PNG 或结构错误都向 stderr 输出一份 canonical JSON，字段为
`status=INVALID`、`training_authorized=false` 和结构化 `errors[]`，退出码固定为 `2`。验证器不会因为结构
通过而输出准确率、GO 或训练授权。

## 测试与 fixtures

`fixtures/` 完全由标准库生成，只包含白底黑直线，不含化学内容或真实样本。运行：

```bash
.venv-train/bin/python -m unittest \
  formal.chemgraph.audit.blind_carrier_v0_1.test_validators -v
```
