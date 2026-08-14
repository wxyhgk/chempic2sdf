# EvidenceCore v0.4 P3 预注册

日期：2026-08-15

状态：`preregistered_before_p3_implementation`

P0/P1/P2 已按顺序提交。P3 只实现：

```text
PNG bytes -> bounded EvidenceCore -> anonymous CandidateGraphSet -> Graph | Ambiguous | Reject
```

P3 期间禁止请求 custodian share、禁止生成正式 hidden A/B、禁止训练。

## 1. 进程边界

Extractor：

```text
stdin  = 一个完整 768x768 canonical PNG bytes
stdout = 一个 canonical EvidenceCore JSON
```

Candidate：

```text
stdin  = 一个 canonical EvidenceCore JSON
stdout = 一个 canonical CandidateSet JSON
```

Candidate 不能看到 PNG、文件名、sample ID、pack 顺序、seed、family、transform、InkProgram、sealed GT、
support certificate 或 generator rejection log。Extractor 不能看到 GT 或 candidate oracle。

## 2. EvidenceCore 容量

EvidenceCore schema 在实现前冻结：

- 最多 64 个 `stroke_primitive`；
- 每个 primitive 只含：两端 Q4 pixel coordinate、Q4 width、support score 0--10000、local residual count、
  orientation bucket、closed/straight flag；
- 最多 64 个 `event_region`；
- event 只含：Q4 centroid、Q4 bbox、incident primitive ID list、neutral kind hypothesis 与 score；
- neutral kind 只能是 `terminal/junction/intersection/glyph_like/unknown`，不能写 chemical truth；
- 全图 residual 只含总 pixel count、component count 和整体 bbox，不含位置列表；
- JSON 数值全为整数，无浮点、NaN 或任意精度字符串；
- canonical stdout 不超过 65,536 bytes；
- primitive/event ID 只由 canonical geometry 排序产生。

允许内部使用完整 raster 计算，但输出不得无损还原 raster。

## 3. 明确禁止的 Evidence 字段

- foreground row-runs、RLE、bitmap、PNG bytes 或任意压缩 raster；
- 完整 pixel/skeleton/edge-point list；
- PNG/tensor SHA、opaque blob、base64、逐像素 embedding；
- 超过冻结上限的细碎 primitive，用 primitive 列表编码像素；
- Scene/Atom/Bond ID、family、Graph signature、bundle ownership；
- crossing/junction 真值、键级真值、OCR 文本真值；
- 可读取外部 cache 的 key。

静态 schema、stdout byte limit、数值范围、信息容量上界和 adversarial serializer 测试必须同时通过。

## 4. CandidateGraph

Candidate 输出 anonymous graph：

```json
{
  "nodes": [{"id": 0, "role": "implicit|text_stroke"}],
  "edges": [{"begin": 0, "end": 1, "order": 1}],
  "role_hypotheses": []
}
```

- 每图最多 128 nodes、128 edges；
- node ID 必须从 0 连续编号，但评分按独立 graph isomorphism，不信 node 顺序；
- edge order 仅 1/2/3；
- C4 只用 `text_stroke` sentinel，不输出 I/l/1/slash 字符；
- CandidateSet 最多 64 个去同构候选；
- 0 个候选必须 Reject，1 个候选 Graph，多个非同构候选 Ambiguous；
- Candidate 不得调用现有 graph canonicalizer，独立 verifier 另做同构。

## 5. Candidate 独立闭包

Candidate 可以读取公开 EvidenceCore schema 和一份手写、实现无关的 geometry interpretation rules 文档。
它禁止 import 或读取：

- `generate_pack_v0_4.py`、`rasterizer_v0_4.py`；
- Scene、fixture、legacy grammar/classifier/canonicalizer；
- pack auditor、sealed GT、GraphSet 或 support certificate；
- PNG decoder、OpenCV image object 或 raster bytes。

Extractor 与 Candidate 分目录、分进程、分依赖审计。Candidate 测试 fixture 必须是手写 EvidenceCore，
不能由 generator GT 自动转换。

## 6. 可用视觉依赖

P3 extractor 允许冻结使用：

```text
Python 3.13.5
NumPy 2.4.3
OpenCV 5.0.0
```

允许的目的仅是 PNG 解码、threshold/morphology、line/contour/component detection。P3 freeze record 必须记录
模块路径、版本、源/二进制文件 manifest hash 和 OpenCV build information hash。Candidate 不得 import
NumPy/OpenCV。

## 7. 公开开发集

允许使用固定公开 zero-seed smoke 做开发：48 observation / 72 invocation。允许读取它的 sealed GT 用于
诊断 extractor，但不得把 sample hash、坐标表或 family-specific observation 表写入实现。

公开开发门：

- primitive geometry extraction recall/precision 分 bucket 报告；
- CandidateSet 对公开 GraphSet 逐 observation 100% set equality；
- strict Graph 100%，collision Ambiguous 100%，错误 Graph 0；
- C1/C2/C3/C4 分别报告，不只总平均；
- EvidenceCore stdout 上限与禁止字段 100%；
- 同 PNG A/B replay 输出逐字节一致。

公开 smoke PASS 只允许 freeze implementation，不是 hidden 结果。

## 8. Freeze 后未知 P3 challenge

P3 implementation freeze 后、正式 custodian share 仍不 reveal。由一个未参与 extractor/candidate 实现的
reviewer 选择一次性 diagnostic seed，生成新的 48/72 public-review pack；实现不能修改。

要求：

- 与 zero-seed PNG hash overlap 为 0；
- CandidateSet exact 100%；
- strict/ambiguity decision 100%；
- extractor/candidate A/B replay byte exact；
- independent verifier 从 EvidenceCore 与 CandidateSet 重算，不信 summary。

该 diagnostic seed reveal 后永久退役。它不是 P4 正式 hidden seed，也不能用于 P3 失败后的同版本重测。

## 9. P3 GO / NO-GO

全部是合取门：

1. P3 prereg commit 早于 implementation；
2. EvidenceCore schema closed 且容量门通过；
3. extractor 无 GT/Scene/generator 依赖；
4. Candidate 只读 EvidenceCore，无 PNG/GT/generator/canonicalizer 依赖；
5. public zero-seed CandidateSet exact 100%；
6. freeze 后一次性 diagnostic CandidateSet exact 100%；
7. exact collision 必须保留 multiple candidates；
8. strict 错误 Graph 为 0；
9. A/B deterministic replay bytes 相同；
10. 两名独立 reviewer 批准依赖闭包、因果链和逐样本结果；
11. `training_authorized=false`。

任一项失败，P3 为 NO-GO。禁止删样本、改 diagnostic seed、把 Reject 算作 strict correct、改 schema 上限
或回退到完整 raster payload。

## 10. 不声称

- 不声称 line detector 已理解化学；
- 不声称 P3 synthetic PASS 等于真实 benchmark；
- 不声称 C4 是 OCR；
- 不声称 wedge/dash/color/text abbreviation 已解决；
- 不声称 P3 PASS 允许训练或正式 hidden pack；
- 不声称 Lean 能证明 OpenCV/Python 实现正确。
