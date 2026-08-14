# EvidenceCore v0.4 独立审计

本目录只评估公开或已揭示的 pack，不参与 PNG 证据抽取，也不向 extractor 写回坐标、样本 ID 或
family 信息。

`evaluate_public_geometry_v0_4.py` 对 InkProgram 的每条两点 primitive 与 EvidenceCore primitive 做
无向、一对一匹配。匹配门限固定为两端各不超过 12 pixel；目标先最大化匹配数，再最小化端点总误差。
未匹配真值计为漏检，未匹配预测计为重复、碎裂或伪 primitive。只有一张图的真值与预测数量相等且
全部一对一匹配，才算 `geometry_exact`。

运行公开 zero-seed smoke：

```bash
python formal/chemgraph/evidence_core_v0_4/audit/evaluate_public_geometry_v0_4.py \
  --pack /tmp/v04-second-a \
  --extractor formal/chemgraph/evidence_core_v0_4/extractor/extract_evidence_core_v0_4.py \
  --output formal/chemgraph/evidence_core_v0_4/public_geometry_validation.json
```

该审计 PASS 只表示冻结 synthetic 微语法上的 primitive geometry exact，不表示真实化学图片可用，也
不授权训练。

`evaluate_public_candidates_v0_4.py` 分进程运行 extractor 和 Candidate，并用本目录独立实现的图同构
签名比较公开 GraphSet。它不导入 Candidate 的解析器或同构代码。逐图同时要求完整候选集合相等和
`Graph/Ambiguous` 决策相等。

`audit_public_stability_v0_4.py` 重跑 fresh-process byte replay、无向角 seam、固定空白角落单像素扰动、
residual 精确变化、容量上限和 closed reject record。单像素扰动允许 residual 改变，但 primitive/event
必须逐字段不变。

`audit_s8_c4_collision_v0_4.py` 只在审计目录导入冻结 generator/rasterizer，构造允许参数空间内的
S8/C4 反例。8 组 style 的 Gray8 与 canonical PNG 均逐字节相同、support gate 均通过，但 GraphSet
要求不同。该脚本不属于 extractor/Candidate 依赖闭包，也不使用 hidden seed。
