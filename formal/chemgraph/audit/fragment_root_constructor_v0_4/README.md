# Fragment Root Constructor v0.4 几何微探针

本目录从 canonical base stroke 线段用整数 cross product 和 Python `Fraction` 精确求交，再把每条 stroke 在所有
内部交点统一切成相邻 atomic fragment。它不使用浮点数，也不会先舍入到 Q4 后合并不同交点。

这一步只处理 geometry：

- proper X 的两条 stroke 都切分；
- T touch 只切分交点位于内部的 stroke，不生成零长 child；
- 多线同点只为每条 parent 生成一个 cut；
- crossing、junction、touch、glyph、bond 和 merge 决策留给后续 CSP；
- collinear positive overlap 当前 closed failure。

当前输入的 base stroke 仍是研究 fixture，不是从 PNG 独立验证过的完整 root。输出也不包含 support RLE、width、
primitive kind、ownership、port、lane 或 Bond。因此状态只能是：

```text
GEOMETRY_CONSTRUCTOR_RESEARCH_ONLY
root_extractor_complete=false
fragment_payload_complete=false
training_authorized=false
```

运行：

```bash
.venv-train/bin/python -m unittest \
  formal.chemgraph.audit.fragment_root_constructor_v0_4.test_fragment_root -v
```
