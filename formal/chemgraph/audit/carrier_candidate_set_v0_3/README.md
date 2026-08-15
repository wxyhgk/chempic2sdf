# Carrier Candidate Set v0.3 数据无关 validator

本目录验证匿名 v0.3 输入和候选集的 closed schema、规范字节、像素 support、引用、Atom/port/lane/Bond
关系、空间图类和枚举清单哈希。它不读 blind、sealed、GT、SDF 或样本身份，不运行模型，也不训练。

## 文件

- `validate_input.py`：严格 canonical JSON、base64、768x768 灰度 PNG 验证；
- `validate_candidate_set.py`：结构、几何、provenance、图类、枚举摘要和 stage trace 验证；
- `build_fixtures.py`：确定性生成单键、glyph 空图、glyph/Bond 和 crossing/junction 多图类 fixture；
- `test_validators.py`：正例、closed reject 和 mutation tests；
- `fixtures/`：仅含公开合成字节，不含 SDF、分子身份、blind 或 sealed 数据。

当前成功状态固定为：

```text
STRUCTURALLY_VALID_NOT_COMPLETE
training_authorized=false
```

这里的 `NOT_COMPLETE` 不是保守措辞，而是明确的证明边界。记录中的 `complete=true` 和 universe hash
仍是枚举器提供的声明；必须由后续独立 verifier 从冻结 boundary/ownership/attachment predicate 重新枚举
全部有限候选，才能形成 candidate completeness certificate。本 validator 不会把自签 hash 升格为证明。

当前专项结果：

```text
45 / 45 tests passed
```

验证器区分：

- `INVALID`：schema、规范字节、输入绑定、像素 support、引用、所有权或派生关系违反合同；
- 合法 `Reject(reason)`：输入超容量、无 support、超表达域或枚举器诚实承认无法完整枚举；
- `STRUCTURALLY_VALID_NOT_COMPLETE`：候选文档内部合法，但候选召回和枚举完备性尚未独立评分。

运行方式：

```bash
cd /home/zhenghe/GK/Project/chempic2sdf
.venv-train/bin/python -m \
  formal.chemgraph.audit.carrier_candidate_set_v0_3.validate_input INPUT.json

.venv-train/bin/python -m \
  formal.chemgraph.audit.carrier_candidate_set_v0_3.validate_candidate_set \
  CANDIDATE.json --input INPUT.json

.venv-train/bin/python -m \
  formal.chemgraph.audit.carrier_candidate_set_v0_3.build_fixtures

.venv-train/bin/python -m unittest \
  formal.chemgraph.audit.carrier_candidate_set_v0_3.test_validators -v
```

当前 closed schema、结构 Validator、fixture 和 mutation tests 已通过独立红队并冻结，但冻结范围仅是：

```text
schema_frozen=true
structural_validator_frozen=true
completeness_verifier_frozen=false
proposer_public_execution_authorized=false
unique_graph_authorized=false
training_authorized=false
```
