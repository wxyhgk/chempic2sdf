# Carrier Candidate Set v0.3 独立完整性探针

本目录不信任候选文档中的：

```text
complete=true
all_candidates_emitted=true
pruned_candidate_count=0
```

它从根级 `boundary_universe` 和 `compatible_boundary_pairs` 重新枚举所有端点互斥 matching，并把结果与
interpretation 中实际出现的 matching 集合比较。

## 结果状态

- `BOUNDARY_MATCHINGS_COMPLETE_INTERPRETATIONS_UNPROVABLE`：matching 完整，但 v0.3 root 不足以证明全部 interpretation；
- `BOUNDARY_MATCHINGS_INCOMPLETE`：独立枚举发现 missing/extra matching；
- `BOUNDARY_MATCHINGS_NONCANONICAL`：matching 集合相同，但 index/规范顺序不是独立枚举的唯一顺序；
- `BOUNDARY_MATCHINGS_OUTSIDE_CAPACITY`：合法 matching 超过冻结上限 128；
- `ROOT_UNAVAILABLE_FROM_REJECT`：closed reject 不携带可重放的 root universe；
- `INVALID`：输入未通过 v0.3 结构 Validator。

不存在 `INTERPRETATIONS_COMPLETE` 成功状态。这是有意的安全边界，不是未实现的默认分支。

## 为什么只能证明 matching

v0.3 根级记录包含 Boundary、compatible pair、Atom proposal 和 Direct pair，但下列候选域只存在于提交者
选择保留的 interpretation 中：

- fragment split 与 primitive；
- crossing/junction/touch region；
- ownership option；
- Atom instance candidate；
- endpoint attachment；
- port partition；
- lane fragment chain；
- Bond lane grouping。

因此一个 root-only verifier 无法同时保证 interpretation 候选的 soundness 和 completeness。允许 verifier
私下引入这些域等于修改冻结协议，不能称为验证 v0.3。

## 可执行反例

`build_counterexamples.py` 生成：

- `full_collision.json`：同一像素保留 glyph 与 Bond 两种 interpretation；
- `omitted_alternative.json`：删除 glyph alternative，刷新全部自签 count/hash/stage 后仍通过结构 Validator；
- `paired_only_matching.json`：只提交 paired matching，漏掉合法 all-unmatched matching，结构 Validator 仍通过；
- `manifest.json`：记录两份同-root文档和探针裁决。

前两份文档的 root projection 完全相同，但 interpretation 集不同。第三份文档由本探针判为
`BOUNDARY_MATCHINGS_INCOMPLETE`。

## 运行

```bash
cd /home/zhenghe/GK/Project/chempic2sdf

.venv-train/bin/python -m \
  formal.chemgraph.audit.carrier_completeness_v0_3.build_counterexamples

.venv-train/bin/python -m \
  formal.chemgraph.audit.carrier_completeness_v0_3.verify_boundary_completeness \
  formal/chemgraph/audit/carrier_completeness_v0_3/counterexamples/paired_only_matching.json \
  --input formal/chemgraph/audit/carrier_completeness_v0_3/counterexamples/input.json

.venv-train/bin/python -m unittest \
  formal.chemgraph.audit.carrier_completeness_v0_3.test_verifier -v
```

本探针不运行模型、不读取 blind/sealed/GT、不生成 SDF、不选择 winner，也不授权训练。
