# Owner Candidate Domain v0.5 R0 结果

日期：2026-08-15

## 裁决

本轮有两个必须分开的裁决：

1. **R0 核心构造与有限语义验证：GO。** 对冻结的 annotation-free observation，constructor 输出的
   factorized known-intersection CSP 精确表示独立 `LegalR0` verifier 定义的全部有限合法世界。
2. **修订协议 `5a8bdd9` 的完整容量条款：NO-GO。** R0 固定只有 2 个 intersection/owner option，不可能通过
   end-to-end constructor 同时命中协议要求的 `MAX_INTERSECTIONS=8` 和 `MAX_OWNER_OPTIONS=16` 成功边界。

因此本结果只授权继续研究 candidate-domain 表示；不授权训练、UniqueGraph 或 SDF。

## 冻结结果

```text
candidate artifact SHA256:       ff0aedcfe46f75e71320df280948a3d75edbc5dd9f5a19d13ecdd82ddf89d133
pixel variables:                 58
singleton domains:               48
two-option domains:              10
adjacency edges:                188
global constraints:               7
pointwise assignments:         1024
LegalR0 assignments:            873
candidate CSP assignments:      873
sound / complete:           873/873
candidate Nodup:               true
false singleton / multi:         0/0
stored annotations covered:      2/2
```

逐 option 指标由 evidence 与 option pair 重新计算，不依赖硬编码比例：eligible `68/68` 被保留，ineligible
`48/48` 被排除。`873` 不存在于 constructor 或 candidate artifact 中，只由独立 verifier 的 truth-table 枚举得到。

## 独立性与稳健性

- Constructor 只读取冻结的 observation，不读取 stored annotation、winner 或 `873`。
- Independent verifier 不 import constructor、observation builder 或 shared common helper；它从冻结 renderer 重建
  source、domain、constraints 和 `LegalR0`。
- 真实 source 的 carrier ID 重命名、stroke/support 列表反序和端点反转会重新经过 canonicalizer；semantic CSP
  sections 保持 byte exact，source binding 正确改变。
- 不同 stroke 复用同一旧 `boundary_member_id` 会 closed reject，防止 canonicalization 抹掉 provenance 冲突。
- `1000` 次串行和 `32` 次 8-worker 逆序 replay 只产生一个 artifact digest。
- 冻结 verification 和 replay 都绑定上面的 candidate SHA；发布命令必须显式使用 `--write`。

## 验证记录

```text
owner_candidate_domain_v0_5 专项：22/22 passed
formal/chemgraph/audit 全量：       277/277 passed
Lean 工程：                         32/32 jobs passed
git diff --check：                  passed
```

Lean 模块 `OwnerCandidateDomainV05` 证明了 pointwise product 与 constrained CSP 的区分、whole-world
soundness/completeness 边界、局部 singleton 和 raw multi-option 的反例、realized ambiguity 对 single-owner constructor
的 NO-GO，以及固定 R0 无法命中原容量 ceiling。该模块没有把外部 `873` artifact 当作 Lean 定理前提。

## 明确未证明

本轮没有证明：

- 两个真实化学 semantic world；
- 两个非同构分子图生成同一 PNG；
- PNG-only carrier/intersection extractor 完备；
- fragment owner 或 residual owner 合法；
- candidate CSP 能唯一恢复 Graph、SDF；
- 当前表示已满足训练条件。

PNG ownership confirmation 只说明同一冻结 PNG/support 存在多份 locally admissible annotation；它不能自动升级为上述结论。

## 后续门

下一步必须先单独预注册 generic capacity clarification：把容量上限定义为 reject-only safety ceiling，并把
frozen-R0 semantic confirmation 与通用构造器容量确认拆成两个合同。未完成该确认前，保持
`training_authorized=false`。
