# Owner Candidate Domain v0.5 Observation Hash 闭包加固

日期：2026-08-15

## 发现

容量逐轴分析发现，`support_input_from_observation()` 过去只检查 observation 的 closed key 集合与
`support_input_sha256` 绑定，没有检查 `png_sha256` 和 `renderer_sha256` 的类型、长度和字符集。这两个值会直接进入
candidate artifact 的 `source_binding`。

因此攻击者可以提供超长字符串，在不增加任何真实结构容量的情况下放大最终 JSON，伪造 `max_output_bytes` 的可达
fixture。它同时说明 observation 的字符串域没有真正 closed。

## 修复

两个字段现在必须同时满足：

```text
native JSON string
length = 64
characters in [0-9a-f]
```

大写十六进制、非十六进制字符、长度错误和非字符串全部 closed reject：

```text
PNG_SHA256_INVALID
RENDERER_SHA256_INVALID
```

新增真实 observation mutation 测试覆盖 65 字符、小写域外字符、大写字符和整数类型。

## 影响

本修复只收紧输入闭包，不改变冻结 observation、candidate semantics 或 artifact bytes：

```text
candidate SHA256: ff0aedcfe46f75e71320df280948a3d75edbc5dd9f5a19d13ecdd82ddf89d133
```

验证结果：

```text
owner_candidate_domain_v0_5 专项：23/23 passed
formal/chemgraph/audit 全量：       278/278 passed
Lean：                              32/32 jobs passed
determinism replay：                1000 sequential + 32 parallel, one digest
```

该修复禁止用伪 SHA padding 测试 output-byte gate，但仍未证明合法 closed R0 artifact 的精确最大字节数。该上界必须
在 capacity axis plan 中由独立的字段级 byte-bound certificate 处理。

训练继续保持 `training_authorized=false`。
