# Relation Transcript Codec v0.4

本目录实现 E2 v0.4 关系验证 transcript 的严格二进制 codec。它只负责可逆、无歧义的字节编码，
不运行 launcher、不创建 authorization，也不把 fixture 当作 authority。

## 冻结布局

编码严格对应 Lean `relationTranscriptBytes`，所有整数均为大端：

```text
u32be(domain_len)    || domain
u32be(tag_len)       || "r01" ... "r46"
u32be(owner_len)     || UTF-8 owner
u32be(verifier_len)  || UTF-8 verifier_id
u32be(theorem_len)   || UTF-8 theorem_id
u32be(source_len)    || lowercase source SHA256
u32be(tree_len)      || lowercase tree SHA256
u64be(input_count)
  repeated input_count times: u64be(input_len) || complete input bytes
u64be(result_len)    || result bytes
EOF
```

冻结 domain 为 `chemgraph-e2-v0.4/relation-transcript/v1`。owner、verifier ID、theorem ID、
offline verifier source/tree SHA256 均按 relation 固定，不能由调用者覆盖。输入列表、每个输入和
result 都必须非空，输入顺序保留且属于编码语义的一部分。

decoder 使用严格 UTF-8，拒绝：未知 domain/relation/identity、整数或计数溢出、声明长度超过剩余
字节、截断、空输入、空 result、尾随字节、大小写或 tag 别名。解析成功后必须逐字节等于重新编码
结果，确保一个语义值只有一种 v1 表示。

## 使用

```python
from transcript_codec import RelationTranscript, decode_transcript, encode_transcript

value = RelationTranscript.canonical(29, [b"auth", b"r1", b"r2", b"registry"], b"result")
raw = encode_transcript(value)
assert decode_transcript(raw) == value
```

codec 不声称输入 bytes 是正确的化学、JSON 或关系证据；这些内容必须由冻结 executable verifier
从原始 FD 重算。codec 也不赋予 transcript 任何授权或权威性。

## Fixtures

`fixtures/` 包含至少五份确定性 `.bin` 及规范 JSON sidecar。sidecar 使用十六进制表达输入和
result，并显式记录 `authority=false`、`authorization_record=false`。重新生成：

```bash
cd formal/chemgraph/challenge_v0_6/transcript_codec_v0_4
python build_fixtures.py
```

## 测试

```bash
python -m unittest -v test_transcript_codec.py
```

测试覆盖 46 个 relation roundtrip、冻结 metadata、未知值、UTF-8、空值、overflow、截断、尾随、
非规范别名、metadata mutation、已知固定 fixture 及全部 sidecar 对照。
