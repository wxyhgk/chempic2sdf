# E2 v0.4 Lean 可信边界决策

## 当前结论

v0.4 合同和离线 verifier 已冻结，但当前 `HardenedLauncherV04.lean` 候选不能冻结：

```text
NO-GO for Lean freeze
execution_authorized=false
training_authorized=false
```

被否决的候选 SHA256：

```text
4d4248551703f76585c0f5b5a28f05d282a1deaa9f86496cdbc594453ba0f4ca
```

该候选已正确证明 phase/state/authority 的主要结构性质，也建立了 length-framed relation transcript
原型；但它仍没有把 raw bytes 唯一解码为 typed fields，部分 relation checker 没有真正执行冻结合同的
相等、求和、上限、分区和 framing 关系，46 个同名 sound theorem 仍是恒等投影。

## 明确的职责边界

v0.4 允许以下分工，不需要仅为语言边界升级合同：

1. 冻结的 Python executable verifier 从冻结 FD 的原始完整字节开始，负责 duplicate-key、UTF-8、
   canonical JSON、closed Draft 2020-12 schema、SHA256 和 OS/文件系统观测。
2. Lean 负责 canonical binary transcript codec、纯 relation checker 的 soundness、framing/path 注入性、
   state reachability、failure isolation 和 authority 必要条件。
3. SHA256 只作 commitment/index；authority 必须来自完整字节重开比较、NOREPLACE ledger/marker 和状态机，
   不使用 hash injectivity 或 collision-free 假设。
4. Python 解析与 Python-to-Lean typed mapping 是显式 TCB。源码、解释器、依赖、测试和完整 tree hash
   必须进入 source/tool/test freeze，报告不得称为“JSON/SHA 已由 Lean 内核证明”。

如果 transcript/TrustedDecodeEvidence 要成为新的持久化记录、证书或 authority gate，就会改变冻结的
17 schemas、R01/R22/R38/R44 和 ContractConformance，必须升级 v0.5。若它只是冻结 verifier 内部的
typed witness，不新增协议记录，则仍可在 v0.4 下实现。

## 为什么不能直接使用 Lean core JSON

Lean 4.33 的 `Json.parse` 是纯 Lean parser，但 object parser 对重复 key 使用覆盖语义，不拒绝重复键。
`FromJson` 也没有 roundtrip、injectivity、closed-schema 或 unknown-field rejection 定律。因此它可以作为
普通 parser，不能单独承担合同要求的 raw-byte 唯一性边界。

后续 transcript codec 必须自行证明：

- `decode (encode value) = .ok value`；
- successful decode 后 re-encode 等于原 canonical transcript bytes；
- encode injective；
- version/domain/tag/length 分离；
- 拒绝 trailing bytes、unknown tags、overlong/noncanonical length、overflow 和 truncation。

## SHA256 调研

审计了两个候选，均不直接引入：

- `functionally/crypto-lean4@b244222e767db168557412276dd5ea4ab9e66e67`：纯 Lean SHA256，MIT；
  依赖多个 `partial def`，没有正确性 theorem，只有 RFC6234/Wikipedia 可执行向量，上游固定 Lean 4.7。
- `SentinelOps-CI/lean-toolchain@83616b2439386a967d9a8e2268c0cea858eaf94a`：纯 Lean SHA256，MIT；
  有若干 schedule/padding/output 结构定理和 NIST 向量，但未找到实现等价 FIPS 180-4 的完整 theorem，
  且固定 Lean 4.21/mathlib。

两者可作算法参考，不能被描述成当前协议的已验证 SHA 信任根。

## 下一实现周期

1. 新建独立 transcript codec 模块及 roundtrip/injectivity/negative decode proofs。
2. Python verifier 从 raw bytes 生成内部 typed transcript，并冻结源码树、解释器和依赖闭包。
3. 为 R01-R46 分别定义 `RelationView`、`checkRxx` 和独立 `ContractObligationRxx`。
4. 每个 sound theorem 必须从 decode evidence 和 `checkRxx = .ok` 推出 obligation，禁止 `P -> P`。
5. 为每条 relation 增加正例、字段 mutation 和反例 theorem；完整输入遗漏按矩阵逐项审计。
6. 最后才把 46 obligations 接回 `AllFrozenGates -> authoritative`，重新做 Oracle 和 Red Team 双审。

在该周期完成前，不实现或运行 launcher，不创建 authorization，也不启动训练。
