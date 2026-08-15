# Launcher 对抗审计

## v0.1 结论

Hardened launcher contract v0.1 的状态是：

```text
NO-GO / not_executed
```

本版本没有实现 launcher，没有生成 dry-run 授权记录，没有运行 E2 pipeline，也没有向
`/opt/chemgraph/results` 发布 v0.6 E2 结果。

这次否决不是因为 E2 的 8 对功能 witness 失败。功能层仍然是 8/8 通过；被否决的是“如何把开发
PASS 提升为权威结果”的证明链。

### 主要原因

1. 当前 Lean `Policy` 是泛型对象，可以把所有谓词设为 `True`、把 `Bytes` 设为 `Unit`，因此
   `AuthorityMarker` 没有精确绑定冻结合同、资源上限、源码和工具哈希。
2. 合同要求的 dry-run/official 两阶段授权、nonce 一次消费和 reviewer 独立性尚未进入形式状态机。
3. 合同称 watchdog 为 external，但又把 watchdog 与 monitor 一起放在 monitor cgroup；monitor OOM
   或阻塞在 fsync 时，没有独立存活者执行 1,800 秒硬 deadline。
4. `quiescenceOK`、`isolationOK`、`publicationOK` 仍是不可拆的 Prop，没有逐项证书证明 stage cgroup
   为空、4 worker 分工完整、mount/network/cpuset 不可逃逸以及 postcheck 失败后的隔离状态。
5. 状态机缺少 `renamedCandidate -> postchecked -> authoritative`。最终路径在 postcheck 前不能算权威，
   authority 必须由独立 detached marker 表示。
6. 完整 path+bytes+metadata tree、uid/gid/mode、link 类型、fsync、tool/source/test closure 尚未成为
   一等形式对象。

机器结果见 `e2_launcher_v0_1_no_go.json`。

## v0.2 最低要求

下一版必须先证明而后实现：

- bootstrap watchdog、monitor、sandbox 三角色，watchdog 不与 monitor 共失效；
- `ContractConformance` 精确绑定合同/schema/source/tool/test 哈希和固定资源值；
- dry-run/official 授权账本、唯一 nonce 和两个独立 reviewer；
- 三阶段 quiescence、隔离、资源、完整字节和 metadata 证书；
- `renamedCandidate -> postchecked -> authoritative` 状态链；
- detached authority marker 是唯一权威判定，只有最终 postcheck 后才能生成；
- 最终定理：从 initial 可达 authoritative，必然推出 v0.2 每一项冻结前门成立。

在这些条件闭合前，`training_authorized=false`。

## v0.2 结论

Hardened launcher contract v0.2 的状态仍是：

```text
NO-GO / not_executed
```

v0.2 把顶层合同和 schema 精确冻结，并引入三 sibling cgroup、授权账本、一等证书、候选结果
postcheck 与 detached marker。这些方向正确，但规范仍不能授权执行：

1. 证书只有字段清单，没有 closed schema 和字段间关系，空数组、零计数、任意摘要和自报
   `passed` 仍可能满足表面协议。
2. nonce 唯一不等于 `(contract, implementation, phase)` 只运行一次；reviewer 独立性和 issuer
   身份也没有可验证绑定。
3. watchdog 自己阻塞或 OOM 时没有更外层 deadline executor。
4. quiescence 要求先 freeze stage、再等待同一 stage 内 supervisor 退出，目标状态可能不可达。
5. 状态枚举缺少迁移表使用的 `detachedMarkerInstalled` 和失败策略使用的 `quarantine`，且
   postcheck 证书、三阶段 root framing 和 marker 唯一命名未完整定义。
6. role-specific mount、signal、cgroup、user namespace、FD 和 writable path 权限矩阵尚未冻结。

机器结果见 `e2_launcher_v0_2_no_go.json`。本版本没有实现 launcher、没有生成授权记录、没有运行
dry-run 或 official pipeline，也没有发布权威结果。

## v0.3 最低要求

v0.3 必须把 v0.2 的文字承诺变成不可真空满足的结构：

- 为 conformance、authorization、attestation、consumption、每类 certificate、postcheck 和 marker
  提供 closed schema、非空基数和跨字段关系；
- 以 `(contract, implementation, phase)` 加 nonce 作为一次消费键，并在消费前冻结授权记录 FD；
- 增加独立 deadline executor，或证明 watchdog 所有操作有界且不可阻塞；
- 使用 supervisor 退出、冻结残留树、快照为空、删除 cgroup 的可达 quiescence 顺序；
- 完整枚举成功和失败状态，定义三阶段 root framing 与 injective marker path；
- 精确冻结 11 个工具、loader 依赖闭包、证书种类全集及 role 权限矩阵；
- 最终 Lean 定理必须从 `Reach initial authoritative` 推出全部 v0.3 冻结门。

在 v0.3 双重审计和形式证明完成前，`execution_authorized=false` 且
`training_authorized=false`。

## v0.3 结论

Hardened launcher contract v0.3 的固定哈希完成红队审查后，状态是：

```text
NO-GO / not_executed
```

v0.3 已闭合 v0.2 的主要结构问题：17 类 closed record schema、46 条跨记录关系、261 条按
phase/source/event 展开的失败迁移、dry-run 与 official 分离状态机、完整 8 GiB 资源门和六角色
隔离矩阵均通过离线复核。

剩余唯一阻断位于 authorization inbox：类别计数声明 `1 + 2 + 1 = 4`，但 entries 和
`entry_count` 只要求至少 4，R29 也没有证明 `entry_count = len(entries) = 4`。因此第 5 个未分类
信任条目仍可能通过 schema，冻结的授权信任集合并不精确。

机器结果见 `e2_launcher_v0_3_no_go.json`。v0.3 没有实现 launcher，没有生成授权记录，没有运行
dry-run 或 official pipeline，也没有发布权威结果。

## v0.4 最低要求

v0.4 只允许进行可审计的最小修订：

- inbox entries 同时固定 `minItems=4` 和 `maxItems=4`；
- `entry_count` 固定为 4；
- R29 明确证明 entry count、entries 长度和三类记录计数之和都等于 4；
- v0.3 其余资源、隔离、状态机、证书和 authority 门必须逐项精确继承。

v0.4 合同、schema、Lean 和 executable verifier 全部闭合前，仍保持
`execution_authorized=false` 与 `training_authorized=false`。

## v0.4 合同冻结

v0.4 仅修复 v0.3 的 authorization inbox 精确集合问题，并通过 Oracle 全量回归和独立红队
复核。固定哈希为：

```text
contract ddb03c311e1e3a41f63d2438a8e4c7bf7b7fa9c3044fe6b15fe399e10c47a6ea
schema   dfa0e0087cf7f065f9219c5d4bdb129723a8a1b3309a7db789828a60819bc5d6
```

红队结论是 `GO-for-Lean-contract-freeze`：允许冻结合同，并实现 Lean 模型与 executable
relation verifier。该结论不授权运行 launcher、创建 dry-run/official authorization、发布 E2 权威结果
或训练模型。

机器结果见 `e2_launcher_v0_4_contract_freeze.json`。当前仍是：

```text
execution_authorized=false
training_authorized=false
```

## v0.4 离线 verifier 冻结

离线 verifier 只验证冻结 JSON、schema、framed hash、关系注册表和状态机等不依赖运行态的事实。
27 项正负测试通过，CLI 正常审计结果仍为 `NO_GO`：R27、R37、R38 静态通过，其余 43 条关系
保持 `unobserved`。输入记录通过 schema 只表示记录形状有效，不会把对应运行态关系提升为通过。

四个完整文件及目录 tree framing 见 `e2_launcher_v0_4_offline_verifier_freeze.json`。当前 verifier
不会运行 launcher、创建 authorization、写 ledger/marker 或授权训练。

## v0.4 Lean 边界

当前 Lean 候选仍是 `NO-GO for Lean freeze`。详细原因、JSON/SHA 调研和下一周期的可信边界见
`E2_V0_4_LEAN_BOUNDARY_DECISION.md`。在 transcript codec、46 个非恒等 checker soundness 和
双重审计完成前，不得把当前候选提交为权威证明，也不得运行 launcher 或训练。

## Python transcript codec 冻结

Python v0.4 transcript codec、5 组二进制 fixture 和对应 canonical JSON sidecar 已通过独立红队
审计，tree SHA256 为：

```text
a6ccdee55385d547d11b9240a81d6f8c091c45377081be34765a2db4f79d572d
```

冻结记录见 `e2_v0_4_python_transcript_codec_freeze.json`。该 codec 只保证 transcript 语法、严格
解码和有序字节编码；它不理解每个 Rxx 的合同语义。后续 checker 必须对 relation-specific inputs
进行等长、逐项、按序完整字节比较。该冻结不授权 launcher、relation gates 或训练。

Lean 端对应 codec 已证明完整字节 roundtrip、successful decode 的精确重编码、encode 单射和严格
EOF，冻结记录见 `e2_v0_4_lean_transcript_codec_freeze.json`。该结论同样只覆盖 codec。

## Constrained Recovery v0.2

v0.2 修复了零 support 假边、lane bundle、provenance、candidate coverage 和 calibration 的有限规格
漏洞，但 proposer support 与 issuer 仍未绑定真实 PNG/独立审计。机器结论见
`constrained_recovery_v0_2_review.json`：允许进入 20 图 blind carrier audit，不允许作为自足 authority，
也不授权训练。
