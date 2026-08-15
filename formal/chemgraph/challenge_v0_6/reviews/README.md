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
