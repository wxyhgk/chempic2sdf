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
