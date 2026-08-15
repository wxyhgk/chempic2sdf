# E2 Launcher v0.4 离线关系验证器

本目录是冻结合同 `e2_launcher_contract_v0_4.json` 的 executable relation verifier
最小骨架，不是 launcher 实现。

它只验证不需要运行系统状态的部分：

- JSON 必须是 UTF-8、拒绝重复 key 和 `NaN`/`Infinity`/`-Infinity`，并与排序、紧凑、
  单个末尾 LF 的规范字节完全一致；序列化端同样禁止生成非有限数；
- 合同和主 schema 必须匹配冻结 SHA256；
- 主 schema 必须逐字段 `const` 精确约束合同；
- 17 个内嵌 Draft 2020-12 schema 全部合法，全部嵌套 `$id` 共 45 个且全局唯一；
- 记录通过显式 schema 名或 `schema_version` 选择内嵌 schema，不能猜测或回退；
- record bundle、state machine、relation protocol 的 domain/长度 framing SHA256 重新计算；
- 46 条 relation 必须按 `R01..R46` 完整、有序、唯一并通过 relation schema；
- 状态机 261 条失败首跳必须精确覆盖 `phase × source × event`，不存在重复或遗漏；失败状态
  图不可达 authority，dry-run/official 成功路径必须连续并进入各自唯一终点；
- R29 的 schema 必须固定四个 inbox entry，且 `entry_count=len(entries)=1+2+1=4`；
- R37 的 role matrix framing 和 R38 的 schema/$id 完整性可以离线判定。

## 安全边界

当前代码不会运行 launcher，不会创建 authorization，不会写 ledger/marker，也不会读取或伪造
cgroup、mount、namespace、pidfd、fsync、NOREPLACE、进程存活或资源峰值证据。

因此只有 R27、R37、R38 可以标记为 `passed`。其余 43 条关系统一为 `unobserved`，包括 R29：
即使 exact-four 结构和一份 inbox JSON 均通过，也无法离线证明其 root ownership、fsync、
NOREPLACE 和 full-byte membership。最终状态固定为 `NO_GO`。

## 运行离线审计

项目现有训练虚拟环境包含 `jsonschema`：

```bash
cd /home/zhenghe/GK/Project/chempic2sdf
.venv-train/bin/python formal/chemgraph/challenge_v0_6/launcher_v0_4/verify_offline.py
```

程序在纯离线检查无失败时返回码 `1`，表示“检查完成但仍是 NO_GO”；任何输入、schema、
验证器或其他普通异常均输出结构化 `NO_GO` 并返回码 `2`，不会以 traceback 或退出码 1
伪装成普通 NO_GO。它不会返回表示 launcher 获准执行的成功码。

验证一份记录时必须显式给出内嵌 schema 名：

```bash
.venv-train/bin/python formal/chemgraph/challenge_v0_6/launcher_v0_4/verify_offline.py \
  --record authorization_inbox_commitment=/path/to/canonical-inbox.json
```

没有 `schema_version` 的 `relation_obligation` 必须使用显式 schema 名。所有记录同样必须是规范
JSON 字节，普通 pretty JSON 会被拒绝。

## 单元测试

```bash
cd formal/chemgraph/challenge_v0_6/launcher_v0_4
/home/zhenghe/GK/Project/chempic2sdf/.venv-train/bin/python -m unittest -v
```

测试覆盖重复 key、非规范 JSON、固定合同哈希、三类 framed hash、17 schema 选择、46 relation
registry、261 状态首跳、role matrix、exact-four、第五个 inbox entry，以及最终报告必须保持
`NO_GO`。本目录不修改 Lean 文件；Lean 仍负责合同列出的可达性和 authority 必要充分性证明。
