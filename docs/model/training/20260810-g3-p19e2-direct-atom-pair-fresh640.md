# G3 P19-E2 Direct AtomPair Fresh640

> 日期：2026-08-10
>
> 状态：已完成，泛化 NO-GO
>
> 类型：固定 checkpoint 零训练泛化审计，不创建模型 tag

## 目标

P19-E1 的 scratch Direct AtomPair presence 头在 train-only capacity64 达到 64/64，但 226 万
参数记住 64 张图并不困难。本轮只回答它能否泛化到未参与训练、样本选择或 checkpoint 选择的
fresh640；不训练、不调阈值、不搜索 step。

## 固定协议

- 模型固定为 P19-E1 step 2,000 best，SHA256
  `b22f23d1490e223a759a86dd230afd50d5e7f91e8036ecb45dbf22c6e11f9d47`；
- presence threshold 固定为 0.5；
- 数据、顺序、GT Atom 与 P4 Oracle Atom 路线固定复用 P19-E0 fresh640；
- 相对旧 v0.10.3 的 0.85 结果按 sample key 逐图计算 rescue/break；
- 普通三布局旧基线合计为 405/480；
- 不运行 Dense Bond、endpoint regression、incidence、chemical type 训练或 SDF 编译；
- 不读取 sealed test、confirm、USPTO 或 MR-TADF。

## GO/NO-GO

- GT Atom overall skeleton exact 至少 512/640；
- GT Atom stress 至少 96/160；
- crossed_bonds 至少 8/41；
- near_nonbonded 至少 12/34；
- coordgen/original/rdkit 合计不得低于旧基线 405/480；
- GT Atom Edge recall 至少 97%；
- 相对旧 v0.10.3 的整图 rescue 必须多于 break；
- P4 Oracle Atom exact 必须复现 608/640；
- 所有冻结 state 前后必须相同。

全部通过才表示当前 pair-local presence 训练方法具备扩量资格。若 capacity64 为 64/64 而本轮
失败，结论应写成跨 scaffold 泛化不足；下一步只能增加独立训练数据或整图 pair context，不能
退回 Dense Bond endpoint，也不能在本轮 fresh640 上调参。

## 实际结果

| GT Atom 指标 | 旧 v0.10.3 | P19-E1 capacity64 |
|---|---:|---:|
| Overall skeleton exact | 469/640（73.28%） | 305/640（47.66%） |
| 非 stress 三布局 | 405/480 | 262/480 |
| Stress | 64/160 | 43/160 |
| Crossed bonds | 0/41 | 3/41 |
| Near nonbonded | 6/34 | 9/34 |
| Edge precision | 98.78% | 93.58% |
| Edge recall | 98.41% | 97.23% |

新模型输出 28,568 条边，其中 TP 26,735、FP 1,833、FN 761。Edge recall 通过 97% 门，
但大量假边使整图 exact 显著回退。逐图相对旧模型只有 13 rescue、177 break：

- coordgen：2 rescue / 52 break；
- original：1 / 46；
- rdkit：2 / 50；
- stress：8 / 29；
- crossed_bonds：3 / 0；
- near_nonbonded：4 / 1。

P4 Oracle Atom 路线为 293/640；P4 Atom exact 精确复现 608/640，所有冻结 state 未变化。
运行没有训练、调阈值或搜索 checkpoint，也未读取 sealed test、confirm、USPTO 或 MR-TADF。

## 决策

正式决策为 `direct_atom_pair_fresh640_no_go`。只有 Edge recall、P4 Atom coverage 与冻结状态
三项通过，其余所有泛化门失败。

P19-E1 的 64/64 是容量记忆，不是泛化能力。这个结果也不否定 AtomPair identity：新目标把
crossed 从 0 提到 3、near-nonbonded 从 6 提到 9，说明它学到少量目标机制；真正问题是只用
64 张 scratch 后普通分布从 405/480 大幅降到 262/480。旧 v0.10.3 用 10k 图训练并达到
469/640，直接证据支持训练覆盖不足，而不是回到 Dense Bond endpoint。

下一周期不得在 fresh640 上调参。应从未使用的 train-only pool 固定 scaffold 隔离的 train/dev，
保留全部无序 AtomPair 与 balanced presence/listwise 目标，先做 20k 规模训练。只有内部 dev
超过旧 v0.10.3 固定基线后，才允许另开一次固定 fresh 审计；若规模训练仍出现 crossed/near
专项改善而普通分布回退，再增加整图 pair context。

## 产物

- 审计目录：`outputs/audits/chemvector-g3-p19e2-direct-atom-pair-fresh640/`；
- `audit.json` SHA256：`ef721be7f3228eef90010ffe0c013834b3ea9007419835cc28768586f836f475`；
- `rows.json` SHA256：`718a577bda795da106ad785d9a991f62941888f73eb15d0c0d8f1258de722bd2`；
- `paired-rows.json` SHA256：`26141013b37ee3ef407c34bceb57b7234c644325afdb38ce1bdd75c47d4008bc`；
- prediction SHA256：`17cae0b9bbe79e1e79ffaf5e0f54424ac08a66d6f72819414ab773b6804ebbd0`；
- 耗时：18.83 秒；
- 本轮无训练，不创建模型 tag。
