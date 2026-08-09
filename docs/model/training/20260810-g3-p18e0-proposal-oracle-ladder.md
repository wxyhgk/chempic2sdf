# G3 P18-E0 冻结候选联合 Oracle Ladder

> 日期：2026-08-10
>
> 状态：已完成，NO-GO
>
> 类型：零训练能力审计，不创建模型 tag

## 目标

P13-P17 已连续否定局部 KEEP/MOVE、pair 和 groupwise 修补。P18 不直接重新训练 joint，先证明
P4-E3 Atom top128 与 v0.10.1 Bond top192 在同一未训练域中是否具有足够的联合可表达上限。

本轮依次测量：

1. GT Atom + frozen Bond proposal，定位 Bond 上限；
2. P4 Atom proposal + GT Bond，定位 Atom 上限；
3. 两套 frozen proposal + Oracle presence/incidence，测联合上限；
4. 将联合 Oracle 编译为无类型 skeleton，测确定性编译损失。

## 固定协议

- 数据：fresh_dev 按 `SHA256(molecule_id + "\\0" + scaffold)` 固定取 160 个完整分子，四种
  variant 各一张，共 640 张；选集确定后恢复为已固定 SHA 的源 manifest 物理顺序；
- Atom：P4-E3 `step-005000.pt`，top128、6px；
- Bond：v0.10.1 `selected.pt`，top192、无序端点 12px；
- baseline：v0.9.71 `samples-030000.pt` 在同一 fresh640 上重放；
- baseline 使用 G3 结构 tensor 直接构造 target，不调用会拒绝重合坐标的旧 ChemSVG compiler；
- 候选源分别运行自己的冻结 vision，不共享中间特征；
- source forward 完成后才读取 GT，禁止 GT count、Bond refs 或 render layer 进入模型；
- 完整候选提取重复两次，tensor digest 必须一致；
- sealed test、confirm、USPTO 与 MR-TADF 均不读取。

## GO/NO-GO

- 联合 untyped skeleton exact 不低于 70%；
- 四种 variant 各不低于 60%；
- 相对同域 v0.9.71 full-object exact 至少保留 20pp headroom；
- Oracle graph 到 skeleton 的编译损失不超过 2pp；
- fresh640 与 P4 pilot20k 的 molecule/scaffold 重叠均为 0；
- 两次候选 tensor digest 相同，冻结 state 不变。

任一门失败即停止 P18 joint 训练，并根据 Atom、Bond、联合 incidence 或 compiler 的阶梯结果
进入对应专项分支。

## 预期产物

- `outputs/audits/chemvector-g3-p18e0-proposal-oracle-ladder-fresh640/audit.json`
- `outputs/audits/chemvector-g3-p18e0-proposal-oracle-ladder-fresh640/candidate-cache.pt`
- `outputs/audits/chemvector-g3-p18e0-proposal-oracle-ladder-fresh640/fresh640.jsonl`
- `outputs/audits/chemvector-g3-p18e0-proposal-oracle-ladder-fresh640/rows.json`

## 实际结果

完整 640 图运行耗时 81.25 秒。两次冻结候选提取的 tensor SHA256 均为
`a4ed27c674c179d8d54424df8891166c2388e2af3a1d38a46676b5f221c3386b`，冻结 state、数据隔离和
compiler 门均通过。

| variant | Atom exact | Bond exact | 联合 skeleton exact |
|---|---:|---:|---:|
| coordgen | 98.13% | 93.13% | 91.25% |
| original | 98.13% | 92.50% | 91.88% |
| rdkit | 97.50% | 90.63% | 90.00% |
| stress | **86.25%** | **35.63%** | **34.38%** |
| overall | 95.00% | 77.97% | 76.88% |

同域 v0.9.71 full-object exact 为 22.03%，Oracle headroom 为 54.84pp；联合图到无类型 skeleton
的编译损失为 0。整体联合上限超过 70%，但 stress 远低于预注册的 60% 分桶门，因此正式
NO-GO，禁止进入 P18-E1 incidence 和 P18-E2 structured selector。

分阶梯因果很清楚：Atom 在 stress 仍有 86.25% exact，而 Bond 只有 35.63%；另外三种风格的
Bond exact 均超过 90%。下一步只拆解 stress Bond proposal，区分 top192 排名截断与 dense
Bond 几何不可达，不调整 incidence、presence 或 compiler。

结果文件 SHA256：

- `audit.json`：`83b033c4a0e3ed5227804335a6f5eecef49c9dbac3f89a2afa21b0ec2132978f`；
- `candidate-cache.pt`：`009d9dcf77c0d65a17530c2afff6566f7f2265adc9383f750f9b43a7f7cd88cc`；
- `fresh640.jsonl`：`7aeb5df329338faab36b0e5a3baa5bc0e078273034c6cd6718e16f66864bb9bc`。
