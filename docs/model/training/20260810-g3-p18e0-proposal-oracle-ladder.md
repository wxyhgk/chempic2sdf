# G3 P18-E0 冻结候选联合 Oracle Ladder

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
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
