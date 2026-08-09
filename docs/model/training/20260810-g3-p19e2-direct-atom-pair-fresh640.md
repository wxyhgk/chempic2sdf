# G3 P19-E2 Direct AtomPair Fresh640

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
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
