# G3 P19-E0 Direct AtomPair Oracle Ladder

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
>
> 类型：零训练复用审计，不创建模型 tag

## 目标

P18 已证明 Dense Bond endpoint residual 在 capacity64 成功但 fresh640 回退。P19 将 Bond identity
固定为无序 AtomPair，不再生成独立 Bond endpoint 或 incidence pointer。本轮先判断已有 v0.10.3
Direct AtomPair 权重能否直接复用。

## 固定协议

- 数据与顺序固定为 P18 fresh640；
- 路线 A：GT Atom anchor + 旧冻结 Atom dense feature + v0.10.3 Direct AtomPair；
- 路线 B：P4 top128 经 6px Oracle 一一选择后的 anchor + 同一旧 Atom feature + Direct AtomPair；
- Atom feature/score 在 route anchor 对应的旧 dense cell 采样，保持 v0.10.3 的 128D 输入协议；
- 枚举全部有效无序 AtomPair，不使用距离、近邻、键长或空白中点裁剪；
- 固定 v0.10.3 threshold `0.85`，不训练、不校准、不搜索 checkpoint；
- GT/P4 Oracle 只用于候选上限审计，不得称为部署结果；
- 不运行 Dense Bond、set/incidence，不读取 sealed test、confirm、USPTO 或 MR-TADF。

## GO/NO-GO

- GT Atom route skeleton exact ≥80%；
- GT Atom stress skeleton exact ≥60%；
- GT Atom edge recall ≥97%；
- P4 Oracle Atom exact 必须复现 P18 的 608/640；
- 所有冻结 state 前后完全一致。

全部通过表示旧 v0.10.3 可直接作为 P19 relation 起点；失败只表示旧权重/特征风格不可直接复用，
下一步仍可在同一稳定 AtomPair identity 上做 train-only capacity64，不能退回 Dense Bond endpoint。
