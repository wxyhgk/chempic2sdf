# G3 P3-E28 五路线周期闭环

> 日期：2026-08-07
>
> 状态：已完成，Five-route Micro Gate NO-GO
>
> 审计：`chemvector-g3-p3e28-five-route-cycle-closure`

## 目标

按P3-E21原预注册Micro门，至少2/5路线通过才允许进入dev与完整双profile SDF。本轮不训练，
只绑定P3-E25/E26/E27 fixed-final结果，并验证剩余两路是否仍可能使通过数达到2。

## 已执行三路

| 路线 | Count exact | known Atom exact | predicted Atom exact | 新增可达 exact |
|---|---:|---:|---:|---:|
| hard latent occupancy | 76/160 | 47/160 | 36/160 | 0/27 |
| soft coverage occupancy | 69/160 | 49/160 | 38/160 | 0/27 |
| query-slot occupancy | 160/160 | 11/160 | 11/160 | 0/27 |

三路均未达到known/predicted Atom exact 152/160，正式NO-GO，且各有同名annotated tag。

## Hybrid 解析剪枝

原`hybrid_structured_count`只融合结构occupancy Count posterior与P3-E16 Count posterior，不改变
candidate selector/ranking。对任何图片：

```text
predicted Atom exact
= (hybrid Count正确) AND (known-Count selector集合正确)
<= known-Count Atom exact
```

即使取已执行三路最好的known-count上限也只有49/160，远低于152/160；其新增可达上限仍为
0/27。因此该路线无需训练即可判定在当前Micro门下解析性不可行。若允许Count梯度反向改变
selector，那是新架构，不是原预注册posterior凸融合，必须另开周期。

## Bond 路线与停止证明

`bond_conditioned_occupancy`仍有合规实现路径：原PNG经过冻结v0.10.10 BondObject，输出最多
192个预测bond score/endpoints/type；top192 candidate anchor只读取这些预测端点的连续距离核，
不得使用GT bond层、旧incidence logits、硬最近邻指派或删除规则。

冻结source：

- `outputs/checkpoints/chemvector-v0101-set-first-style10k/selected.pt`；
- SHA256：`e982057d57dd4c03864eae890df12e1d8710fce60dfc452b94511d86a5f967ac`；
- vision SHA256：`1eede2e814efc0c440c26eba88bc8e9ed86e0911d7727b770a992c1ea2671c6a`。

但前三路0通过、hybrid解析失败；bond即使通过，五路最多也只有1路通过，严格小于原门2。
因此本周期必须现在停止，不能进入dev、v0.10.10/v0.10.21完整SDF、confirm_v2、外部测试或
sealed_test。继续训练bond不会改变本周期决策，只能作为下一代图条件selector新周期的证据审计。

## 决策

机器审计需验证全部输入SHA、三路decision/指标、hybrid上界与`maximum_possible_passes=1<2`，输出
`five_route_micro_gate_no_go`。本轮零训练，不创建tag；不得读取dev、旧confirm、fresh或sealed。

## 实际结果

机器审计全部检查通过：

- 已执行三路通过数：0；
- best known-count Atom exact上界：49/160；
- hybrid可过门：否；
- 剩余bond路线最大贡献：1路；
- maximum possible passes：1；
- minimum required passes：2；
- cycle can reach Micro gate：否。

最终决策为`five_route_micro_gate_no_go`。audit SHA256：
`aa1ce90bb3da5ba5043ceaf2fef208954b853e3f7e22d80d2fc060080ec2a136`。

没有运行dev、v0.10.10/v0.10.21 SDF、旧confirm、confirm_v2、fresh、USPTO、MR-TADF或
sealed_test。本轮零训练，不创建tag。下一周期若继续，应至少预注册两条真正引入新图证据的
graph-conditioned selector；第一条可使用本文冻结Bond endpoint连续incident特征，第二条必须是
独立可比较路线，不能再把只改Count的hybrid算作selector方案。
