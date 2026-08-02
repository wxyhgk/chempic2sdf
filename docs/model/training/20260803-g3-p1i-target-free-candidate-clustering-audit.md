# G3 P1-I Target-free 候选聚合可达性审计

> 日期：2026-08-03
>
> 状态：已完成，3px target-free聚合GO
>
> 性质：零训练、只读几何审计

## 唯一问题

P1-F证明89.13%的representable GT有多个6px候选，P1-G证明动态身份监督只有约5pp收益，
P1-H又否定了训练后期assignment振荡。P1-I只回答：不看GT、不看文字或键，仅按预测anchor
距离，能否先把重复candidate折叠为target-free对象，并保持不同真实Atom可分。

## 固定协议

- 冻结P1-A、vision和top-128候选，不加载P1-G ranker；
- 只读train5k与route100，禁止confirm、fresh、sealed、USPTO、MR-TADF；
- 扫描`0 / 0.5 / 1 / 1.5 / 2 / 3 / 4 / 5 / 6px`；
- `0px`是不聚合基线；其余阈值将anchor距离不超过阈值的候选连边，取无向连通分量；
- 聚合过程完全target-free；GT只在聚合完成后评价；
- 对每个representable GT，若component内任一candidate在6px内，则该component兼容该GT；
- 用最大二部匹配计算GT到component的一一覆盖，报告Atom recall、整图exact、每GT兼容
  component数和同时兼容多个GT的碰撞component。

## 预注册门

某个非零阈值必须同时满足：

- train5k一一覆盖Atom recall至少99.9%，整图exact至少99%；
- route100一一覆盖Atom recall至少99.9%，整图exact至少99%；
- train5k仍对应多个component的GT比例不超过25%。

满足时选择最小阈值，进入component-level ranker；没有任何阈值满足则radius clustering NO-GO，
不能靠手工NMS解决重复候选，下一步需要学习型对象聚合。P1-I不训练、不生成checkpoint/SDF、
不修改前端。

## 正式结果

train5k与route100只读扫描完成，`updates=0`，没有读取禁止数据。关键阈值：

| 阈值 | Train Atom recall | Train exact | 多component GT | 平均component/图 | Route recall/exact |
|---:|---:|---:|---:|---:|---:|
| 0px | 100% | 100% | 89.13% | 128.00 | 100% / 100% |
| 1px | 99.999% | 99.98% | 83.38% | 120.55 | 100% / 100% |
| 2px | 99.998% | 99.94% | 46.69% | 89.60 | 100% / 100% |
| 3px | 99.992% | 99.74% | 20.01% | 67.20 | 100% / 100% |
| 4px | 99.970% | 99.12% | 3.97% | 50.40 | 99.967% / 99% |
| 5px | 99.902% | 97.46% | 0.46% | 43.61 | 99.870% / 96% |
| 6px | 99.861% | 96.54% | 0.06% | 41.98 | 99.837% / 96% |

3px每图平均折叠60.80个重复候选，train仅13个representable GT失去一一component覆盖，
13张图不exact；route100零损失。4px也刚好通过门，但route已有1张图和1个Atom损失，且
train最大component从3px的17增至54，链式误合并风险明显增加。

## 决策

3px与4px通过全部预注册门。按协议选择最小通过阈值：

```text
selected_threshold = 3px
decision = target_free_radius_clustering_go
```

这说明无需GT即可先把大部分重复candidate折叠成对象，而且不是依靠化学规则、文字或键长。
下一步不能直接把component当成一个坐标：P1-I的覆盖定义是component内“任一member”接近GT。
必须先审计target-free representative（centroid、medoid、base-score member）是否仍保持覆盖，
然后再训练component-level ranker。

## 产物

```text
outputs/audits/chemvector-g3-p1i-target-free-candidate-clustering/audit.json
SHA256 171df20b1112712b95ed512774972484874dc45e0584ae43e57e16da75e8e735

审计源码
SHA256 903a1d616b32853a34ae6f480ea1c064848a8d2979559fa90197416db28fec9b
```
