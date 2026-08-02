# G3 P1-G 等价集合动态匹配 Train5k

> 日期：2026-08-03
>
> 状态：已完成，因果NO-GO，能力NO-GO
>
> 性质：只改变监督身份的单变量因果实验

## 已知事实

P1-F确认train5k中89.13%的representable GT Atom在6px内有多个top-128候选。P1-D的固定
身份exact为5.22%，允许候选身份置换后的几何exact为18.68%，相差13.46pp。因此P1-G不再
要求模型复现一个不可观察的固定candidate ID。

## 唯一改动

冻结P1-A Atom head、vision、top-128候选抽取和60,545参数P1-D ranker结构。每一步先用原
hybrid标签固定representable GT集合，再在每个GT的6px候选集合上做最大基数二部匹配：

1. 每个GT严格分配一个候选；
2. 每个候选最多分配给一个GT；
3. 最大基数优先，当前ranker分数只在完整匹配之间选择代表；
4. 精确分数相同时按candidate ID确定性打破平局；
5. 动态选中的M个候选作为positive，其余候选作为negative，继续使用8×8 boundary-band loss。

禁止把一个GT附近的所有候选同时标正，因为推理只需要一个Atom，该做法会训练出重复顶点。

## 固定协议

- train5k 5,000张，固定顺序，无shuffle；
- route-select100 100张，仅step 0和final读取；
- 5 epochs、6,250 updates、25,000 exposures、batch 4；
- AdamW，学习率`3e-4`、weight decay`0.01`、gradient clip`1.0`；
- CUDA BF16，确定性算法；
- checkpoint固定在1,250/2,500/3,750/5,000/6,250步；
- 不读取confirm、fresh、sealed、USPTO、MR-TADF；
- 不训练Count、Bond、OCR、ChemScene或SDF编译器。

## 预注册门

同架构P1-D基线取P1-F独立重放值：train5k几何exact`18.68%`、几何Atom recall
`94.256%`；route100 representable几何exact`17%`，标准known-count F1`93.31%`。

因果GO同时要求：

- train5k几何exact至少`28.68%`，即相对P1-D提高至少10pp；
- route100 representable几何exact不低于17%；
- route100标准known-count F1不低于92.81%，允许0.5pp统计波动；
- 全部协议检查通过。

独立能力门另设为train5k几何exact至少50%、几何Atom recall至少98%。因果GO只证明监督修正
有效，不自动代表Atom集合已经可部署。

## 输出

```text
outputs/checkpoints/chemvector-g3-p1g-equivalent-set-matching-train5k/
```

若因果NO-GO，停止该动态自匹配loss；若因果GO但能力NO-GO，下一步审计剩余几何错误，而不是
追加相同epoch或扩大ranker。

## 正式结果

唯一正式训练已完成5 epochs、6,250 updates、25,000 exposures。训练及完整train5k、route100
评估总耗时692.92秒。14项协议检查全部通过；Atom head、vision、Count向量和候选集合保持
冻结，未读取任何禁止数据。

| 指标 | P1-D固定身份监督 | P1-G动态等价匹配 | 变化 |
|---|---:|---:|---:|
| train5k几何Atom recall | 94.256% | 95.350% | +1.094pp |
| train5k几何exact | 18.68% | 24.18% | +5.50pp |
| route100几何Atom recall | 93.490% | 94.401% | +0.911pp |
| route100几何exact | 17/100 | 22/100 | +5 |
| route100标准known-count F1 | 93.31% | 94.22% | +0.91pp |
| route100标准whole-set | 17/100 | 22/100 | +5 |

train5k选择154,911个representable Atom，匹配147,707个，整图exact 1,209/5,000。route100
选择3,072个representable Atom，匹配2,900个；按全部3,080个GT的标准known-count口径匹配
2,902个。

每个epoch相对固定hybrid身份分别改写`16,582 / 16,786 / 16,819 / 16,912 / 16,886`个
candidate assignment，证明动态路径持续实际生效。该计数只比较动态身份与固定身份，未保存
相邻epoch逐样本assignment，因此不能据此断言存在或不存在跨epoch振荡。

## 决策

route护栏均通过，但train5k几何exact只提高5.50pp，低于预注册10pp因果门；24.18% exact和
95.35% recall也未达到50%/98%能力门。正式结论：

```text
protocol passed
causal NO-GO
capability NO-GO
decision = causal_no_go
```

这不是“改等价监督完全无效”：train和route在exact与Atom recall上方向一致，收益可复现；但
硬分数自匹配只能拿回约5pp，不能单独解释或解决剩余错误。按预注册停止追加同一loss和epoch。
下一步先做零训练审计，区分跨epoch硬匹配不稳定与候选组本身不能target-free去重，再决定是否
使用软集合匹配或显式候选对象聚合。

## Checkpoint与独立审计

```text
final checkpoint
outputs/checkpoints/chemvector-g3-p1g-equivalent-set-matching-train5k/step-006250.pt
SHA256 1416b06d1cac1cddf6248d5d99d2b8883f26a96f8f20362c7619b98807a9f9a1

独立审计
outputs/audits/chemvector-g3-p1g-equivalent-set-matching-train5k-final/audit.json
SHA256 3038f89cabb63e4bcf005a5c200bab5fa6c38ac7fd035dd70c766939b87333f4
```

独立审计重新加载final checkpoint、P1-A和vision，校验训练器/匹配器/checkpoint SHA与五个
checkpoint文件，并完整重算train5k、route100和Gate。所有保存指标与重算结果逐项一致，结论
为`audit_passed_causal_no_go`。
