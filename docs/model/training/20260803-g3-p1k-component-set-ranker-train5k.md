# G3 P1-K Component Set Ranker Train5k

> 日期：2026-08-03
>
> 状态：已完成，因果GO，能力NO-GO
>
> 性质：3px target-free对象集合上的presence排序实验

## 架构与唯一训练目标

冻结P1-A和vision，将top-128 candidate按3px anchor连通分量折叠为每图平均约67个component。
每个component使用P1-J选定的base-softmax anchor和feature、最大base logit、member数与spread；
padding到128并用attention mask隔离。60,609参数单层set ranker只预测component presence residual。

训练时先用原hybrid固定representable GT集合，再将base-softmax component anchor与GT在6px内
最大基数匹配；组内仍有多个component时按当前分数动态一一选择。loss只比较有效component中
最弱8个positive与最强8个negative，不训练Count、坐标、Atom feature、Bond、OCR或编译器。

## 固定协议

- train5k 5,000张，route-select100 100张；
- 5 epochs、6,250 updates、25,000 exposures、batch 4；
- AdamW `lr=3e-4`、weight decay 0.01、clip 1.0；
- CUDA BF16、无shuffle、固定seed `20260713`；
- checkpoint 1,250/2,500/3,750/5,000/6,250；
- step 0必须逐项复现P1-J：train representable known-count recall 98.176%、exact 64.22%，
  route recall 97.461%、exact 53%；
- 禁止读取confirm、fresh、sealed、USPTO、MR-TADF；禁止生成SDF或更新前端。

## 预注册门

因果GO：全部协议通过，final相对step 0的train exact至少+5pp、route exact至少+3pp，且
train/route Atom recall均不回退。

能力GO：train exact≥75%、Atom recall≥99%；route exact≥60%、Atom recall≥98%。

因果GO但能力NO-GO时只说明component训练方向成立，不接推理；因果NO-GO时停止追加同一loss或
epoch。Count head保持冻结，deploy-count仅作诊断，不参与上述component排序门。

## 正式结果

唯一正式训练完成5 epochs、6,250 updates、25,000 exposures；含step-0全量复核与final
train5k/route100评估总耗时1,097.98秒。14项协议检查全部通过，component label、P1-A、
vision、Count logits和候选身份保持冻结，未读取任何禁止数据。

| 口径 | Step 0 | Final | 变化 |
|---|---:|---:|---:|
| Train known Atom recall | 98.176% | 98.945% | +0.769pp |
| Train known exact | 64.22% | 78.06% | +13.84pp |
| Route known Atom recall | 97.461% | 98.405% | +0.944pp |
| Route known exact | 53/100 | 68/100 | +15 |
| Train deploy-count exact | 8.52% | 10.06% | +1.54pp |
| Route deploy-count exact | 9/100 | 10/100 | +1 |

final train5k在154,911个representable GT中匹配153,277个，整图exact 3,903/5,000；route在
3,072个中匹配3,023个，整图68/100。Count未训练，deploy-count MAE保持train 3.7998、route
3.83，因此deploy整图仍受独立Count瓶颈限制。

## 决策

因果四门全部通过：train exact +13.84pp、route +15pp，且两域recall均提高。能力门中train
exact≥75%、route exact≥60%、route recall≥98%通过；train recall为98.945%，低于99%门
0.055pp，相当于还差85个匹配。按预注册结论：

```text
protocol passed
causal GO
capability NO-GO
decision = causal_go_capability_no_go
```

不能因接近99%而修改能力门，也不追加相同epoch。P1-J的component representative oracle只漏
16个GT，而final仍漏1,634个，说明剩余主要是component集合排序，不是聚合覆盖天花板。下一步
做零训练最小替换审计：计算每张失败图需要将多少selected component替换为unselected component
才能达到完整一一覆盖，再决定训练一跳action verifier还是更强结构化集合目标。

## Checkpoint与独立审计

```text
final checkpoint
outputs/checkpoints/chemvector-g3-p1k-component-set-ranker-train5k/step-006250.pt
SHA256 69243466079b326e94c6c1b356c7dc6c2c5c88aaeeda648b1cac343e0263d4ed

独立审计
outputs/audits/chemvector-g3-p1k-component-set-ranker-train5k-final/audit.json
SHA256 bdec70128d7957c098eaa9693f589a6b6195d1182a2dd2bc82712f5aa8464c11
```

独立审计校验五个checkpoint、训练器和component源码SHA、冻结lineage，并完整重算train5k、
route100和Gate。所有保存指标与重算值逐项一致，结论
`audit_passed_causal_go_capability_no_go`。
