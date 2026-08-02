# G3 P1-F 等价候选监督审计

> 日期：2026-08-03
>
> 状态：已完成
>
> 性质：零训练、固定checkpoint的监督身份审计

## 唯一问题

P1-C到P1-E都把top-128中的一个固定candidate指定为每个GT Atom的hybrid positive，同一GT
6px内其余候选均作为negative。模型最终按几何一一匹配评分，固定candidate ID不是化学输出。
P1-F只回答：当前低whole-set/boundary exact有多少来自“几何等价候选被错误当成负例”。

## 固定输入

- P1-D final：`a8fd94c...afd97c`；
- P1-E CAP/SHAM/ALIGN final：分别`5c312969...dc393`、`573d4525...75039`、
  `445d7591...29aa4`；
- P1-A、vision、top-128、hybrid matcher、train5k和route100全部冻结；
- SHAM使用train选定seed `31033`，不重新选seed；
- 禁止读取confirm、fresh、sealed、USPTO、MR-TADF或其他外部数据。

## 两种同口径Exact

每图令`M`为hybrid representable GT数，保持模型score不变并选择top-M：

1. `fixed_identity_exact`：selected candidate ID集合严格等于固定hybrid-positive ID集合；
2. `geometry_equivalent_exact`：selected M个predicted anchor与这M个representable GT anchor在
   6px内存在完整一一匹配，不要求candidate ID等于固定positive。

同时报告匹配Atom数、`geometry exact && !fixed exact`的identity-artifact rescue，以及selected
fixed-negative中落在任一representable GT 6px内的数量。

## 候选多重度

对每个representable GT统计top-128中6px候选数，报告1/2/3/4/5+分布、均值、p50/p90/max和
至少2个候选的GT比例。该统计与模型无关，只计算一次。

## 判读

- geometry exact相对fixed exact提高`>=10pp`，且至少25%的GT有多个6px候选：固定身份监督是
  主要可操作瓶颈，下一步转等价组/集合匹配loss；
- 提高`3..10pp`：身份监督是次要瓶颈，不能单独解释失败；
- 提高`<3pp`：否定固定身份是主因，转更高分辨率对象证据或目标定义审计。

只按完整train5k作主判读，route100只作同口径迁移诊断。P1-F不保存checkpoint、不训练、不
修改前端、不生成SDF。

## 正式结果

唯一只读审计已完成，`updates=0`，没有读取confirm、fresh、sealed、USPTO、MR-TADF或其他
外部数据。train5k共5,000张图、154,911个representable GT Atom；route100共100张图、
3,072个representable GT Atom。

train5k候选多重度：

| 6px内候选数 | GT Atom数 |
|---:|---:|
| 1 | 16,840 |
| 2 | 45,208 |
| 3 | 30,500 |
| 4 | 37,235 |
| 5+ | 25,128 |

平均每个GT有`3.193`个候选，p50为3、p90为5、最大16；`138,071 / 154,911 =
89.13%`的GT至少有两个6px内候选。固定candidate identity不是偶发歧义，而是绝大多数Atom
都会遇到的系统性歧义。

完整train5k结果：

| 模型 | 固定身份Atom recall | 几何等价Atom recall | 固定身份exact | 几何等价exact | 提升 |
|---|---:|---:|---:|---:|---:|
| P1-D | 88.676% | 94.256% | 5.22% | 18.68% | +13.46pp |
| CAP | 84.869% | 92.241% | 5.26% | 19.06% | +13.80pp |
| SHAM | 88.760% | 94.209% | 5.64% | 18.18% | +12.54pp |
| ALIGN | 88.400% | 93.981% | 5.96% | 19.92% | +13.96pp |

route100同样复现：92.06%的representable GT有多个候选；四臂几何等价exact相对固定身份
exact提高`11..13pp`。因此结果不是train5k独有的统计现象。

## 决策

预注册的主要瓶颈门要求exact提升至少10pp且多候选GT至少25%。实测四臂全部提升超过10pp，
多候选率达到89.13%，正式判定：

```text
fixed_identity_major_bottleneck
```

这证明P1-C到P1-E的固定身份监督错误地惩罚了大量几何等价候选，也解释了为何扩大ranker或增加
局部patch仍难以拟合固定标签。它不等于Atom集合问题已解决：修正评价后最佳几何exact仍只有
19.92%。下一步必须使用“一组候选只动态选一个”的一一匹配监督，不能简单把同组全部标正，
否则会鼓励同一原子产生重复顶点。

## 产物

```text
outputs/audits/chemvector-g3-p1f-equivalent-candidate-supervision-train5k/audit.json
SHA256 3d024bac88199b52d1bc5bdbc53dae500a83c695665d49c135bd876e5515801a

审计源码
SHA256 ee276a76b870f76a4148851f205420c20fd6bd34bbc1e0dbe2bb68b82c0218ae
```
