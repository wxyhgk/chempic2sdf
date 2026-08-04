# G3 P2-J Pointer-Matched Query Set Capacity64

> 日期：2026-08-04
>
> 状态：机制GO / 4,800步严格容量NO-GO / 允许受控续训

## 受控改动

P2-J与P2-I保持相同291,394参数、64张图、初始化、优化器、4,800 steps和全局唯一pointer。
唯一改动是query-to-GT Hungarian代价：不再使用pointer概率加权平均坐标，而是使用query在每个
GT兼容候选集合上的总pointer log-probability mass。query身份、pointer监督和推理解码由此使用
同一种证据。

## Final结果

| 域 | P2-I matched/exact | P2-J matched/exact | Count exact |
|---|---:|---:|---:|
| easy32 | 943/967，12/32 | **963/967，28/32** | 32/32 |
| hard32 | 768/807，5/32 | **798/807，26/32** | 32/32 |

两个域分别只剩4和9个漏点，但严格门要求双域全Atom、32/32，因此4,800步正式仍为
`pointer_matched_query_set_capacity_no_go`。

## Checkpoint曲线

| step | easy32 | hard32 |
|---:|---:|---:|
| 800 | 867/967，0/32 | 649/807，0/32 |
| 1,600 | 878/967，1/32 | 675/807，0/32 |
| 2,400 | 915/967，5/32 | 715/807，1/32 |
| 3,200 | 934/967，10/32 | 757/807，10/32 |
| 4,000 | 954/967，20/32 | 776/807，16/32 |
| 4,800 | **963/967，28/32** | **798/807，26/32** |

exact和matched在最后三个checkpoint持续上升，没有平台。final loss约0.0349，其中object约
0.0241、pointer约0.0030、Count约0.0008。

## Pointer-linked Query上界

固定final checkpoint，以训练时同一pointer-mass Hungarian取得GT-linked query，再运行相同
唯一pointer解码：

| 域 | top-N existence重合 | linked-query pointer |
|---|---:|---:|
| easy32 | 954/967，98.66% | **967/967，32/32** |
| hard32 | 793/807，98.27% | **807/807，32/32** |

pointer与Count均已达到容量，最后断点只剩existence边界的13/14个query身份。该oracle只用于
诊断，不计入部署结果。

## 决策

- pointer-mass query matching机制GO，4,800步严格容量NO-GO。
- 由于step3200到4800仍连续提高，允许P2-K从final checkpoint恢复optimizer，固定续训到总
  8,000 steps；不得重置初始化、换数据或事后挑阈值。
- P2-K仍未双域32/32则停止继续堆epoch，转显式existence boundary loss。

```text
model = chemvector-g3-p2j-pointer-matched-query-set-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2j-pointer-matched-query-set-capacity64/step-004800.pt
SHA256 = a4e5031ae4994b73c62a9ca97625c720aa06d2f816cf322b109fed20f5f2ad0a
training images = 64
samples seen = 19,200
decision = NO-GO at 4,800 steps
```
