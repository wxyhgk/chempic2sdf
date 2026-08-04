# G3 P2-V Fixed Identity Pointer Utility Capacity64

> 日期：2026-08-04
>
> 状态：pointer utility机制GO / 严格容量NO-GO

## 实验

P2-V是P2-Q的协议修正版：冻结P2-J representation，使用P2-R正确Action Cache；每个query的
64D hidden拼接10D target-independent pointer utility，再进入9,877参数MLP。目标为固定标签
weighted BCE-only，训练1,600步。

| 域 | P2-Q错误identity | P2-V正确identity | Count exact |
|---|---:|---:|---:|
| easy32 | 955/967，22/32 | **966/967，31/32** | 32/32 |
| hard32 | 793/807，23/32 | **801/807，28/32** | 32/32 |

正确identity使easy提高9张、hard提高5张；pointer utility机制成立，但32/32严格门仍未通过。

## 学习曲线

| step | easy | hard |
|---:|---:|---:|
| 400 | 963/967，28/32 | 799/807，27/32 |
| 800 | 965/967，30/32 | 800/807，28/32 |
| 1200 | 966/967，31/32 | 800/807，28/32 |
| 1600 | 966/967，31/32 | 801/807，28/32 |

loss在终点仍下降，matched atoms与exact尚未同时平台。因此只允许一次恢复同optimizer的固定终点续训。

## 决策

- 机制结论：`fixed_identity_pointer_utility_mechanism_go`。
- 容量结论：`fixed_identity_pointer_utility_capacity_no_go`。
- P2-W完整恢复P2-V optimizer，从1,600续到预先承诺的4,800步；只用final checkpoint评判，
  不做事后选模。若仍不过门，停止增加相同epoch。

```text
model = chemvector-g3-p2v-fixed-identity-pointer-utility-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2v-fixed-identity-pointer-utility-capacity64/step-001600.pt
SHA256 = fab3efb9d06a613dac1d61f0b524d162a0223ea20143a7e5b5cb20148d1893a7
Action Cache SHA256 = d2134087c5c1d79cb698a3aefa90fcfeb6e3defeba3e602e1e1317a5362e8aca
training images = 64
samples seen = 6,400
trainable parameters = 9,877
decision = NO-GO
```
