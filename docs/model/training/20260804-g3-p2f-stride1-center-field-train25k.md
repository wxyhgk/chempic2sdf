# G3 P2-F Stride-1 Center Field Train25k

> 日期：2026-08-04
>
> 状态：协议通过 / route能力NO-GO / 小幅正增益

## 受控问题

P2-A使用5,000张图重复5轮。P2-F保持相同的6,250 updates和25,000 exposures，但改为25,000
张唯一图只跑1轮，用来区分“更多独立数据”与“重复更多epoch”。从P2-A final checkpoint继续，
重置优化器；新manifest排除route100的molecule和scaffold。

历史train5k与route在CSV口径存在2个scaffold重叠，因此新25k不能同时做到“包含全部旧5k”和
“route scaffold零重叠”。P2-F优先后者：新25k route molecule/scaffold overlap均为0，与旧5k
实际重叠4,876张。该限制在manifest audit中如实记录。

## 数据流恢复

首次manifest按CSV key顺序写入，但自动替换样本`train_003868`在tar中物理追加，训练在第14,873
个样本被身份检查阻止。没有错误标签参与训练。随后按tar主PNG的真实顺序重建manifest并完整预检
25,000条身份；从step2500 checkpoint恢复field、vision block和AdamW，step2600–3700 loss与
首次运行逐位相同。

```text
manifest SHA256 = b0f10eb0922599636895e19294e55b95ba9c051492b7bd69ae42a639c2846161
manifest audit SHA256 = 2c9b984c3c3fcf349e12cc8f0dd3250a60fd58c956b4affaf7aed6996ea11e53
```

## 结果

预注册能力门要求final至少`3075/3080、99/100`：

| route top128 | matched | exact |
|---|---:|---:|
| P2-A step0 | 3073/3080 | 98/100 |
| P2-F final | **3074/3080** | **99/100** |

增加独立数据补回1个Atom和1张exact，方向为正，但matched未达到3075门，正式判NO-GO。11项协议
检查全部通过，两次route重放一致。结果不支持直接扩大到100k；最后几个密集/交叉点更可能需要
改变center target或局部结构目标。

```text
model = chemvector-g3-p2f-stride1-center-field-train25k
checkpoint = outputs/checkpoints/chemvector-g3-p2f-stride1-center-field-train25k/step-006250.pt
SHA256 = 9e9d7725e9e9ffecde7bcbe9fbbb095f09204a480b18acbb014d88ef5ac83668
final metrics SHA256 = 7a1f82955b45ab564515bd10c632b00301b9b67e258cd6f919516b8cb62748f3
```
