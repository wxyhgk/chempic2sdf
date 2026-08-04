# G3 P1-Z Stride-2 Center Field Train5k

> 日期：2026-08-04
>
> 状态：协议通过 / route因果GO

## 协议

从P1-U capacity64 checkpoint初始化stride-2连续中心场及视觉最后一个stride-4 block，重置
AdamW状态。在固定train5k上训练5 epochs、6250 updates、25000 exposures，batch 4、BF16、
学习率`3e-4`。Atom head和其余视觉骨干冻结；route100只在step0和final读取。

预注册GO要求route top128 recall至少增加0.5pp、exact至少增加10张，并逐项复核P1-Y独立step0。

## 结果

| route top128 | matched | recall | exact |
|---|---:|---:|---:|
| step0 capacity64 | 2989/3080 | 97.045% | 49/100 |
| train5k final | **3071/3080** | **99.708%** | **96/100** |

11项协议检查全部通过，两次final route重放一致。训练数据扩量同时大幅提高recall与整图exact，
证明此前候选覆盖的主要限制确实是中心场只训练了64张图。

```text
model = chemvector-g3-p1z-stride2-center-field-train5k
checkpoint = outputs/checkpoints/chemvector-g3-p1z-stride2-center-field-train5k/step-006250.pt
SHA256 = fb0398d2d5e4c8d5b0d83405892df3469054051aa3627db554df80e35e782753
final metrics SHA256 = 1fdb39ad5cb5e9e81be876ba7515a3930010bc6bd64884e6da7a322ee9f5a493
```
