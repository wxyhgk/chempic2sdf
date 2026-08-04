# G3 P3-A Count Head Train5k

> 日期：2026-08-04
>
> 状态：Count-only泛化NO-GO

## 实验

加载P2-W，冻结encoder、decoder、existence与pointer，只训练8,513参数Count head。train5k固定
10 epochs、1,570 steps、50,000 exposures；route100只在step0/final读取，不参与选模。

| 指标 | step0 | final |
|---|---:|---:|
| route100 Count exact | 3/100 | **8/100** |
| route100 Count MAE | 7.66 | **4.59**（正式batch4 joint） |
| route100 joint matched | 2399/3080 | **2674/3080** |
| route100 whole-set exact | 0/100 | 0/100 |
| train5k Count exact | - | 727/5000（14.54%） |
| train5k Count MAE | - | 4.832 |

三项协议检查相关重复项全部通过，joint matched atoms改善，但Count exact与MAE门均失败。

## 数值审计

Count-only final在batch32统计为2,953个预测原子，在正式batch4 joint路径为2,959个；两者Count exact
均为8/100。Transformer BF16在临界class logits上存在batch-shape舍入差异，因此后续正式结构指标
统一使用batch4 joint路径，训练/评测必须记录batch size。

## 决策

- `count_head_train5k_generalization_no_go`。
- 仅训练head无法从冻结的64图representation恢复精确计数；训练集自身14.54% exact说明不是单纯
  route100风格偏移，也不应继续追加相同head epoch。
- P3-B在train5k中候选完整覆盖的4,937张图上联合训练encoder/decoder、Count、existence和pointer；
  route100仍全部100张评测，不能删除两张candidate oracle不完整样本。

```text
model = chemvector-g3-p3a-count-head-train5k
checkpoint = outputs/checkpoints/chemvector-g3-p3a-count-head-train5k/step-001570.pt
SHA256 = 2eccba37a58c0b2cadf92be12153fa4697e5fbaa912778129a8571fd6ebed6d6
training images = 5,000
samples seen = 50,000
trainable parameters = 8,513
decision = NO-GO
```
