# G3 P2-Y Joint Cardinality Decode Route100

> 日期：2026-08-04
>
> 状态：泛化断点已量化

## 协议

冻结P2-W与P2-X joint decoder，在未参与capacity64训练的route100上运行。route100与train5k的
sample key和molecule重叠均为0；重复评测一致。Action Cache不参与推理。

## 结果

| 路径 | matched atoms | Count exact | whole-set exact |
|---|---:|---:|---:|
| 128候选oracle | 3073/3080 | - | 98/100 full recall |
| 独立top-N，预测Count | 2360/3080 | 3/100 | 0/100 |
| Joint MAP，预测Count | **2399/3080** | 3/100 | 0/100 |
| 独立top-N，真实Count | 2771/3080 | 100/100 | 4/100 |
| Joint MAP，真实Count | **2827/3080** | 100/100 | 4/100 |

Joint MAP在预测/真实Count下分别增加39/56个matched atoms，方向在未见分子上仍成立。但P2-W
只在64张图训练，Count MAE为7.66；即使给真实Count，query/pointer仍比候选oracle少246个原子。

## 决策

- `count_generalization_bottleneck_confirmed`，但Count不是唯一断点。
- P3-A先冻结query/pointer，只用train5k训练Count head并在route100单次评测，隔离Count可学性。
- P3-A之后必须继续P3-B的query/pointer扩量训练；不能用Count改善掩盖known-count仅4/100的问题。
- 本项不训练模型，不创建tag。

```text
source checkpoint = outputs/smoke/chemvector-g3-p2w-fixed-identity-pointer-utility-step4800/step-004800.pt
audit = outputs/audits/chemvector-g3-p2y-joint-cardinality-decode-route100/metrics.json
route/train molecule overlap = 0
decision = count_generalization_bottleneck_confirmed
```
