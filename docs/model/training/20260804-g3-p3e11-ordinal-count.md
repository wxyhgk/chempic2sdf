# G3 P3-E11 Ordinal Count

> 日期：2026-08-04
>
> 状态：NO-GO；有序期望损失不改善confirm

## 协议

与 P3-E10 expanded3316 h64 完全相同，只将损失改为：

```text
CrossEntropy + 0.1 * SmoothL1(softmax期望Count, GT Count)
```

30 epoch、1,560 step、99,480 exposures，fixed-final。route100不参与选择。

## 结果

| 集合 | E10 CE exact / MAE | E11 Ordinal exact / MAE |
|---|---:|---:|
| train3316 | 84.4% / 0.32 | 83.4% / 0.33 |
| dev256 | 40.2% / 1.41 | 42.2% / 1.28 |
| confirm256 | **42.2% / 1.27** | 35.5% / 1.34 |
| route100 | 37% / 1.57 | **39%** / 1.61 |

dev和route exact局部提高，但confirm exact下降6.7pp、MAE也回退，所有正式门未过。不能利用
已看过的route100把该checkpoint选为winner。

## 决策与产物

`NO-GO`。恢复普通CE。下一步利用每个训练molecule的四种variant做batch级一致性监督，验证
风格敏感是否为Count泛化的主要来源。

- checkpoint：`step-001560.pt`
- SHA256：`4783daec7947a821def010181b2d2d2e2271260cb20672840c9c5867896f8414`
- 输出：`outputs/smoke/chemvector-g3-p3e11-ordinal-count-expanded3316-h64`
