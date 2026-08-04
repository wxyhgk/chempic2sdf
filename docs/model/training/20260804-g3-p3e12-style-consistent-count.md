# G3 P3-E12 Style-Consistent Count

> 日期：2026-08-04
>
> 状态：NO-GO；四variant一致性未改善confirm

## 协议

恢复 P3-E10 expanded3316 h64 的普通CE。唯一变化是每个batch由16个molecule的四种variant
组成，并加入：

```text
CE + 0.1 * 四variant log-probability方差
```

数据、模型、exposure、optimizer与final-only Gate不变。

## 结果

| 集合 | E10 CE exact / MAE | E12 Style exact / MAE |
|---|---:|---:|
| train3316 | 84.4% / 0.32 | 70.7% / 0.68 |
| dev256 | 40.2% / 1.41 | 40.6% / 1.36 |
| confirm256 | **42.2% / 1.27** | 41.4% / **1.25** |
| route100 | 37% / 1.57 | **39% / 1.55** |

一致性显著降低train拟合，但confirm exact没有提高，所有正式门仍未过。route100的局部改善不能
用于反向选择checkpoint。

## 决策与产物

`NO-GO`。停止继续调DeepSets正则/loss。当前最强固定checkpoint仍为 P3-E10 expanded3316 h64。
下一方向改为结构化cardinality curve：对每个K计算最大匹配增量，学习“是否还应加入一条Atom”，
直接对齐joint matching，而不是继续做129类整图分类。

- checkpoint：`step-001560.pt`
- SHA256：`06a51cd3d5f19622d0a6c2f3f267ff818992b7e8b99b94734ef2ece97dc3f5f4`
- 输出：`outputs/smoke/chemvector-g3-p3e12-style-consistent-count-expanded3316-h64`
