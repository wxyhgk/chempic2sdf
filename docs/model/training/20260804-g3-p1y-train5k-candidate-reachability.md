# G3 P1-Y Train5k Candidate Reachability

> 日期：2026-08-04
>
> 状态：固定128候选NO-GO；top1024/scale仍未全覆盖
>
> 性质：P1-X之后、train5k selector之前的冻结候选上限审计

## 为什么先审计

P1-X只在64张容量图上证明`stride2 top96 + stride1 top32`可达且selector能过拟合到100%。在
train5k上直接训练selector之前，必须先证明同一候选集合包含全部GT；否则任何下游loss都无法恢复
候选中不存在的Atom。

## 固定128结果

冻结只训练过capacity64的P1-U/P1-V，在train5k和route-select100上完整重放：

| 域 | matched | exact |
|---|---:|---:|
| train5k | 151842/155216 | 3346/5000 |
| route100 | 3032/3080 | 69/100 |

候选recall约98%，但整图合取指标远未全覆盖，因此P1-Y训练在启动前判NO-GO。缓存保存在
`outputs/audits/chemvector-g3-p1y-candidate-reachability/`，audit SHA256为
`35e4429f062af0dd9e3956f85a27425c21bc5beadc02d419666a2f971d0336c5`。

## 扩展预算审计

同一冻结模型每尺度提取top1024，并检查固定预算阶梯：

| 预算 stride2+stride1 | train recall/exact | route recall/exact |
|---|---:|---:|
| 96+32 | 97.826% / 66.92% | 98.442% / 69% |
| 128+128 | 99.474% / 89.28% | 99.513% / 89% |
| 384+384 | 99.812% / 95.02% | 99.838% / 97% |
| 1024+1024 | 99.899% / 97.32% | 99.870% / 98% |

`1024+1024`仍缺train 157个、route 4个Atom。该实验只能证明top1024/scale NO-GO，不能断言
所有regional peaks都不可达；脚本和输出明确使用
`top1024_per_scale_no_go_full_peak_reachability_unresolved`，不作过度结论。

## 结论

真正的问题是P1-U/P1-V只在64张容量图上训练，分数与表示没有覆盖5k分布。扩大selector token
到2048既没有全覆盖，也违背CPU目标。下一步先在train5k正式训练连续中心场，再重新审计128预算。

```text
audit = outputs/audits/chemvector-g3-p1y-expanded-multiscale-budgets/audit.json
SHA256 = 205e4fc7607048e4262403aed6d5a15e588409f9d16885ad6cb3564101d6c3fa
```
