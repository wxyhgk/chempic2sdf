# G3 P6-E9 Structural Residual Arms

> 日期：2026-08-09
>
> 状态：已完成，NO-GO

固定P6-E1标准化peak prior与cap 2.0，三臂均从零residual严格复现prior开始：global、local17、
multiscale(17+41)。learned branch不读取peak logit；最终分数为`prior+2*tanh(residual)`。

train使用5,171 hard及按variant/role/degree匹配的等量easy，每hard 192次曝光、32:32 batch、D4、
AdamW `3e-4`。hard优化6px等价集合NLL；easy优化prior KL、0.25 winner hinge及0.01 residual L2。
三臂只读取calibration选模：先要求全部easy harmful=0，再按hard恢复率、MAE、参数量排序。两个
holdout cache保持000，dev、confirm和外部集不读取。每个实际模型均保存checkpoint并打同名tag。

## 结果

| arm | 参数 | calibration hard | hard MAE | easy harmful | easy保持率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| global | 115,581 | 53.81% | 6.86px | 38 | 86.35% |
| local17 | 126,486 | 50.70% | 7.21px | 61 | 90.90% |
| multiscale | 137,391 | 52.41% | 7.09px | 58 | 92.11% |

三臂都能从prior为0%的hard集合恢复约一半目标，说明residual机制确有信号；但没有任何一臂满足
2,814个easy上harmful=0，因此决策为`residual_calibration_no_winner`。按预注册规则不得读取
holdout、训练trigger或进入dev/confirm/端到端阶段。local patch没有超过global，当前问题不是
单纯缺少候选中心分辨率，而是“何时允许推翻强prior”尚未得到安全证书。

checkpoint：

- global：`outputs/checkpoints/chemvector-g3-p6e9-residual-global-hard5171/step-031026.pt`，
  SHA256 `e44c7dc26fb2106d4a06ed9fa669e08f3331206c9e86a5b52f4d38a30e8e2c1c`；
- local17：`outputs/checkpoints/chemvector-g3-p6e9-residual-local17-hard5171/step-031026.pt`，
  SHA256 `2bf5c8a84a41658fba01c1f4416ea5ce1d82c8114d49ccab763e69347e3f07da`；
- multiscale：`outputs/checkpoints/chemvector-g3-p6e9-residual-multiscale-hard5171/step-031026.pt`，
  SHA256 `e10967bfff722788106adb2ad40632384ffd2605ca404738bde09a93ec6f37d2`。

周期audit SHA256：`a1f36522f6322409ed66f64d029d62d5ad9e641554b0531f286ba213a6435fdf`；
两个holdout cache保持mode `000`，`holdout_read=false`。
