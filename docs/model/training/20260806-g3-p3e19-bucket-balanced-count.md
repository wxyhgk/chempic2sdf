# G3 P3-E19 Count 区间平衡训练

> 日期：2026-08-06
>
> 状态：完成，严格 NO-GO
>
> 基础 checkpoint：P3-E16 `step-008665.pt`
>
> 输出：`outputs/smoke/chemvector-g3-p3e19-bucket-balanced-count-epoch1-h64`

## 目标

P3-E18 发现 held-out 字体并非 Count 下降的主因：相同 Count 区间在 style90k 与 fresh_dev 的
准确率接近，但 style90k 有66,020张21–40原子图，61–80仅2,372张。本轮验证一个单一假设：
P3-E16 的高 Count 残差主要由交叉熵的区间贡献失衡造成。

## 固定协议

- 精确恢复 P3-E16 模型与 AdamW optimizer；
- h64 DeepSets、输入 evidence、学习率、weight decay、梯度裁剪全部不变；
- style90k 每张图恰好再看一次，1,421 step、90,932 exposure；
- 唯一变化：每张图的 CE 按所属 Count 区间乘 `N/(5*N_bucket)`；
- 五个非空区间对总 loss 的理论贡献相等，平均样本权重保持1；
- fixed-final only，不保存或选择中间 checkpoint；
- fresh_dev 不训练、不调权重、不看中间结果，只在最终模型评估一次；
- Count 评估固定 batch16，与 P3-E18 正式口径一致；训练启动前必须逐值复现 P3-E18 基线；
- route100只报告，不参与决策；sealed_test和MR-TADF不读取。

## 预注册门

- fresh_dev Count exact 相对 P3-E18 的36.2625%至少提高5pp；
- fresh_dev MAE 相对1.98075至少下降0.25；
- 61–80原子 exact 相对9.65%至少提高10pp；
- 1–20与21–40各自回退不超过5pp；
- 原 confirm256 exact不低于45%，MAE不高于1.2；
- 任一能力门失败即为NO-GO，不用中间step补选。

## 结果

训练完成1,421 step、90,932 exposure，完整覆盖每张style图一次。协议检查全部通过。加权训练
平均原始梯度范数为8.024，平均clip scale为0.178，说明逆频率权重确实产生了很强的更新信号。

| 集合/指标 | P3-E16 | P3-E19 | 变化 |
|---|---:|---:|---:|
| fresh_dev exact | 36.2625% | 39.2125% | +2.95pp |
| fresh_dev MAE | 1.9808 | 1.7961 | -0.1846 |
| fresh 1–20 exact | 55.80% | 61.45% | +5.65pp |
| fresh 21–40 exact | 52.55% | 50.35% | -2.20pp |
| fresh 41–60 exact | 27.05% | 34.55% | +7.50pp |
| fresh 61–80 exact | 9.65% | 10.50% | +0.85pp |
| confirm256 exact | 50.00% | 48.83% | -1.17pp |
| confirm256 MAE | 0.926 | 0.883 | 改善 |

七个预注册能力门中，常见Count回归保护与confirm保护四门通过；fresh总exact +5pp、MAE
下降0.25、61–80 +10pp三门失败。route100提高到50/100只是历史诊断，不参与决策。

训练域也出现相同形态：41–60从26.87%提高到31.93%，61–80只从12.10%提高到13.03%。
因此61–80失败不能归因于新字体；即使五个区间对loss的理论贡献相等，全局129类分类头仍未
学会可加和的对象计数。

## 决策

`bucket_balanced_count_no_go`。分布失衡假设只得到部分支持，单纯逆频率重加权正式否决；不追加
相同加权epoch，也不根据route100选择该模型。下一步改为可加和occupancy / Poisson-binomial
Count表示，让总Count由128个候选的存在概率相加得到，而不是直接从全局特征分类129个类别。

## 产物

- checkpoint：`outputs/smoke/chemvector-g3-p3e19-bucket-balanced-count-epoch1-h64/step-010086.pt`；
- SHA256：`d44245087edf331ceca672608ab3fd80dad45ac62a39f533fb5673a3beb31382`；
- 训练时间：8.37秒（包括全部最终Count评估，不包括启动时evidence加载）；
- 本模型NO-GO，不替换P3-E16 winner；
- 文档复核日期：2026-08-06。
