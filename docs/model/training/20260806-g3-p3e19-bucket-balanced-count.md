# G3 P3-E19 Count 区间平衡训练

> 日期：2026-08-06
>
> 状态：预注册，待运行
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

待运行后填写。

## 决策

待运行后填写。若高 Count 未明显提高，则否决单纯重加权路线，下一步改为可加和 occupancy /
Poisson-binomial Count 表示，而不是继续调区间权重。
