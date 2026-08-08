# G3 P5-E8 Generic Direction Capacity

> 日期：2026-08-08
>
> 状态：已完成，GO

## 目标

P5-E6/E7确认当前残差的主要失败是degree=2等局部多线结构中的方向泛化，而不是OCR、距离或
预测命中另一GT。本轮不训练，验证能否从普通原子构造足够大的通用source→anchor方向监督。

## 固定数据协议

- 只用P4-E9 fit 14,000图/3,500分子；其scaffold与calibration/holdout已隔离；
- 每个渲染原子按`SHA256(seed, sample_key, target_index)`生成一个连续角度和`(6,12]px`半径；
- synthetic source的32x32 crop必须完整在画布内；
- 目标必须是source唯一最近GT，且与第二近GT距离差至少1px；
- 合格池按稳定hash每个variant选择12,500对，共50,000对；不针对MR-TADF或特定分子采样。

## 容量门

- 精确50,000对，四variant各12,500；
- degree=2至少15,000对；
- `TEXT`至少5,000对；
- 全部offset严格位于`(6,12]px`；
- 不读dev、`confirm_v2`、外部或封存集。

全部通过才允许运行冻结P4-E3骨干构建图像/center-field crop cache，并预注册通用方向预训练；
否则停止该路线。本轮不训练，不创建tag。

## 实际结果

14,000图共有422,364个渲染原子，得到421,122个无歧义有效方向对；908个因另一GT更近被拒，
334个因最近/第二近距离差不足1px被拒。稳定选择的50,000对分布为：

| 维度 | 数量 |
|---|---:|
| 分子 / scaffold | 3,499 / 3,311 |
| coordgen/original/rdkit/stress | 12,500 / 12,500 / 12,500 / 12,500 |
| degree1 / degree2 / degree3plus | 8,730 / 24,753 / 16,517 |
| IMPLICIT / TEXT | 36,782 / 13,218 |

全部容量门通过，最终`generic_direction_capacity_go`。这使方向监督从1,378个稀有残差扩大到
50,000个一般原子局部，同时degree=2和文字角色都有充分覆盖。允许构建冻结P4-E3 image/
center-field crop cache并训练一个两阶段模型：通用方向预训练后，用通用/真实残差混合批次微调。

audit SHA256：`951f7a8b0e36b1d551727f5a6d11c0bd582bd0c215aac42aa0706308b2e2b209`；
pairs SHA256：`d45ad1bc6f3ba88e8a88593a66a2ea6c2ca4a76849bd552ce72cbbf0f28a9faa`。本轮没有训练，
不创建tag，dev、`confirm_v2`、外部和封存集均未读取。
