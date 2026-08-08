# G3 P5-E8 Generic Direction Capacity

> 日期：2026-08-08
>
> 状态：已预注册，等待审计

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
