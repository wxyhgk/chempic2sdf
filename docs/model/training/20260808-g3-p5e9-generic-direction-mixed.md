# G3 P5-E9 Generic Direction Mixed

> 日期：2026-08-08
>
> 状态：已预注册，等待训练

## 目标

P5-E8提供50,000个一般原子source→anchor方向对。本轮保持P5-E5的45,353参数完整crop头，验证
通用方向预训练能否解决degree=2等局部几何的跨scaffold方向泛化。

## 固定训练

模型名：`chemvector-g3-p5e9-generic-direction-mixed50k-fit1378`。

1. 冻结P4-E3对fit图在线生成50k image/center-field crop cache；generic candidate score/rank置0；
2. 随机初始化方向头，50k通用对训练10,000步，batch64、AdamW `3e-4`；
3. 不保存阶段一checkpoint，只在线记录原calibration/holdout指标；
4. 继续5,000步混合微调，每批32通用对+32真实残差，学习率降为`1e-4`；
5. D4增强、weight decay `1e-4`、clip 1与既有heatmap loss保持不变。

最终calibration/holdout 6px恢复率都至少80%、MAE都不超过3px才GO。不读dev、`confirm_v2`、
外部或封存集。只保存最终checkpoint；无论GO/NO-GO，提交结果并创建与模型名相同的annotated tag。
