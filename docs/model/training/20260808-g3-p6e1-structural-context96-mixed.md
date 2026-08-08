# G3 P6-E1 Structural Context96 Mixed

> 日期：2026-08-08
>
> 状态：已预注册，等待训练

## 模型

模型名`chemvector-g3-p6e1-structural-context96-mixed50k-fit1378`，共150,365参数：

- 96px灰度、center-field和相对坐标经五次stride-2编码到3x3，bottleneck卷积感受野覆盖完整96px；
- 对称解码回96px并截取中心32px结构feature；
- 中心32px另有精细局部分支，二者融合后仍输出32x32目标heatmap；
- 中心采样与P5逐像素一致，batch64 BF16峰值约414MiB、稳定训练步约6.4ms。

## 固定训练

- 构建50k generic与1,378真实动作的96px image/center-field cache；所有真实source边距至少55px；
- 50k generic训练10,000步，batch64、AdamW `3e-4`；
- 再以32 generic+32真实动作混合微调5,000步，学习率`1e-4`；
- D4、weight decay `1e-4`、clip 1和目标环带保持不变；
- 只保存最终checkpoint，两域恢复率均至少80%、MAE均不超过3px才GO；
- 不读dev、`confirm_v2`、外部或封存集。

无论GO/NO-GO，提交结果并创建与模型名一致的annotated tag。
