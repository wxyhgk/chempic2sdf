# G3 P16 Explicit Linear KEEP/MOVE

> 日期：2026-08-09
>
> 状态：已预注册，尚未训练

P15-E0意外发现，不含Bond的7维显式baseline在train-only三折OOF上达到0.84743 AUC，而P14的
68,877参数神经模型在calibration高分区仍由harmful主导。P16直接检验失败是否来自模型容量和优化，
不再增加任何输入信息。

## 固定协议

- 模型名：`chemvector-g3-p16-explicit-linear-keep-move-fit562`；
- 输入固定为MOVE二维offset、距离、P6 peak logit、source center score、alternative center score和
  两者差值，共7维；
- 训练集固定为P11-E1分层后的281 rescue + 281 harmful；
- 每维使用train均值/标准差标准化；
- 单一L2 logistic，7个权重加1个偏置，共8参数；
- L2固定0.01，最多64次阻尼Newton，确定性求解；
- 不使用神经网络、Bond、top128上下文、特征搜索、epoch或checkpoint选择。

训练准确率至少75%才满足容量门。之后第一次读取P11-E1 calibration标签，每个source对有效top8
逐项打分并取最高分MOVE；只在非负logit阈值中选择。至少20 rescue且0 harmful才GO，否则NO-GO。
历史holdout、confirm、MR-TADF和USPTO不得读取。正式checkpoint无论结果均保存标准化统计、权重、
偏置和阈值，提交后创建同名annotated tag。
