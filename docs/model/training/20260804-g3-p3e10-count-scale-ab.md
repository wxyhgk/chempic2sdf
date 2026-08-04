# G3 P3-E10 Count 数据量 / 容量 A/B

> 日期：2026-08-04
>
> 状态：两臂均 NO-GO；数据扩量有效，hidden32轻度欠拟合

## 协议

保持 P3-E9 输入特征、优化器、30 epoch 和 fixed-final 不变。dev256/confirm256逐样本不变。

- expanded64：训练从440增到829个molecule、1,760增到3,316图，hidden64不变。
- expanded32：相同3,316图，只将DeepSets hidden从64缩到32。
- route100不参与训练、阈值或checkpoint选择。

## 结果

| 模型 | Train exact / MAE | Dev exact / MAE | Confirm exact / MAE | Route100 exact / MAE |
|---|---:|---:|---:|---:|
| P3-E9 1760 h64 | 85.2% / 0.43 | 32.4% / 1.67 | 31.6% / 1.56 | 34% / 1.95 |
| expanded3316 h64 | 84.4% / 0.32 | 40.2% / 1.41 | **42.2% / 1.27** | **37% / 1.57** |
| expanded3316 h32 | 60.7% / 0.93 | 41.0% / 1.39 | **42.6% / 1.16** | 29% / **1.38** |

扩充molecule使confirm exact提高约10.6pp，数据量因果GO。但两个臂均未达到50% exact、confirm
MAE1.0与route100 40%的预注册门，正式NO-GO。hidden32降低confirm MAE但明显降低train和
route100 exact，说明过拟合减轻同时出现欠拟合；后续保留hidden64。

## 下一步与产物

Count的剩余错误以近邻整数为主。下一轮保持expanded3316和hidden64，只将普通CE改为
`CE + 0.1 * SmoothL1(softmax期望Count, GT Count)`，验证有序距离监督。

- h64 SHA：`9f734bf72d1f0c432dd2d7da14212bc94b013caf12849f30c752801a1a5bb3b7`
- h32 SHA：`72c9e7dc7c238f551d39000e38e8a49532128855b1d956c8b8dad5e57bf8ab21`
- 输出：`outputs/smoke/chemvector-g3-p3e10-deepsets-count-expanded3316-*`
