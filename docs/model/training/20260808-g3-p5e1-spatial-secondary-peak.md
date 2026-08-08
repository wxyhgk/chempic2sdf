# G3 P5-E1 Spatial Secondary Peak

> 日期：2026-08-08
>
> 状态：已预注册，等待训练

## 目标

P5-E0证明P4-E9B即使强制选择正确source，holdout offset也只有18/67落入6px。当前head在卷积后
使用全局空间池化，再用MLP回归方向；本轮只替换这一表示，不训练NO_ACTION或pointer。

## 固定模型

- 输入为32x32灰度crop、32x32 center-score crop和score/rank；
- 卷积始终保持32x32空间分辨率，score/rank只通过FiLM调制；
- 输出32x32 secondary-peak logits，不再直接回归二维向量；
- 有效支持为距source 4--13px的离散环带，监督目标仍严格来自Oracle `(6,12]px`；
- 推理取有效环带最大logit位置，量化误差上界小于1px；
- 参数少于5万。

## 训练与门

- 数据固定为P4-E9B fit-cache中的264个refine正动作，不加入负例；
- 每步对crop和offset同步应用8种正方形旋转/镜像之一，不加模糊、线宽或风格增强；
- 32D P4-E3 feature没有已知旋转等变协议，本轮不输入定位头，避免D4增强产生错误通道监督；
- batch64、3000步、AdamW、学习率`3e-4`、weight decay `1e-4`、clip 1；
- calibration/holdout只读取各自Oracle动作图，并强制使用正确source；
- 两域6px恢复率均至少80%、MAE均不超过3px才GO；
- 不读dev、confirm_v2、外部集或封存集。

模型名固定为`chemvector-g3-p5e1-spatial-secondary-peak-fit264`。无论GO/NO-GO，实际训练后提交
结果并创建同名annotated tag。GO后才恢复set-conditioned NO_ACTION/pointer；NO-GO则停止。
