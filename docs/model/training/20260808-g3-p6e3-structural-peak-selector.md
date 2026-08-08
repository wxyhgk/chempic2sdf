# G3 P6-E3 Structural Peak Selector

> 日期：2026-08-08
>
> 状态：已预注册，等待训练

## 模型

模型名`chemvector-g3-p6e3-structural-peak-selector50k-fit1378`，共115,581参数。
P6-E1冻结生成最多8个target-free区域极大值；新模型把96px原图、center field、相对坐标编码为
12x12结构token，以候选偏移、峰强度、source分数和rank构成query，依次做候选到结构token的
cross-attention与8候选间self-attention，最后只输出候选分数，不生成新坐标。

## 固定训练

- 从冻结P6-E1为50k generic和1,378 real fit动作生成候选cache；
- 只用至少存在一个6px兼容候选的动作训练，不把不可达样本标成错误身份；
- 对6px内全部等价候选最大化总概率，不指定单一candidate identity；
- generic预训练10,000步，再用32 generic+32 real混合微调5,000步；
- batch64、D4、AdamW、`3e-4→1e-4`、weight decay `1e-4`、clip 1；
- calibration/holdout恢复率均至少80%、MAE均不超过3px才GO；
- 不读dev、`confirm_v2`、MR-TADF、USPTO或封存集。

无论GO/NO-GO，保存最终checkpoint、提交结果并创建与模型名一致的annotated tag。
