# G3 P6-E3 Structural Peak Selector

> 日期：2026-08-08
>
> 状态：已完成，NO-GO

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

## 结果

- 候选容量：generic 49,965/50,000、real 1,365/1,378至少存在一个6px兼容峰；
- generic fit：99.56%、0.68px；real fit：90.49%、2.60px；
- calibration：54.84%、6.78px；holdout：56.72%、5.78px；
- 四个能力门全部失败，决策为`structural_peak_selector_no_go`。

最终checkpoint：

```text
outputs/checkpoints/chemvector-g3-p6e3-structural-peak-selector50k-fit1378/step-015000.pt
SHA256 f022ad9db63dd9e8b9ef13c3c8bf28ca800d379b7f7ede706d613629f35513e7
```

P6-E2已证明top-4 Oracle达到100%/89.55%，但列表头结果几乎等于P6-E1 top-1，说明训练学会
沿用原峰排序，没有把结构token转化为跨scaffold选择能力。50k generic中原top-1本来就有
99.59%正确率，峰强度是压倒性的shortcut；继续相同配比和输入扩量没有意义。

下一步先冻结checkpoint做选择变化与候选峰强度消融，量化模型是否只复制第一峰。只有去除
峰强度后结构证据仍有可分信号，才训练hard-choice版本；否则重新设计候选级结构特征。

audit SHA256：`63cab029cb19c6dddb77667af2ad4907899db19dd0fae7b44c926df80541018f`；
generic/real候选cache SHA256分别为
`ca85d778e51b03d6379e2f95456d9d2c882254afdc479320c28db7039b484028`与
`a15ae2e1d2d26da86b61fd726adba6a89cf165fa483134be630a41dbaf621a5f`。
`confirm_v2_read=false`，`dev_and_external_read=false`。
