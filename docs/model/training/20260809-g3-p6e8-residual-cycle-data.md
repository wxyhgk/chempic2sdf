# G3 P6-E8 Residual Cycle Data

> 日期：2026-08-09
>
> 状态：已完成，GO

扫描P5-E3已隔离的58,376张训练图，为每个合法Atom构造确定性6--12px source，冻结P6-E1生成
最多8个候选。第一峰错误但top-8在6px可达为hard；第一峰正确样本按固定1/64哈希抽样为easy。
同一次BF16重放直接保存96px图像、center field、候选坐标/logit/mask，禁止二次改变batch身份。

按scaffold哈希做80/10/10切分。数据门：至少4,000 hard、2,000分子、1,500 scaffold，各split
非空、scaffold零重叠、easy不少于hard。仅用train hard的prior margin从`1/2/4/8`选择覆盖95%
所需的最小residual cap。不读取dev、`confirm_v2`、MR-TADF、USPTO或封存集。

## 结果

- hard 6,552、easy 27,504；
- 4,091分子、3,889 scaffold；
- train/calibration/holdout为5,171/643/738，scaffold重叠为0；
- variant为coordgen 802、original 840、rdkit 906、stress 4,004；
- train-only最小95% margin cap为`2.0`；
- 全部门通过，决策为`residual_cycle_data_go`。

产物SHA256：audit `5c543e12116d846528388d1d79a983e5817d0ef21f333eb4e3dbd8505175e016`；
hard cache `c9495efb304212c850f448d255f4db96052ef93737e95fa71e4a4192dd88016b`；
easy cache `4931a5e39ec6d3492a7e75920f20f8d62ea00a2ee2e95eb87acca7ded46972cf`。
