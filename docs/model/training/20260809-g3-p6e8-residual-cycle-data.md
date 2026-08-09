# G3 P6-E8 Residual Cycle Data

> 日期：2026-08-09
>
> 状态：已预注册，等待审计

扫描P5-E3已隔离的58,376张训练图，为每个合法Atom构造确定性6--12px source，冻结P6-E1生成
最多8个候选。第一峰错误但top-8在6px可达为hard；第一峰正确样本按固定1/64哈希抽样为easy。
同一次BF16重放直接保存96px图像、center field、候选坐标/logit/mask，禁止二次改变batch身份。

按scaffold哈希做80/10/10切分。数据门：至少4,000 hard、2,000分子、1,500 scaffold，各split
非空、scaffold零重叠、easy不少于hard。仅用train hard的prior margin从`1/2/4/8`选择覆盖95%
所需的最小residual cap。不读取dev、`confirm_v2`、MR-TADF、USPTO或封存集。
