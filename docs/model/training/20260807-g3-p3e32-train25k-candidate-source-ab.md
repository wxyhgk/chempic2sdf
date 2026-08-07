# G3 P3-E32 Train25k Candidate Source A/B

> 日期：2026-08-07
>
> 状态：已完成，直接替换NO-GO / 候选源互补信号成立
>
> 审计：`chemvector-g3-p3e32-train25k-candidate-source-ab`

## 假设

P3-E31确认33个tail-only Atom中23个来自stress，角色和元素并不集中；正确candidate平均排第151。
当前P3-E22 top256来自只训练5,000张、重复5轮的stride-1 center field。仓库已有训练25,000张
唯一图一轮的P2-F checkpoint；它在旧route100上补回1个Atom和1张exact，但因旧预注册门还差
1个matched Atom而正式NO-GO。

旧route100结论不等于它在当前style_train Micro160上无效。本轮不训练，固定A/B两个已有checkpoint，
重新生成同一Micro160的top256 candidate，直接验证更多唯一训练图是否改善stress候选排名。

## 固定输入

- Micro160 SHA256：
  `06f44d30a88013643be6041fe55741fd75f6d58bcf1d875a22224972a7944501`；
- A/train5k checkpoint SHA256：
  `2869451245f50f98f5e3f5259179ca338e82ad81a8fd59de2d2037a10056404b`；
- B/train25k checkpoint SHA256：
  `9e9d7725e9e9ffecde7bcbe9fbbb095f09204a480b18acbb014d88ef5ac83668`；
- 两者使用同一Atom head、vision架构、stride-1连续中心场、top256提取和6像素审计匹配；
- 只读取Micro160原PNG，不训练，不读取dev/confirm/fresh/外部或sealed_test。

## GO/NO-GO门

- 旧27张tail-required图中，train25k top128至少恢复5张；
- 旧133张top128可达图最多回退3张；
- train25k top128 Atom recall严格高于train5k；
- train25k top192仍保持160/160 candidate可达；
- 33个旧tail-only目标和160张样本身份完整。

全部通过才为`train25k_candidate_source_go`。否则停止直接复用该checkpoint；不得因它在旧route100
的微小正增益而绕过当前门。零训练审计不创建tag。

## 实际结果

| 指标 | train5k | train25k |
|---|---:|---:|
| top128 Atom recall | 99.599% | **99.757%** |
| top128 exact | 133/160 | **145/160** |
| stress top128 exact | 21/40 | **29/40** |
| top192 recall / exact | **100% / 160** | 99.939% / 155 |
| top256 recall / exact | **100% / 160** | 99.951% / 156 |
| known-Count raw recall | 95.916% | **96.403%** |
| known-Count raw exact | 53/160 | **56/160** |

对旧27张tail图，train25k恢复13张，仅使旧133张可达图回退1张；33个旧tail-only目标中18个回到
top128，新排名中位数为121、均值128.61。这证明更多唯一训练图确实显著改善了P3-E31定位的
stress候选排名，不是随机微小波动。

但train25k在top192仍漏5张stress图，top256仍漏4张，因此未通过“top192 160/160”门，正式决策
为`train25k_candidate_source_no_go`，不能直接替换train5k source。两个checkpoint的错误集合明显
互补；下一步只允许用共同stride-1网格的`source_id`做确定性候选并集审计，不使用像素聚类或修改
匹配半径。若192容量并集不能同时保留train25k恢复点和train5k兜底点，再决定是否升到256容量。

## 产物

- audit SHA256：`c85a6de8f4075a045cd80616f42cfede5856e13a0a39d0602f1bdee3359e45a5`；
- train25k top256 cache SHA256：
  `a93c4109ad2abf7d0d25e6ae87a4d10eb7204d69af7b6ef6e410b871d156bacb`；
- 实际前向耗时：16.11秒；
- dev、confirm、fresh、USPTO、MR-TADF和sealed_test均未读取。

本轮没有训练，不创建tag。
