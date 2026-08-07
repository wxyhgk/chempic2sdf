# G3 P4-E1 Pilot Union Cache

> 日期：2026-08-07
>
> 状态：预注册，待构建GPU cache
>
> 审计：`chemvector-g3-p4e1-pilot-union-cache`

## 目标

P4-E0已建立无泄漏pilot20k/full80k/confirm_v2。P4-E1固定P3-E33的256 winner，对pilot20k、
Micro160和旧dev640分别运行train5k与train25k stride-1 center field，并按精确source-id合并为
可训练cache。confirm_v2保持未读。

## Candidate协议

顺序固定为`train25k前160 + train5k补齐 + train25k剩余`，容量256。每个candidate包含：

- train5k 32维视觉特征；
- train25k 32维视觉特征；
- 两源normalized score、rank和availability共6维；
- 当前anchor由train5k还是train25k提供的one-hot 2维。

总维度72。某source未进入其top256时，对应视觉特征和score为0、rank为1、availability为0。
只按共同stride-1 source-id对齐和去重，不使用像素距离/NMS。

## 门禁

- pilot20k与dev640 candidate Atom recall均不低于99.9%；
- Micro160 candidate exact为160/160；
- 每张图256个candidate ID全部唯一，feature严格为256×72；
- 输入checkpoint、manifest与P4-E0 SHA固定；
- confirm_v2、fresh、外部和sealed_test不读取。

全部通过才为`pilot_union_cache_go`，允许训练P4首条source-aware selector。零训练cache不创建tag。
