# G3 P4-E1 Pilot Union Cache

> 日期：2026-08-07
>
> 状态：已完成，Candidate Oracle NO-GO
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

## 实际结果

双source前向与cache协议全部完成，但泛化可达性未过门：

| split | recall | exact | stress recall / exact |
|---|---:|---:|---:|
| pilot20k | 99.8831% | 19,419/20,000 | 99.6846% / 4,605/5,000 |
| dev640 | 99.4228% | 549/640 | 99.1235% / 122/160 |
| Micro160 | 100% | 160/160 | 100% / 40/40 |

256个source-id和72维feature在全部20,800张图上完整唯一，confirm_v2未读。失败不是cache实现问题，
而是两个冻结center field的并集在更广泛分布仍漏704个pilot Atom和187个dev Atom。Micro160因P3
阶段特意选择top192/256可达分子，不能代表一般candidate oracle。

最终决策`pilot_union_cache_no_go`。暂停source-aware selector训练，因为不可达GT无法通过任何
selector恢复。下一步启动P4-E2：从train25k checkpoint继续在无泄漏pilot20k训练center field，
再重建Micro/dev union；只有dev candidate recall达到预注册门才恢复selector路线。

## 产物

- audit SHA256：`6b615143a8af8dd5f7cf6bd8ccb003c01a5076facc62ad8f9eb0dfdf6cf51365`；
- pilot cache SHA256：`bc6af5eb9589cbf1b4c394852eb57335baefd338b1d931f1aa2a75ee181d2bf8`；
- dev cache SHA256：`36ca7ff6b737a252a2f218b1b01aed101bf787689d6168d202521be278338079`；
- Micro cache SHA256：`d041516f07d49c815894778ce9d262b88c3dc9e02e1d0a4afaad5577605e2a62`；
- combined tensor SHA256：`3ac66be224dcbeb20dc66e6d0c93e3b9c2938f35237071916a2496d34033b9ab`；
- 总耗时1238.48秒；confirm_v2、fresh、外部和sealed_test未读。

本轮没有训练模型，不创建tag。
