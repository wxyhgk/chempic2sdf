# G3 P2-G 渲染可恢复性审计

> 日期：2026-08-04
>
> 状态：审计通过 / P2-F正式NO-GO不变 / 残差定位完成

## 受控问题

P2-F把中心场训练从5,000张重复图扩到25,000张唯一图后，route100达到`3074/3080、99/100`。
本审计回答最后一张失败究竟属于模型容量、top-K候选预算，还是合成stress图已经丢失可分辨证据。

审计不修改图片、标签、manifest或checkpoint，也不删除失败样本。完整route100继续是正式主指标，
可恢复性分桶只作为并列诊断，不能把P2-F的NO-GO改写成GO。

## 判据

初始纯几何规则将任意小于6px的原子中心和任意单条非入射键穿心都判为困难，route100共标记7张；
但其中6张已被P2-F完整命中，说明该规则过于保守。最终规则读取真实`layers.json`中的
`visible_text`，不再用元素类型猜测文字是否显示：

1. `anchor_alias`：两个都无可见文字的Atom anchor距离小于6px。
2. `multi_bond_occlusion`：同一无文字Atom anchor被至少两条非入射Bond中心线穿过。
3. 任一条件成立记为`ambiguous`；两者同时成立记为`severe`。
4. 单条穿线、两个有文字原子坐标接近只保留为`raw_geometry`证据，不直接判不可恢复。

这不是化学过滤，也没有键长启发式。阈值只用于审计“768图上的两个监督中心是否仍可分开”，不参与
模型推理、候选删边或benchmark事后排除。

## 全量数据结果

| 数据 | 图片 | 原始几何标记 | ambiguous+severe | severe | eligible |
|---|---:|---:|---:|---:|---:|
| route100 | 100 | 7 | 2 | 1 | 98 |
| train25k | 25,000 | 1,697 | 238 | 16 | 24,762 |

route100的两张精细困难图是：

- `train_001275_stress`：一处无文字原子被两条非入射键穿过，P2-F仍完整命中。
- `train_001276_stress`：atom17/31两个无文字中心只相距约2.85px，同时存在多键穿心；它是唯一
  `severe`，也是P2-F唯一top128失败。

## 冻结checkpoint重放

固定P2-F final checkpoint，分别取同一中心场的top128、top256和top1024 regional maxima。没有
重新训练、调阈值或选checkpoint。

| 候选预算 | 全部 | eligible | ambiguous | severe |
|---|---:|---:|---:|---:|
| top128 | 3074/3080，99/100 | 3003/3003，98/98 | 24/24，1/1 | 47/53，0/1 |
| top256 | 3076/3080，99/100 | 3003/3003，98/98 | 24/24，1/1 | 49/53，0/1 |
| top1024 | 3076/3080，99/100 | 3003/3003，98/98 | 24/24，1/1 | 49/53，0/1 |

候选从128扩大八倍后仍缺相同4个GT中心，因此最后残差不是selector容量或候选K不足。另一方面，
`eligible`与较轻`ambiguous`已经全部可达，不能据此继续把普通训练图从25k盲目扩到100k。

## 决策

- P2-F预注册门没有改变，正式状态仍为`route_causal_no_go`。
- P2-G结论为`residual_isolated_to_severe_render`，只完成残差归因，不产生模型tag。
- 后续新数据在写入训练manifest前应先运行同一审计；`severe` stress图应重新布局，而不是让模型拟合
  几乎重合的两个隐式中心。既有route100保留原样，禁止事后替换。
- 中心候选召回在可恢复route上已达到100%。下一主问题转为未知Count下从高召回峰中整组选择Atom，
  不能继续训练同一个P2-E动态硬匹配boundary selector。

## 复现

```bash
.venv-train/bin/python -m \
  chemvector.evaluation.audits.atoms.audit_g3_p2g_render_recoverability

.venv-train/bin/python -m \
  chemvector.evaluation.audits.atoms.evaluate_g3_p2g_recoverability_buckets
```

产物：

```text
outputs/audits/chemvector-g3-p2g-render-recoverability/audit.json
outputs/audits/chemvector-g3-p2g-render-recoverability/route_checkpoint_buckets.json
```

冻结checkpoint：

```text
outputs/checkpoints/chemvector-g3-p2f-stride1-center-field-train25k/step-006250.pt
SHA256 = 9e9d7725e9e9ffecde7bcbe9fbbb095f09204a480b18acbb014d88ef5ac83668
```
