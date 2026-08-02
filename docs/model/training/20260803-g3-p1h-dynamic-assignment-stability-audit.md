# G3 P1-H 动态指派稳定性审计

> 日期：2026-08-03
>
> 状态：已完成，后期指派不稳定不是主因
>
> 性质：零训练、五checkpoint只读审计

## 唯一问题

P1-G相对P1-D在train与route均提高约5pp，但未过10pp因果门。P1-H只回答：硬动态匹配是否
在训练后期仍频繁更换同一GT的candidate代表，从而形成移动监督目标。

## 固定输入与统计

- 只读取P1-G的1,250/2,500/3,750/5,000/6,250步checkpoint；
- 图像、P1-A、vision和top-128候选每批只计算一次；
- 只读取train5k，禁止confirm、fresh、sealed、USPTO、MR-TADF；
- 每个checkpoint使用正式6px最大基数动态匹配；
- 按`sample key + GT index`比较相邻checkpoint分配的candidate ID；
- 报告相邻轮Atom turnover、发生任意turnover的整图比例、五轮始终稳定的Atom/整图比例；
- 同时重算每个checkpoint冻结后的几何Atom recall与整图exact。

## 预注册判读

主判读只看`step 5000 → 6250`：

- Atom turnover `>=10%`：后期动态指派不稳定是主要可操作瓶颈；
- `3%..10%`：次要瓶颈；
- `<3%`：否定后期硬匹配振荡是主因，下一步转target-free候选聚合可达性审计。

P1-H不训练、不保存checkpoint、不接前端、不生成SDF。

## 正式结果

五个checkpoint在同一次候选前向上完成train5k只读重放，`updates=0`，没有读取禁止数据。

| checkpoint变化 | changed Atom | Atom turnover | changed image | Image turnover |
|---|---:|---:|---:|---:|
| 1,250→2,500 | 1,336 / 154,911 | 0.862% | 868 / 5,000 | 17.36% |
| 2,500→3,750 | 922 / 154,911 | 0.595% | 566 / 5,000 | 11.32% |
| 3,750→5,000 | 856 / 154,911 | 0.553% | 626 / 5,000 | 12.52% |
| 5,000→6,250 | 666 / 154,911 | 0.430% | 571 / 5,000 | 11.42% |

五轮始终使用同一candidate代表的Atom为`152,263 / 154,911 = 98.29%`；整图所有Atom五轮
均稳定为`3,379 / 5,000 = 67.58%`。整图比例较低是因为一张图任一Atom变化都会计错，主判读
仍按逐Atom turnover。

冻结几何能力随checkpoint单调提高：

| Step | Atom recall | 几何exact |
|---:|---:|---:|
| 1,250 | 94.953% | 20.68% |
| 2,500 | 94.995% | 21.92% |
| 3,750 | 95.224% | 22.74% |
| 5,000 | 95.226% | 23.08% |
| 6,250 | 95.350% | 24.18% |

## 决策

最后两轮Atom turnover只有0.43%，低于预注册3%门，正式判定：

```text
late_assignment_instability_not_primary
```

因此不能把P1-G未过门归因于硬匹配标签在后期来回振荡，也没有依据改成软匹配来专门解决该
问题。下一步转向target-free几何候选聚合可达性：检查能否先把同一原子的重复候选折叠为一个
对象，同时不合并两个真实原子。

## 产物

```text
outputs/audits/chemvector-g3-p1h-dynamic-assignment-stability-train5k/audit.json
SHA256 8b6cac454e69b5ecda29116645c4be05795cd1ab35eb04a042a38448ce566396

审计源码
SHA256 1b243dc50e7fe6bae741d5d9251b3a753ed778ee6eb896d7f7302dcf7a77d1f2
```
