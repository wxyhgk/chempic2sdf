# G3 P4-E7 Offset Radius A/B

> 日期：2026-08-08
>
> 状态：已完成，offset头非主因
>
> 审计：`chemvector-g3-p4e7-offset-radius-ab`

## 目标

P4-E6中15个6px无峰漏点距最近完整峰只有6.12–10.57px。P4-E7固定P4-E3的峰身份和排序，
比较预测offset坐标与同一source-id的stride-1 cell center，并报告6/8/10/12px曲线，判断offset头
是否把本来正确的峰推过了6px门。

## 固定协议

- 只读已暴露dev640；候选身份、score和顺序逐项固定；
- 预算固定为top128和top1024；
- anchor模式为预测offset和cell center `(x+0.5,y+0.5)`；
- 半径固定为6、8、10、12px，全部使用最大基数优先Hungarian；
- P4-E3预测offset/top1024/6px必须重放32,342 matched、597 exact；
- cell center在6px至少增加15个matched、aggregate exact不退且paired exact零break，才判定
  `offset_head_harmful_go`；否则为`offset_head_not_primary`；
- 10px曲线只做诊断，不修改正式6px指标；
- 不训练、不保存cache、不读pilot/confirm_v2或外部/封存集。

本轮零训练，不创建tag。

## 实际结果

预测offset/top1024/6px严格重放32,342 matched、597 exact。只替换为同source-id cell center后，
matched仍为32,342，exact降为596；paired为0 rescue / 1 break。top128仅多1个matched且少1张
exact。cell center的+15 matched、exact不退和paired零break三道门全部失败，正式决策
`offset_head_not_primary`。

预测offset的半径曲线为：

| 预算 | 半径 | matched / 32,400 | recall | exact / 640 |
|---:|---:|---:|---:|---:|
| 128 | 6px | 32,195 | 99.3673% | 550 |
| 128 | 10px | 32,252 | 99.5432% | 573 |
| 1024 | 6px | 32,342 | 99.8210% | 597 |
| 1024 | 8px | 32,366 | 99.8951% | 615 |
| 1024 | 10px | **32,377** | **99.9290%** | **621** |
| 1024 | 12px | 32,383 | 99.9475% | 625 |

10px相对6px净恢复35个Atom和24张exact，paired exact为24 rescue / 0 break。这是oracle匹配的
定位诊断，不允许把正式6px指标改成10px：更宽半径会隐藏局部多对象和错误拓扑。它证明多数
剩余视觉峰在正确区域附近，但当前candidate anchor没有达到6px精度；同时top128即使10px仍只有
99.5432%，说明排名和局部多对象问题仍独立存在。

cell center与预测offset几乎相同，说明简单删除offset头或改固定0.5不能修复。下一步先对P4-E6
的28个matching conflict构建6px候选/GT连通分量，确定局部multiplicity需要2、3还是更多对象；
之后再以键端点/局部patch做候选anchor细化，而不是继续训练当前cell offset。

## 产物

- audit SHA256：
  `1334135d7b32cc312c39947130e50b103c4331d17c2ab337aaaeff9afffb7e75`；
- 运行耗时33.94秒；
- pilot、confirm_v2、fresh和外部/封存集均未读取。
