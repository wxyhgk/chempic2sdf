# G3 P1-V Continuous Stride-1 Capacity64

> 日期：2026-08-04
>
> 状态：协议通过 / 表示机制GO / 双域容量NO-GO
>
> 性质：无损one-center target的连续768中心场实验

## 模型

P1-V在P1-T连续384 feature之后再进行一次feature级上采样：

```text
stroke 96x192x192
 -> Conv1x1 + PixelShuffle
 -> continuous feature 32x384x384
 -> shared depthwise 3x3
 -> Conv1x1 + PixelShuffle
 -> continuous feature 32x768x768
 -> shared depthwise 3x3
 -> unified center/offset
```

两个PixelShuffle都只生成feature，不直接生成phase logit。冻结P1-A base放大到768后与零初始化
residual相加。可训练参数`96,579`：decoder `17,379`，最后一个stride-4视觉block `79,200`。

固定双域协议与P1-U一致：64次域曝光、62张唯一图、4,800 updates、300 epochs、19,200
exposures。共享vision初始化与P1-T逐项相同；新stride-1 decoder使用固定seed初始化。

## 确定性恢复与评估修复

首次训练完整完成4,800步，但旧regional-max实现对每个plateau重新扫描整张图，在约10万峰的
768场上退化为近似`峰数 x 像素数`复杂度。评估被终止，训练权重没有写最终checkpoint。

代码随后改为一次线性扫描：`bincount`计算plateau大小，`minimum.at`取得每个plateau首个行优先
cell。随机768场约65,796个峰的提取耗时约`0.9秒`，并保持原有代表点、排序和plateau语义。

正式结果从已写入的step-4000 checkpoint恢复field、vision block和AdamW状态，重跑最后800步。
恢复后的step-4100到4800所有loss和梯度与首次运行逐位相同，证明训练轨迹未改变。原日志和恢复
日志分别保存在`train.jsonl`与`train-resumed-step4000.jsonl`。

## 结果

| 域 | known matched/exact | top128 matched/exact |
|---|---:|---:|
| overfit32 | 1399/1424，22/32 | 1420/1424，30/32 |
| adjacent32 | 1447/1473，18/32 | **1473/1473，32/32** |

7张stride-4碰撞图：

- known-count使用不同峰：`3/7`；
- top128使用不同峰：**`7/7`**。

全部15项协议检查通过，两次完整重放完全一致。正式决策是
`dual_stride1_capacity_no_go`，不进入5k。

## 结论

P1-V解决了P1-U的表示丢失：专项集top128从`1468/1473、27/32`提高到全量100%，碰撞对从
`4/7`提高到`7/7`。因此连续stride-1是明确的表示机制GO。

但每图约有98,000个regional maxima，原始分数的known-count排序较差，并使原overfit32回退。
当前瓶颈已从“能否生成两个近邻Atom”转为“如何从高召回多尺度峰集合中整组选择”。下一步P1-W
先冻结P1-U/P1-V做零训练排名审计：记录每个GT首次进入的rank、最小充分K、两尺度候选互补和
重复峰结构；只有审计证明固定top-K可覆盖两个域，才训练集合级selector。禁止继续增加分辨率、
追加P1-V epoch或重新引入固定像素聚类。

```text
decision = dual_stride1_capacity_no_go
checkpoint = outputs/smoke/chemvector-g3-p1v-continuous-stride1-capacity64/step-004800.pt
SHA256 = 790dead79f17b6857c2f873a9a7d83b98a167aea3cb5768727d3497c80a43b71
```
