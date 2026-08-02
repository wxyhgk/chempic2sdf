# 2026-08-01 ChemTraverse GT Count Oracle

> 状态：三项门槛全部失败，CountHead-conditioned AR 路线 NO-GO
> 性质：只读因果诊断，不是可部署模型，不创建 checkpoint 或 Git tag
> 数据边界：固定 capacity route-select 100；未读取 confirm400、fresh dev、sealed test、MR-TADF 或 USPTO

## 要回答的问题

ChemTraverse P2 自由生成时经常少生成或无限生成。这里不训练新参数，直接把每张真值图的原子数
`N` 和键数 `M` 交给语法约束器，检验“只要知道数量，当前 AR decoder 能否恢复正确拓扑”。

Oracle 只约束：

- 达到 `N` 后不能新建原子；
- 达到 `M` 后不能新建键；
- 未达到 `N/M` 时不能结束；
- 仍由模型决定原子属性、树父子关系、回边端点和键类型。

因此它泄漏整图基数，但不泄漏具体 topology。输出 summary 和每条 record 均显式写入
`"oracle_counts": true`，不得与正常推理成绩合并。

## 固定输入

- checkpoint：`outputs/smoke/chemtraverse-a1-capacity5k-observable-meta/last.pt`
- checkpoint SHA256：`597b5614f2fbf0a51de1a743c2a0ae41f432c2b6836685c06b7b8d4acb129424`
- shard：`Data/sdf/chemvector_joint_v1/style_train_100k/shards/train-000001.tar`
- manifest：`Data/sdf/chemtraverse_v0_1_capacity/dev_route_select_100.jsonl`
- molecule ID 目标：常量 `sdf`
- 解码：greedy，beam width 1

运行命令：

```bash
.venv-train/bin/python -u -m chemvector.evaluate \
  --checkpoint outputs/smoke/chemtraverse-a1-capacity5k-observable-meta/last.pt \
  --shards Data/sdf/chemvector_joint_v1/style_train_100k/shards/train-000001.tar \
  --sample-manifest Data/sdf/chemtraverse_v0_1_capacity/dev_route_select_100.jsonl \
  --target-molecule-id sdf \
  --oracle-counts \
  --samples 100 \
  --generate-samples 100 \
  --batch-size 4 \
  --workers 0 \
  --beam-width 1 \
  --output-dir \
    outputs/evaluations/chemtraverse-a1-capacity5k-observable-meta-gtcount-route100
```

## 预注册门槛

只有同时满足以下三项，才允许训练独立 CountHead：

| 指标 | 门槛 |
|---|---:|
| completed | >= 90/100 |
| count-pair exact | >= 80/100 |
| Skeleton exact | >= 5/100 |

## 结果

| 指标 | 无 oracle 的常量 meta | GT count oracle | 门槛 |
|---|---:|---:|---:|
| completed / grammar / scene | 67 | 68 | 90 |
| count-pair exact | 1 | 68 | 80 |
| Skeleton exact | 0 | 0 | 5 |
| Mol / SDF valid | 1 | 2 | - |
| chemical exact | 0 | 0 | - |

Oracle 的 68 个完成样本全部进入 `topology_mismatch`，其余 32 个为
`generation_incomplete`。评测耗时 142.69 秒。完整产物：

```text
outputs/evaluations/chemtraverse-a1-capacity5k-observable-meta-gtcount-route100/
  generated/generation_metrics.json
```

## 结论

GT count 能机械保证已完成序列的数量正确，却几乎没有提高 completion，而且没有救回任何正确
骨架。当前失败不是独立 CountHead 能解决的停止问题；模型仍未学会稳定的父子选择、回边引用和
整图拓扑决策。

决策如下：

1. 不实现、不训练 CountHead-conditioned AR。
2. 保留 count oracle 作为诊断入口，默认关闭。
3. 下一组实验转向 decision-only grammar loss，排除大量 forced token 对结构决策梯度的稀释。
4. 同时实现 G3：count-conditioned Atom slots 到 Bond group queries 的短集合解码，避免把
   ChemTraverse 长序列当成唯一模型输出；ChemTraverse 仅作为确定性编译协议保留。

## 验证

生成约束、评测接入和 decision loss 相邻聚焦测试：

```text
23 passed in 1.85s
```
