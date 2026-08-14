# v1 P2-D2-E8 Stable Bond Query Set Capacity64

> 日期：2026-08-14
>
> 状态：已完成，正式 NO-GO
>
> 模型：`chemvector-v1p2d2e8-stable-bond-query-set-capacity64`

## 因果依据

E7 的单图一步更新在 64 图间产生 assignment 震荡。随后固定同一张 21-Bond 图做隔离：原
`logsumexp` 组概率在 loss 约 0.14 时仍只能恢复 20/21，因为组内多个低概率候选之和可以超过
错误 argmax。将组目标改为 `max log p` 后，同一架构在 step 700 达到 known exact、0.5 count、
predicted exact 全部 1/1。由此确认架构可表达，E8 只验证修正 loss 后的跨图稳定训练。

## 固定协议

- 数据和冻结视觉组件与 E7 完全一致；
- 组内 pointer 目标固定为 latent `max log p`，与推理 argmax 对齐；
- 每 8 张变量图累积梯度后执行一次 optimizer update；
- AdamW 学习率 `1e-4`，最多 2,400 updates，即 19,200 microsteps；
- 每 100 updates 在同一 capacity64 全量评估；
- existence threshold 固定 0.5，不扫描阈值、不做像素 NMS。

## GO/NO-GO

门仍为 known-count exact 64/64、count 64/64、predicted exact 64/64，且四种布局分别 16/16。
通过只证明 64 图容量，下一步才允许预注册 style fit 训练；任一失败则继续停在容量层。

## 执行命令

```bash
.venv-train/bin/python -m \
  chemvector.training.bonds.objects.train_v1p2d2e8_stable_bond_query_set_capacity64
```

## 实际结果

967,305 参数完成 2,400 optimizer updates、19,200 microsteps，耗时 160.56 秒。best 为
update 2200：

- known-count precision/recall：98.31%/49.85%，exact 0/64；
- 0.5 threshold count accuracy：0/64，MAE 24.94；
- predicted precision/recall/F1：98.67%/12.56%/22.29%，exact 0/64。

正式判定 `stable_bond_query_set_capacity_no_go`。相对 E7，loss 不再出现百级尖峰，证明梯度
累积确实修复了数值不稳；但 learned query 在不同图上的 Hungarian 身份仍不稳定，存在性头
长期偏向 no-object。单图 1/1 与跨图 0/64 的对照说明，瓶颈是全局 query 身份没有绑定候选
内容，而不是模型参数不足或训练图片不够。

下一周期停止全局 learned Bond query。改为 candidate-conditioned same-Bond grouping：直接监督
候选 edge 间是否属于同一 GT Bond，先形成内容相关的等价组，再对每组预测存在性。该路线不
依赖像素 NMS，也不会要求一个全局 query 在不同分子中维持任意对象编号。

## 产物

- checkpoint：
  `outputs/checkpoints/chemvector-v1p2d2e8-stable-bond-query-set-capacity64/final.pt`；
- checkpoint SHA256：
  `218c7a1330487206fc226b8ebe37ae3860002fa16eafbe8facfadd779858a4ab`；
- result SHA256：
  `b3fb030f75b6a6c47e12081709742220c557c9373e11aa1362782bf4cd21ce63`；
- 模型 tag：`chemvector-v1p2d2e8-stable-bond-query-set-capacity64`。

所有保留集读取数为 0，生成 PNG 数为 0。
