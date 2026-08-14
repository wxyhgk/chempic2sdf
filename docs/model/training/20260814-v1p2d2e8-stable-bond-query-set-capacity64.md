# v1 P2-D2-E8 Stable Bond Query Set Capacity64

> 日期：2026-08-14
>
> 状态：已预注册，等待训练
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
