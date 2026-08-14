# 有限微场景审计 v0.1

这份协议定义 Lean 有限反例之后的真实栅格验证，不使用现有 SDF 数据训练模型。

## 状态空间

核心画布使用 3×3 归一化有理点阵，共 9 个位置、36 条无向线段。完整枚举空图及任取 1–4 条不同边且最大度不超过 4 的图，未过滤前为：

```text
1 + C(36,1) + C(36,2) + C(36,3) + C(36,4) = 66,712
```

另设两个有限核：

- Multiplicity：`[2]` 对 `[1][1]`，以及 `[3]` 对 `[2][1]`、`[1][1][1]`。
- Text：`I/N/H/F/Cl/Br/Ph/R1/+/-`、三种字号和四方向连接子集。

几何核心完整穷举；旋转、反射、线宽、lane gap、颜色、模糊和缩放作为 observation shell 分层采样。

## 唯一合法观测

碰撞比较必须发生在模型实际读取的最后一个确定性预处理步骤之后：

```text
PNG bytes
  -> production resize / color conversion / normalization / quantization
  -> contiguous model tensor
  -> SHA256(shape || dtype || little-endian tensor bytes)
```

PNG 文件名、压缩参数和 metadata 不属于观测。必须另外检查 raw RGBA、RGB u8、灰度量化、768/384/192 resize 和 blur 0/0.5/1.0 的碰撞阶梯。

## 图签名

原始 Atom ID 和 Bond 列表顺序不属于图语义。对最多 9 个节点的微场景，直接枚举所有保持节点颜色的置换，取“节点颜色 + 上三角边颜色”字典序最小编码。边颜色包含 order 和 display type；节点颜色包含 `IMPLICIT` 或 `TEXT(token)`。

同时生成擦除文字 token 的 structure signature，用于分别定位骨架和 OCR 错误。

## 分类

- `graph_strict`：同一 production observation 只有一个 full graph signature。
- `field_strict`：同时只有一个 canonical Oracle field，且每根 Bond 有 carrier。
- `sampled_shell_collision_free`：在已枚举的预处理扰动阶梯中没有发现异图碰撞。这个名称只陈述有限实验事实，不等价于数学上的逆映射稳定性。
- `prior_resolved(prior_id)`：只在预注册先验下唯一，不能冒充无条件 strict。
- `ambiguity`：剩余同输入异图，包括 crossing/junction 和 double/parallel-single。

单答案训练候选至少必须同时满足 `graph_strict`、`field_strict` 和 `sampled_shell_collision_free`；在观测度量、异图最小间隔和拒识半径尚未定义前，不能声称满足数学稳定性。

## 冻结验收

1. 同一 Scene 重复三次、单进程和多进程的输入 digest 完全一致。
2. Atom ID、Bond 顺序、lane 顺序、端口交换后 graph/field signature 不变。
3. 两组 Lean fixture 在真实栅格枚举器中产生对应的碰撞证书。
4. 未登记 exact collision 自动失败，禁止人工删除。
5. Oracle field 到 production decoder 在上述有限碰撞自由样本上 graph exact 为 100%。
6. crossing 假 junction、chain shortcut、fan chord、text-only 假 Bond 全为 0。
7. 推理目录不含 SDF 和 sidecar，并审计 sidecar 读取数为 0。
8. 独立 reviewer 用未知 seed 连续两轮复现通过；代码、阈值或权重变化后重新计数。

Lean 中的 `FiniteAudit.lean` 自动检查 8 个最小抽象场景。CPU 枚举器现已完整运行两轮
66,712 个核心图，并得到逐字节相同的 summary 和碰撞证书；结果见
[单一样式 production tensor 全量结果](20260814-finite-production-audit-v0_1.md)。

该结果完成的是固定黑色 3px 样式的 exact collision 门，不包含本协议上方列出的
Multiplicity/Text 全量核和 observation shell，也没有完成 Oracle decoder，因此训练仍未授权。
