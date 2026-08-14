# finite_micro_scene_v0_1

这个目录把 `formal/chemgraph` 中的有限可辨识性结论落到真实栅格输入。它不读取 SDF、
MolBlock、Render JSON、GT 对象数量或模型训练数据。

## 当前范围

- 3x3 点阵，共 9 个位置和 36 条无向边；
- 完整枚举空图及任取 1、2、3、4 条边，共 66,712 个核心场景；
- 图签名忽略 Atom ID、Bond 顺序和无向端点顺序；
- 内置 crossing/junction 与 double-bond/parallel-singles 两组 Lean fixture；
- 默认直接复用 `chemvector.core.image_io.prepare_input_image` 和 `image_tensor`；
- 按最终 CHW float32 tensor SHA256 建 observation 等价类并输出碰撞证书。

`production_tensor_digest` 的公开输入只有图片和 prepared 标志；validation 会检查函数签名，
不使用硬编码的“无 sidecar”通过项。Scene AST 只在独立渲染阶段产生图片和图真值。

fixture 是核心枚举以外的 4 个额外场景，因此全量报告的总处理数为 66,716。

## 测试

```bash
python -m unittest \
  formal.chemgraph.audit.test_finite_micro_scene_v0_1
```

## 小规模运行

```bash
python -m formal.chemgraph.audit.run_finite_micro_scene_v0_1 \
  --limit 100 \
  --output formal/chemgraph/audit/results/finite_micro_scene_v0_1_smoke
```

## 全量运行

```bash
python -m formal.chemgraph.audit.run_finite_micro_scene_v0_1 \
  --output formal/chemgraph/audit/results/finite_micro_scene_v0_1
```

输出：

- `summary.json`：规模、唯一 observation 数和碰撞统计；
- `validation.json`：固定计数、fixture 和预处理源码 SHA256；
- `collision_certificates.json`：每个同输入异图等价类的代表场景与规范图签名。
- `runtime.json`：非承诺的耗时信息，不参与其他产物的 SHA256。

独立图同构复核：

```bash
python -m formal.chemgraph.audit.verify_finite_micro_scene_v0_1 \
  --certificates formal/chemgraph/audit/results/finite_micro_scene_v0_1/collision_certificates.json \
  --expected-sha256 <collision-certificates-sha256> \
  --output formal/chemgraph/audit/results/finite_micro_scene_v0_1/independent_review.json
```

verifier 不导入 generator 的规范签名函数；它用 NetworkX 重新构图，并匹配 Node `color`
与化学 Edge `order`。`display_type` 是受 grammar 映射约束的渲染字段，不进入化学图同构。

`--image-is-prepared` 会先生成与默认路径相同的 768x768 prepared image，再走
`image_tensor`，用于验证 prepared-input 分支与默认分支等价。它不跳过或替换生产预处理。

Grammar v0.1 全量审计还会生成 `scene_manifest.jsonl`。它逐场景记录 raw/production hash、
两种规范图签名、互斥 grammar 层、三个累计域中的 collision class、未激活的 prior 和是否可作
simple-domain 单答案候选。`passed_full_audit` 只表示枚举与产物完整；是否满足唯一恢复必须另看
`validation.json` 的 `identifiability_gate_status`，训练权限始终为 `false`。

## 当前边界

v0.1 核心只使用黑色、3 px 实线和固定 3x3 几何点阵。它验证有限状态枚举、真实 production
预处理碰撞、图规范化以及两个 Lean 反例，不覆盖字体、颜色、线宽、lane gap、模糊或缩放的
stable observation shell。因此 `validation.json` 固定写入 `training_authorized=false`；即使全量
状态为 `passed_full_audit`，也不代表已经获准恢复模型训练。
