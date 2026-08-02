# 2026-08-01 ChemTraverse P2/P2-B Count/Stop 失败取证

> 状态：只读取证与下一实验预注册，不修改主训练或推理代码
> 数据边界：只使用 capacity `train_5k`、`dev_500`、P2 和 P2-B route-select 产物
> 禁止读取：fresh_dev、sealed_test、MR-TADF、USPTO、P2-B confirm 400

## 结论

`count/stop` 是 P2 当前最显著的失败表面，但现有结果还不能证明它是唯一或首要的因果瓶颈。

支持 count/stop 路线的证据很强：

- P2 的 144 个 incomplete 全部运行到 3,072 token 才停止，而这些样本的真值序列最长只有
  2,136 token，因此不是上下文容量不足，而是模型没有在正确状态结束。
- 356 个完成样本中，atom count exact 和 bond count exact 各只有 9，count-pair exact 只有 2。
- 273/356 个完成样本少预测原子，274/356 个少预测键；同时存在少量失控长尾，最多多预测
  89 个原子和 121 个键。模型同时存在提前停止和继续生成过久两种相反错误。
- 当前自由生成 count 与真值的 Pearson 相关性只有 atom `0.330`、bond `0.262`。
- 56 个四种布局全部完成的分子中，没有一个分子的四张图得到相同的预测 count pair。

但不能直接把 354 个 `count_mismatch` 解读为“只修 count 就能修好结构”：

- `failure_bucket` 按 count、topology、typed graph 的顺序短路。count 不一致时，样本不会进入
  topology 首因，因此 354 是分桶定义和模型错误的共同结果，不是因果证明。
- 仅有的 2 个 count-pair exact 样本全部 topology mismatch。
- 去掉 meta 后，完成样本的首个语义分歧主要是 bond payload 204、atom role 67、element 46、
  atom attribute 39；首个语义错误并不主要是结束控制符。
- ChemTraverse 当前要求模型生成无法从图片观察的数据库 `molecule_id`。P2 完成的 356 张中
  molecule ID exact 为 `0/356`，模型只生成了 3 个记忆化 ID。训练时 teacher forcing 始终喂入
  正确 ID，推理时后续结构却条件于错误 ID，这是发生在第一个原子之前的 exposure shift。

因此下一步不能直接训练更重的 count 模块。必须先用不训练的 2x2 oracle 实验证明：在消除
`molecule_id` 混杂并给出正确 count 时，现有 AR decoder 的 topology 是否真的能明显恢复。

## 固定产物

P2：

```text
outputs/evaluations/chemtraverse-a1-capacity5k-dev500/generated/generation_metrics.json
```

P2-B route-select 100：

```text
outputs/evaluations/chemtraverse-a1-capacity5k-controlw4-route-select100/generated/generation_metrics.json
```

训练与验证清单：

```text
Data/sdf/chemtraverse_v0_1_capacity/train_5k.jsonl
Data/sdf/chemtraverse_v0_1_capacity/dev_500.jsonl
```

本报告不使用 P2-B confirm 400，也不使用任何正式或外部 benchmark。

## 定量取证

### P2 全量 500

| 指标 | 结果 |
|---|---:|
| completed / grammar / scene | 356 / 500 |
| incomplete 且达到 3,072 token | 144 / 144 |
| SDF valid | 13 / 500 |
| atom count exact（完成样本内） | 9 / 356 |
| bond count exact（完成样本内） | 9 / 356 |
| count-pair exact | 2 / 500 |
| Skeleton exact | 0 / 500 |
| atom count MAE（完成样本内） | 11.301 |
| bond count MAE（完成样本内） | 16.604 |
| atom count 低估 / 正确 / 高估 | 273 / 9 / 74 |
| bond count 低估 / 正确 / 高估 | 274 / 9 / 73 |
| atom count 误差范围 | -21 到 +89 |
| bond count 误差范围 | -27 到 +121 |

完成与未完成样本的真值复杂度不同：

| 子集 | 真值 atom 均值 | 真值 bond 均值 | 真值 token 中位数 | 真值 token 最大值 |
|---|---:|---:|---:|---:|
| completed 356 | 26.92 | 29.32 | 723 | 1,767 |
| incomplete 144 | 39.65 | 42.95 | 972 | 2,136 |

较大分子更容易进入不终止循环，但所有真值仍在 3,072 上下文内。完成样本的预测/真值 token
长度比中位数为 `0.857`，P95 为 `3.239`，范围 `0.494--5.459`。这再次说明错误不是单向的：
多数样本过早结束，少数样本生成远超真值的结构。

### P2-B route-select 100

统一将所有 traversal control token 的 CE 乘 4 后：

| 指标 | P2 基线同 100 | P2-B |
|---|---:|---:|
| completed | 72 | 56 |
| SDF valid | 2 | 5 |
| count-pair exact | 0 | 0 |
| Skeleton exact | 0 | 0 |

逐样本配对后，completion 转移为：

| P2 | P2-B | 数量 |
|---|---|---:|
| complete | complete | 46 |
| complete | incomplete | 26 |
| incomplete | complete | 10 |
| incomplete | incomplete | 18 |

在共同完成的 46 张上，atom MAE 从 `9.783` 变成 `11.587`，bond MAE 从 `13.696` 变成
`14.087`。统一加权不仅没有解决结束，还破坏了属性和文字学习：

- `COMPONENT_END`：40.2% -> 6.6%。
- `LINK`：74.3% -> 62.7%。
- element：96.34% -> 92.59%。
- chemical bond type：96.94% -> 95.74%。
- text byte：72.82% -> 62.61%。

高频 `DOWN/UP` 控制了加权梯度，稀有结束决策没有得到独立监督。P2-B 已足以淘汰“所有控制符
统一乘一个权重”的方向。

### 跨布局不稳定

`dev_500` 是 125 个分子各四种布局。P2 的每分子完成图片数分布为：

| 四图中完成数 | 分子数 |
|---:|---:|
| 0 | 7 |
| 1 | 17 |
| 2 | 20 |
| 3 | 25 |
| 4 | 56 |

在 56 个四图全完成分子中：

- 四图预测 atom count 完全一致：0/56。
- 四图预测 bond count 完全一致：0/56。
- 四图预测 count pair 完全一致：0/56。

这不是单纯的最后一个 stop token 偶发错误。现有 AR 路径没有形成稳定的整图基数表征，独立的
全图 count 监督具有明确实验价值。

### Count 标签范围

当前 capacity 数据全部是单组分结构，可以在第一次实验中使用确定性的单组分解码门。

| 数据 | atom min/P50/P95/P99/max | bond min/P50/P95/P99/max | cycle rank max |
|---|---|---|---:|
| train_5k | 10 / 29 / 55 / 82 / 100 | 9 / 32 / 59 / 87 / 102 | 11 |
| dev_500 | 14 / 28 / 57 / 72 / 79 | 14 / 30 / 59 / 77 / 81 | 8 |

其中 cycle rank 定义为 `bond_count - atom_count + 1`。它比直接联合 100x102 个 count pair 更
紧凑，可作为第三个一致性标签，但不能替代 atom/bond 两个必报 count。

## `molecule_id` 是必须先排除的混杂

当前序列开头包含：

```text
<meta 768 768 utf8:...molecule_id...>
```

数据库 ID 不存在于分子图片中。P2 在 356 个完成样本里只生成以下三个 ID：

```text
chembl:CHEMBL1111111       247
coconut:CNP0111111.1        74
chembl:CHEMBL111111         35
```

真值 ID 命中为 0。原始 token 首分歧多数发生在位置 18--20 的 ID 字节，而不是第一个原子。
这会导致：

1. 训练 loss 被不可学习字段占用；
2. teacher forcing 的结构部分总是条件于正确 ID；
3. 自由生成的结构部分总是条件于训练未见的错误 ID 前缀；
4. token-level 首分歧指标被无关字段污染。

可部署协议应从 AR 标签删除数据库 ID，或统一输出固定的 `<id:unknown>`。真实 sample key 和
数据来源由推理调用方在 ChemScene/SDF sidecar 中写入，不应由图片模型猜测。

在修改协议和重训前，只能把“强制喂入真值 meta”作为诊断 oracle，不能作为可部署推理方案。

## P0：不训练的 2x2 因果实验

固定使用 P2 checkpoint 和同一 route-select 100，不读取其余 400。四个格子为：

| 组 | Meta 前缀 | Count 约束 |
|---|---|---|
| O0 | 当前模型自由生成 | 无 |
| O1 | 强制真值 meta，仅作 oracle | 无 |
| O2 | 当前模型自由生成 | 真值 atom/bond oracle gate |
| O3 | 强制真值 meta，仅作 oracle | 真值 atom/bond oracle gate |

O0 直接复用现有 artifact，不重复运行。O1--O3 必须保持相同 checkpoint、贪心解码、图像顺序和
最大长度，不能调 beam、阈值或样本。

### Oracle count grammar gate

当前 capacity 全部单组分。给定真值 `N` 个 atom、`M` 个 bond：

1. 已生成 atom 数达到 N 后，禁止 `DOWN` 及其后续新 `atom`。
2. 已生成 bond 数达到 M 后，禁止 `DOWN` 和 `LINK` 产生新 bond。
3. atom 未达到 N 时禁止 `COMPONENT_END`。
4. atom 达到 N 但 bond 未达到 M 时，只允许合法 `LINK` 或 `UP`；若不存在合法动作，该分支失败。
5. 只有 atom=N、bond=M、parent stack 已回到根，才允许 `COMPONENT_END`。
6. 只有 component 已关闭且必需 text/stereo postamble 完成，才允许 `MOL_END`。

该 gate 不指定原子位置、元素、键端点或键级，不能直接泄漏 topology；它只提供整图基数上界。

### P0 GO/NO-GO

- O2 或 O3 必须达到 completion >= 90/100。
- oracle-gated count-pair exact 必须 >= 80/100；未达到表示 grammar 经常进入无合法路径，单纯
  count 头不能救回。
- O2 或 O3 的 Skeleton exact 必须 >= 5/100。若仍为 0，证明 topology/暴露偏差是独立主瓶颈，
  不得把 count head 当下一主路线。
- 若 O3 相对 O2 的 completion 提高 >= 10，或 Skeleton 提高 >= 5，必须先删除/常量化 meta ID
  并从相同初始化重训；不能先训练 count head 掩盖协议错误。
- 若 O2/O3 达不到上述门槛，独立 count-conditioned AR 路线 NO-GO，转向分段 decoder 或
  count-conditioned junction/bond set 对照。

## 独立 CountHead v0.1

只有 P0 oracle GO 后才实现本节。

### 输入与结构

输入复用 `ChemVectorModel.encode_images()` 的视觉 memory `[B, L, 512]`，不读取 decoder token。
这样 count 不能靠 teacher-forced 序列作弊。

建议使用约 1M 参数以内的集合级头：

1. 对 memory 做 LayerNorm。
2. 计算固定网格的 mean pooling 和 max pooling。
3. 使用 atom、bond 两个 learned query 分别对 memory 做无 QKV 投影的 dot-product attention
   pooling，避免引入完整 Transformer 层。
4. 拼接两个 query feature、mean 和 max，经一个共享 `Linear -> GELU` bottleneck 后分出 atom、
   bond 分类 logits。
5. atom 类为 `0..128 + overflow`，bond 类为 `0..128 + overflow`。capacity 真值最大为 100/102，
   overflow 用于防止将来静默截断。
6. 可选第三分支预测 cycle rank `0..16 + overflow`，只用于一致性，不替代 atom/bond 指标。

不建议第一版只用标量 L1 回归。P2 同时有提前停止和失控长尾，均值回归容易落在不存在的中间
计数；分类分布还能为 top-k 解码提供不确定性。

### 标签

标签直接从 ChemScene 获取：

```text
y_atom  = len(scene.junctions)
y_bond  = len(scene.bond_objects)
y_cycle = y_bond - y_atom + component_count
```

标签按分子生成，四种布局共享同一组 count。数据划分继续使用既有 molecule-group 清单，不按
图片重新随机拆分。

### Loss

预注册第一版：

```text
L_count = CE(atom)
        + CE(bond)
        + 0.25 * EMD_ordinal(atom)
        + 0.25 * EMD_ordinal(bond)
        + 0.20 * CE(cycle)
        + 0.10 * JS_four_view(atom)
        + 0.10 * JS_four_view(bond)
```

`EMD_ordinal` 是预测分布与 one-hot 真值的累计分布 L1；它让差 1 比差 20 的代价小。四视图 JS
要求一个 batch 含同分子的四种布局，直接针对当前 0/56 跨布局 count 一致问题。

第一版不再改变 AR token CE，不增加 control token 权重，保证增益能够归因于集合级 count。

### 与解码结合

不要只取 count argmax 后硬锁一路。固定取联合 count posterior 的 top-4 `(N,M)`：

```text
score(sequence, N, M)
    = normalized_AR_log_probability
    + 0.5 * [log P_atom(N) + log P_bond(M)]
```

每个 `(N,M)` 使用与 P0 相同的 grammar gate 独立解码，最终从完成且可解析的候选中按固定 score
选一个。禁止用 RDKit valid 或 benchmark exact 回选候选，否则会把实验变成后处理调参。

若 top-4 中没有合法完成序列，保留最高分 partial 并明确标记 count-gate exhausted，不能偷偷
回退到无限制 AR。

## StopHead 只作为第二阶段

第一轮只实现 CountHead 和确定性 gate。若 count head 自身准确但 gate 经常无合法路径，再增加
独立 StopHead：

```text
input = [decoder decision hidden,
         image count features,
         N - generated_atoms,
         M - generated_bonds,
         at_root,
         legal_link_exists]

output = CONTINUE / CLOSE_COMPONENT / END_MOLECULE
```

StopHead 只在 grammar 的 record 决策位置训练，使用按类别归一化的 focal CE，使稀有
`CLOSE_COMPONENT` 与 `END_MOLECULE` 各自贡献固定梯度。它不负责预测 `UP/DOWN/LINK` 的全部
细节，避免再次被高频 traversal 动作主导。

StopHead 不与 CountHead 同轮首次引入。否则 route-select 有提升时无法判断来自全图 count 还是
局部 stop 分类。

## 最小训练与选择协议

### P1-Count：32 图管线门

- 使用既有 overfit_32，不改变图片或分子。
- CountHead 的 count-pair exact 必须 32/32。
- 四视图 count posterior 的平均 JS 必须 <= 0.01。
- 接到 P1-B decoder 后，gated Skeleton exact 不得低于原有 30/32；目标是修复剩余两个 count
  mismatch，而不是以破坏 topology 换 count。
- 任一项未过，先修实现，不运行 5K route-select。

### P2-Count：冻结视觉表征探针

- 从 P2 `last.pt` 加载视觉 encoder，先冻结 encoder，只训练 CountHead。
- 使用 train_5k 全部四视图；按完整分子 batch。
- 固定训练 20 epochs，AdamW `3e-4`，不按 dev 延长或改学习率。
- 只在固定 route-select 100 报一次结果，不读取 confirm 400。

冻结探针先回答“现有视觉 memory 是否已经包含可解码的全图 count”。若失败，不允许当场解冻
继续调参；需要另立一次预注册的 last-stage-unfreeze 对照。

### Route-select 100 GO/NO-GO

CountHead 自身必须同时满足：

- atom count exact >= 60/100；
- bond count exact >= 50/100；
- count-pair exact >= 40/100；
- atom MAE <= 1.5，bond MAE <= 1.5；
- 同分子四视图 count-pair 一致率 >= 60%。

端到端 top-4 gated decoding 必须同时满足：

- completion >= 85/100（P2 基线 72）；
- count-pair exact >= 20/100（基线 0）；
- Skeleton exact >= 5/100（基线 0）；
- SDF valid >= 2/100（不得低于基线）；
- 不能以更多 atom role、element 或 chemical bond type 错误换取 completion。

未同时满足时，CountHead v0.1 NO-GO，不读取 confirm 400，不追加 epoch，不改变 top-k 或 score
权重。达到门槛后才可运行一次预先保留的 confirm 400。

## 路线决策

1. 先做 O0--O3，验证 count 和 meta 的因果上界。
2. 若 oracle count 不产生 Skeleton，优先分段/分层 decoder；不要训练 count 头制造表面 valid。
3. 若 oracle count 明显恢复 Skeleton，先删除/常量化不可观察 ID，再做独立 CountHead。
4. CountHead GO 后才讨论 StopHead；两者不在第一轮同时加入。
5. 即使 CountHead GO，也只说明它解决基数与终止，不代表元素、键级、价态和 topology 已解决。

这套顺序能区分四个假设：不可观察 meta 污染、全图 count 缺失、局部 stop 控制失败，以及长 AR
的 topology exposure bias。直接增加数据或继续调统一 token 权重无法给出这种因果区分。
