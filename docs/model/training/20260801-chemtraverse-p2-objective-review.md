# 2026-08-01 ChemTraverse P2 训练目标独立审查

> 状态：候选 A 已完成 route-select100 并判定 NO-GO；未读取 confirm400，正式 50K 继续禁止启动
> 边界：只使用已消费的 P2 `dev_500` 与 `route-select100`；未读取
> `route-confirm400`、fresh、sealed、MR-TADF 或 USPTO
> 实现约束：训练入口仅增加默认关闭的可选 objective；`evaluate.py` 和核心模型代码不变

## 结论

P2 失败不能简单归结为“数据只有 5K”。当前目标同时存在三项可证实的问题：

1. 等权 token CE 把大量语法强制位置、不可观察的 `molecule_id` 和真正的图遍历决策混在
   同一个平均数里；
2. `control weight=4` 只放大整个控制组，没有改变控制组内部近 30:1 的频率差，因此新增
   梯度主要流向 `DOWN/UP`，并未流向稀有的结束和闭环决策；
3. teacher forcing 只测量“真实前缀给定后”的局部分类，不能训练或度量自由运行时的前缀
   漂移。P2 的平均目标长度约 834 token，一次早期错误会改变后续 DFS 栈、引用和所有字段
   的条件分布。

下一步最多运行两个低成本候选：

- A：grammar-aware decision-balanced CE；
- B：固定 5% 的 clean-label token context corruption。

两者不得在第一轮组合，也不得追加第三个事后调参版本。全局 focal loss、标准 scheduled
sampling 和 Professor Forcing 本轮不进入候选。

## 审计证据

### P2 dev500

来源：

- `outputs/evaluations/chemtraverse-a1-capacity5k-dev500-token-audit/teacher_forcing_metrics.json`
- `outputs/evaluations/chemtraverse-a1-capacity5k-controlw4-dev500-token-audit/teacher_forcing_metrics.json`

基线共有 416,980 个有效 label，平均每图 833.96 个。关键类别为：

| 类别 | 数量 | 占全部 label | P2 准确率 | control×4 准确率 |
|---|---:|---:|---:|---:|
| grammar marker | 85,828 | 20.58% | 99.55% | 99.39% |
| numeric | 153,644 | 36.85% | 61.42% | 61.06% |
| text byte | 15,096 | 3.62% | 72.82% | 62.61% |
| traversal control | 33,416 | 8.01% | 86.86% | 86.79% |
| element | 15,292 | 3.67% | 96.34% | 92.59% |
| chemical bond type | 16,624 | 3.99% | 96.94% | 95.74% |

控制组内部不是均匀分布：

| token | 数量 | 占控制组 | P2 准确率 | control×4 准确率 | 变化 |
|---|---:|---:|---:|---:|---:|
| `COMPONENT_END` | 500 | 1.50% | 40.20% | 6.60% | -33.60pp |
| `DOWN` | 14,792 | 44.27% | 81.28% | 84.01% | +2.73pp |
| `UP` | 14,792 | 44.27% | 94.48% | 94.27% | -0.20pp |
| `LINK` | 1,832 | 5.48% | 74.34% | 62.72% | -11.63pp |
| `CHEMTRAVERSE_END` | 500 | 1.50% | 92.60% | 89.40% | -3.20pp |

`DOWN + UP` 占控制标签的 88.54%。统一乘 4 后，这个比例完全不变。按 dev500 数量估算，
加权归一前 `COMPONENT_END` 的有效质量只有 2,000，而 `DOWN + UP` 为 118,336，仍相差
59 倍。因此 P2-B 不是“稀有结束符加强实验”，而是“高频上下栈 token 加强实验”。

### 同一 route-select100 的严格复核

route-select100 含 83,636 个 label。P2 与 P2-B 使用相同样本、相同 50M 模型、相同视觉
初始化、相同随机种子、5K 训练数据、3 epochs、3,750 steps 和 `3e-4` 学习率。

| 指标 | P2 | control×4 |
|---|---:|---:|
| raw teacher token accuracy | 82.05% | 80.91% |
| `COMPONENT_END` | 39/100 | 5/100 |
| `DOWN` | 2,480/2,980 | 2,521/2,980 |
| `LINK` | 278/368 | 243/368 |
| `UP` | 2,810/2,980 | 2,835/2,980 |
| element | 96.04% | 92.47% |
| chemical bond type | 97.40% | 96.27% |
| display bond type | 88.38% | 83.06% |
| text byte | 73.21% | 63.43% |

未加 grammar mask 时，P2 的 61 个 `COMPONENT_END` 错误全部被预测成 `UP`；P2-B 则有
95 个被预测成 `UP`。这说明两个标签共享相似的局部前缀，而高频 `UP` 的先验压过了结束符。

但是 `UP` 在根节点父栈为空时属于非法 continuation。以 gold prefix 应用当前 ChemTraverse
grammar mask 后：

| grammar-masked 指标 | P2 | control×4 |
|---|---:|---:|
| `COMPONENT_END` | 86/100 | 38/100 |
| `LINK` | 278/368 | 243/368 |
| `CHEMTRAVERSE_END` | 100/100 | 100/100 |
| 全部非强制位置的 full-vocab accuracy | 53,166/67,996 | 52,234/67,996 |

所以 grammar 解码确实能消除部分错误，但 P2-B 在真正合法的候选中仍更偏向继续 `DOWN`：
其 `COMPONENT_END` 的 masked 错误为 62 次 `DOWN`。这与端到端结果一致：新版 grammar
greedy 在 route-select100 上复核 P2，仍只有 completed/grammar/scene `72/100`、SDF valid
`2/100`，count-pair、Skeleton 和 typed graph 全为 0。

## 不可观察的 molecule_id

ChemTraverse 当前把 `molecule_id` 放在开头的 `<meta>` UTF-8 字段中。它是数据集主键，
没有渲染到输入图片，因此对未见分子没有可学习的视觉因果信号。

对 route-select100 按 grammar 状态重新计数：

- meta `molecule_id` byte：1,996，占全部 label 的 2.39%；
- 真正 TextObject byte：960；
- 即 meta ID 占所有 byte 标签的 67.52%；
- P2 对 meta byte 的 teacher accuracy 为 1,374/1,996 = 68.84%；
- P2 对真正文字 byte 的 teacher accuracy为 790/960 = 82.29%。

前者的 68.84% 主要来自共同前缀，不代表模型能推断 ID。现有 generation 记录经常在 token
19/20，即 meta ID 的变化部分首次分歧。这个不可避免的错误发生在第一个 component 之前，
随后作为错误前缀输入整个图遍历 decoder，构成明确的 exposure-bias 混杂。

因此两个候选开始前必须共同执行同一项协议清洗：

1. 训练目标中的 `molecule_id` 固定写为非空 sentinel `sdf`；
2. 真实 ID 只保留在 WebDataset key/meta sidecar，由 evaluator 在模型外关联；
3. loss、sequence exact 和 checkpoint 选择均不再考查真实 ID；
4. A、B 和重新运行的等权 CE baseline 必须使用同一清洗目标。

这是错误监督的移除，不算第三个模型候选。长期可以由推理器直接注入整个已知 meta prefix；
本轮固定 `sdf` 与已经冻结的 observable-meta B0 完全一致，且仍满足 ChemTraverse v0.1 的
非空 UTF-8 约束。不得改成不同长度的 sentinel，否则不再是单变量 A/B。

## 候选 A：Grammar-Aware Decision-Balanced CE

### 目标

让训练目标与 grammar-constrained inference 对齐，并在合法决策中纠正 token 频率，而不是
放大所有控制符。

对 gold prefix `y_<i` 运行同一个 `ChemTraverseGrammarV01`，得到合法集合 `A_i`：

```text
L_i = 0                                           if |A_i| = 1
L_i = alpha(y_i) * CE(logits_i restricted to A_i, y_i) otherwise
L   = sum(L_i) / sum(alpha(y_i))
```

具体预注册如下：

- `|A_i|=1` 的位置 loss 权重为 0；route-select100 中有 15,640/83,636 = 18.70%；
- 非控制且非强制位置 `alpha=1`；
- 控制 token 频率只能从 `train_5k` target 统计，不能使用 route-select 调权；
- 控制权重公式固定为
  `alpha(t)=clip(sqrt(f_max/f_t), 1, 4)`，随后按 batch 有效权重均值归一到 1；
- 按当前 dev 分布估算，`DOWN/UP≈1`、`LINK≈2.84`、
  `COMPONENT_END/CHEMTRAVERSE_END=4`；实际值以 train-only 计数一次性冻结；
- 不加入 focal、不做 token corruption、不改变学习率、步数或 checkpoint 终点。

这里必须对 logits 做合法集合内的 masked softmax。仅把强制位置权重设为 0、但仍让合法决策
与 2,525 个非法 token 竞争，会继续优化推理时根本不会发生的错误。

### 为什么不先加 focal

Focal loss 解决的是大量易样本压过困难样本，但它本身不修正 label 频率，也不知道哪些错误
会被 grammar mask 消除。全局 focal 会优先放大 numeric、byte 等高损失位置；在移除 meta ID
之前还会放大不可观察噪声。候选 A 已直接去掉强制位置并平衡合法决策，再叠 `gamma` 会增加
无法归因的超参数。因此第一轮明确不用 focal。

### 训练入口与独立 loss 模块

loss 位于 `chemvector/training/chemtraverse_decision_loss.py`，并已通过显式参数接入共享训练
入口，默认关闭时仍使用原等权 CE。启用后先用
`build_chemtraverse_decision_plan(input_ids, tokenizer)` 从 GT prefix 构造 sparse plan；训练集
所有 plan 的 `decision_token_frequencies` 合并后，一次性用
`balanced_control_token_weights` 冻结权重。forward 后调用
`grammar_aware_decision_balanced_loss(output["logits"], input_ids, tokenizer,
token_weights=weights, plan=plan)`，再将返回的 `loss` 与 `loss_per_sample` 写回 output。
权重不得按 batch 或 route-select 动态重算。

启用参数为 `--chemtraverse-decision-loss` 和
`--chemtraverse-decision-frequencies PATH`。训练、contrastive mismatch 和 validation 共用同一
objective；checkpoint 固化完整 objective，resume 时要求完全一致。频率产物中的
`target_molecule_id` 也必须与命令行 sentinel 一致。

Train-only 频率由只读 builder 生成，不手写 JSON：

```bash
.venv-train/bin/python Data/sdf/build_chemtraverse_decision_frequencies.py \
  --shards \
    Data/sdf/chemvector_joint_v1/style_train_100k/shards/train-000000.tar \
    Data/sdf/chemvector_joint_v1/style_train_100k/shards/train-000001.tar \
  --sample-manifest Data/sdf/chemtraverse_v0_1_capacity/train_5k.jsonl \
  --target-molecule-id sdf \
  --output Data/sdf/chemtraverse_v0_1_capacity/train_5k_decision_frequencies_sdf.json
```

本次实际审计 5,000/5,000 样本，得到 4,140,981 个有效位置，其中 decision
3,347,001、forced 793,980。冻结控制频率为 `DOWN/UP/LINK/COMPONENT_END/MOL_END =
150216/150216/17932/5000/8`，对应权重 `1/1/2.894301/4/4`。artifact 绑定的 train
manifest SHA256 为 `0022245e1bbc98a638a94b03426b3cc12c09781ed355baafbd748ad918d66e21`。

候选 A 的训练入口为：

```bash
.venv-train/bin/python -m chemvector.train \
  --train-shards \
    Data/sdf/chemvector_joint_v1/style_train_100k/shards/train-000000.tar \
    Data/sdf/chemvector_joint_v1/style_train_100k/shards/train-000001.tar \
  --validation-shards \
    Data/sdf/chemvector_joint_v1/style_train_100k/shards/train-000001.tar \
  --output-dir outputs/smoke/chemtraverse-a1-capacity5k-decision-balanced \
  --sequence-version traverse-0.1 \
  --sample-manifest Data/sdf/chemtraverse_v0_1_capacity/train_5k.jsonl \
  --validation-sample-manifest Data/sdf/chemtraverse_v0_1_capacity/dev_500.jsonl \
  --target-molecule-id sdf \
  --data-commitment Data/sdf/chemvector_joint_v1/style_train_100k/commitment.json \
  --epochs 3 --train-samples-per-epoch 5000 \
  --batch-size 4 --gradient-accumulation 1 --image-size 768 \
  --learning-rate 3e-4 --weight-decay 0.01 --workers 0 \
  --validation-batches 125 --save-every 1250 --log-every 100 \
  --seed 20260713 --no-shuffle --deterministic \
  --init-vision-checkpoint outputs/checkpoints/chemvector-v4-clean-50k/last.pt \
  --chemtraverse-decision-loss \
  --chemtraverse-decision-frequencies \
    Data/sdf/chemtraverse_v0_1_capacity/train_5k_decision_frequencies_sdf.json
```

该命令相对 observable-meta B0 只改变 sequence loss；sentinel、初始化、数据、顺序、曝光、
优化器参数和最终 checkpoint 均不变。

## 候选 B：5% Clean-Label Context Corruption

### 目标

验证主要残差是否来自 teacher forcing 的前缀分布偏移，而不改变 label loss 权重。

预注册配置：

- 使用与清洗 baseline 相同的等权 CE；
- `token_corruption_probability=0.05`，固定三轮，不退火、不扫描概率；
- 只污染 decoder 输入上下文，label 始终保持干净；
- 使用现有 `[UNK]` replacement 实现，不增加第二次 forward；
- 不叠候选 A、不加 focal、不改变其余训练参数。

平均 834 token 的序列约有 42 个上下文位置被替换。该实验的诊断信号是：clean teacher
指标基本不变，但自由生成的完成、count 和 Skeleton 提升。如果 teacher 与 generation 同时
退化，则固定 5% 对当前数据量过强，直接判 NO-GO，不在同一 route-select 上继续扫描 1%、
2% 或 10%。

### 为什么不选 scheduled sampling / Professor Forcing

- 标准 scheduled sampling 需要按 token 串行 rollout；平均 834、最大 3,072 token 会破坏
  Transformer 当前的并行 teacher forward，计算成本和实现风险都明显高于本轮 pilot。
- Professor Forcing 还需要 free-running trajectory、隐藏状态判别器和对抗优化；在离散 grammar
  mask 下归因更困难，且至少多一次昂贵 rollout，不属于低成本 route-select 实验。
- token corruption 虽不完全模拟模型的合法错误，但只需一次并行 forward，足以先判断
  exposure bias 是否值得继续投入。

## Route-Select100 预注册门槛

先以固定 sentinel 目标重跑一次等权 CE baseline `B0`。三组必须使用：

- 相同 vision checkpoint SHA；
- decoder 独立但相同 seed 初始化；
- 相同 `train_5k`、顺序、batch 4、3 epochs、3,750 steps、`3e-4`；
- 相同最终 checkpoint，不按 route-select 选 step；
- 相同 grammar greedy、beam 1、max length 3,072；
- 同一个已消费的 route-select100；confirm400 保持不读。

候选 A 或 B 必须同时满足以下全部条件，任何一项失败即 NO-GO：

| 层级 | 硬门槛 |
|---|---:|
| completed / grammar valid / scene valid | 均 >= 85/100，且不得低于清洗 B0 超过 2 |
| SDF valid | >= 10/100，且 >= B0 + 5 |
| count-pair exact | >= 5/100，且 >= B0 + 3 |
| Skeleton exact | >= 3/100，且 >= B0 + 2 |
| typed graph exact | >= 1/100，且 >= B0 + 1 |
| grammar-masked `COMPONENT_END` | >= 85/100 |
| grammar-masked `LINK` | >= 75% |
| 非强制位置 teacher accuracy | >= 77.5% |
| element / chemical bond type | 分别 >= 95% / 96% |
| 真正 TextObject byte | >= 80% |

raw overall token accuracy、真实 `molecule_id` accuracy 和 sequence exact 只报告，不用于选
winner。它们分别被强制位置、不可观察标签和长序列合取效应污染。

统一 evaluator 会在 `teacher_forcing_metrics.json` 的
`chemtraverse_grammar_masked` 下写入与门槛直接对应的统计：

- 非强制位置：`non_forced_decision_{correct,total,accuracy}`；
- `COMPONENT_END`：`control_tokens.[COMPONENT_END]`；
- `LINK`：`control_tokens.[LINK]`；
- 真实文字 byte：`byte_contexts.text_object`；
- 不可观察的 meta ID byte 单列为 `byte_contexts.meta_molecule_id`，不得与真实文字合并。

这些指标按 GT prefix 重放与生成器相同的 grammar，并只在合法 token 集合内取 argmax；raw
全词表 `token_accuracy` 和原有 `token_categories` 仍原样保留，便于同时审计 mask 前后差异。

若两个候选都过门，按以下固定顺序选择且不看加权总分：Skeleton exact、count-pair exact、
SDF valid、completed、非强制位置 teacher accuracy。若都不过门，停止，不组合 A+B、不追加
epoch、不扩大数据，也不读取 confirm400。

## 候选 A 最终结果

### 产物身份与配对

候选 A 使用最终 `last.pt` 做 route-select100，未使用同一步数的 periodic checkpoint：

| 产物 | SHA256 |
|---|---|
| 候选 A `last.pt`（实际评测） | `b4aa62f8519c4693a8f4552d5c70fbc6bcbff138cb4822842134693ce0b3c313` |
| 候选 A `checkpoint-0003750.pt`（未用于评测） | `9eb4eeb689cbf6ac3a490e480b13ade015e913cf540d721864c14064c4d33d28` |
| 清洗 B0 `last.pt` | `597b5614f2fbf0a51de1a743c2a0ae41f432c2b6836685c06b7b8d4acb129424` |
| 清洗 B0 teacher metrics | `f6ff1d21610220c4478057895e69670599de864edda8ccf724ded0f645d43b5f` |
| 清洗 B0 generation metrics | `2ac7ac3d31246de775a4a245df3c90fce28b915b4ed24daa6426d709a1a2399c` |
| 候选 A teacher metrics | `ff70ef0bbe0c5fcd7e57f872c774799acbcb5f0a3bfc17be2412edb6edef015b` |
| 候选 A generation metrics | `fde9e4feea8547173344a64dde53b6fe7d463400341a07df2900f85e818206bf` |

两组 `config.json` 完全相同，SHA256 均为
`b9c02d199e0b5de59179dae710239ffbfa9889665507afc97fc92a3b16dc04ff`。训练参数逐字段比较后，
除 `output_dir` 外只有候选 A 新增的两个预注册 objective 参数不同。两组还绑定相同 tokenizer、
train/dev manifest、视觉初始化、数据 commitment 和 seed；route100 的 100 个 sample key 及顺序
完全相同，均为 beam 1、非 oracle count。

残余审计限制是：两组记录的 Git revision 同为
`8df009b71c6548908db0600c1b977759e8c1751a`，但都来自 dirty worktree，且 tracked-status SHA256
不同。因此数据、模型配置、初始化和命令变量满足配对，源码工作树不是可声称 bitwise identical
的干净快照。该限制不改变下面多个硬门槛失败所决定的 NO-GO 结论。

### 训练 validation 轨迹

三轮均完整消费 5,000 个训练样本；validation 固定为 125 batches。这里的 loss 是候选 A 的
grammar-masked decision-balanced objective，不能与 B0 的全 token CE 数值直接比较；同表的
token accuracy 仍是 raw full-vocabulary argmax。

| epoch | global step | validation loss | raw token accuracy | decision / forced | decision fraction |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,250 | 1.322372 | 21.0037% | 330,414 / 78,142 | 80.8736% |
| 1 | 2,500 | 1.224550 | 24.5359% | 330,414 / 78,142 | 80.8736% |
| 2 | 3,750 | 1.077434 | 26.0953% | 330,414 / 78,142 | 80.8736% |

objective loss 持续下降，但下降没有转化为 graph exact；不能用训练曲线继续追加 epoch 绕过固定
route-select 门槛。

### Route100 teacher forcing

两组均为相同 100 个样本和 81,940 个有效 token。B0 的旧评测早于统一 grammar-masked evaluator，
所以它没有合法集合内的分桶；不得拿 B0 raw accuracy 冒充 masked baseline。

| 指标 | 清洗 B0 | 候选 A |
|---|---:|---:|
| raw full-vocab CE | 0.740045 | 5.574096 |
| raw token accuracy | 82.5360% | 26.4303% |
| exact sequence | 0/100 | 0/100 |
| samples/s | 40.763 | 18.554 |
| grammar-masked overall | 未记录 | 80.2795% |
| non-forced decision | 未记录 | 50,141/66,300 = 75.6275% |
| forced positions | 未记录 | 15,640 |
| meta molecule ID byte | 未记录 | 300/300 = 100.00% |
| 真实 TextObject byte | 未记录 | 469/960 = 48.8542% |

raw 类别完整配对如下。候选 A 的 objective 不再训练非法 token 竞争和 forced positions，因此
raw full-vocab 指标大幅下降；这些值仍按预注册原样报告，不能在看到结果后改成另一套门槛。

| raw token 类别 | 清洗 B0 | 候选 A |
|---|---:|---:|
| atom chirality | 3,049/3,080 = 98.99% | 183/3,080 = 5.94% |
| atom role | 2,985/3,080 = 96.92% | 0/3,080 = 0.00% |
| bond direction | 3,247/3,348 = 96.98% | 0/3,348 = 0.00% |
| bond orientation | 2,499/3,348 = 74.64% | 0/3,348 = 0.00% |
| bond stereo | 3,325/3,348 = 99.31% | 685/3,348 = 20.46% |
| chemical bond type | 3,263/3,348 = 97.46% | 285/3,348 = 8.51% |
| display bond type | 2,961/3,348 = 88.44% | 54/3,348 = 1.61% |
| element | 2,987/3,080 = 96.98% | 2,291/3,080 = 74.38% |
| grammar marker | 17,146/17,176 = 99.83% | 426/17,176 = 2.48% |
| numeric | 19,204/30,796 = 62.36% | 17,516/30,796 = 56.88% |
| text byte（meta 与 TextObject 混合） | 1,131/1,260 = 89.76% | 188/1,260 = 14.92% |
| traversal control | 5,833/6,728 = 86.70% | 29/6,728 = 0.43% |

raw control token 也保留用于审计：

| raw control | 清洗 B0 | 候选 A |
|---|---:|---:|
| `CHEMTRAVERSE_END` | 42/100 = 42.00% | 0/100 = 0.00% |
| `COMPONENT` | 100/100 = 100.00% | 0/100 = 0.00% |
| `COMPONENT_END` | 20/100 = 20.00% | 29/100 = 29.00% |
| `DOWN` | 2,515/2,980 = 84.40% | 0/2,980 = 0.00% |
| `EOS` | 100/100 = 100.00% | 0/100 = 0.00% |
| `LINK` | 261/368 = 70.92% | 0/368 = 0.00% |
| `UP` | 2,795/2,980 = 93.79% | 0/2,980 = 0.00% |

候选 A 在合法集合内的 control 结果为：`COMPONENT 100/100`、`DOWN 2,142/2,980`
（71.8792%）、`UP 2,630/2,980`（88.2550%）、`LINK 269/368`（73.0978%）、
`COMPONENT_END 100/100`、`CHEMTRAVERSE_END 100/100`、`EOS 100/100`。

### Route100 自由生成

| 指标 | 清洗 B0 | 候选 A | 变化 |
|---|---:|---:|---:|
| completed | 67/100 | 100/100 | +33 |
| grammar valid | 67/100 | 100/100 | +33 |
| scene valid | 67/100 | 100/100 | +33 |
| chemically valid / mol valid | 2/100 | 32/100 | +30 |
| SDF valid | 1/100 | 31/100 | +30 |
| SDF roundtrip match | 1/100 | 31/100 | +30 |
| count-pair exact | 1/100 | 0/100 | -1 |
| Skeleton exact | 0/100 | 0/100 | 0 |
| atom-typed Skeleton exact | 0/100 | 0/100 | 0 |
| bond-typed Skeleton exact | 0/100 | 0/100 | 0 |
| typed graph exact | 0/100 | 0/100 | 0 |
| exact sequence | 0/100 | 0/100 | 0 |
| chemical match | 0/100 | 0/100 | 0 |
| generation seconds | 343.646 | 79.682 | -263.964 |

B0 的首因是 `generation_incomplete=33 / count_mismatch=66 / topology_mismatch=1`；候选 A 的
100 个样本全部变成 `count_mismatch`。平均生成长度也从 1,825.36 token 降到 413.72 token。

计数残差进一步表明候选不是“更准确地停止”，而是系统性过早停止：

| 计数 | 清洗 B0 | 候选 A |
|---|---:|---:|
| atom MAE | 24.07 | 13.42 |
| atom under / exact / over | 56 / 2 / 42 | 97 / 2 / 1 |
| atom 误差范围 | -53 .. 71 | -43 .. 1 |
| atom 预测均值 / GT 均值 | 27.85 / 30.80 | 17.40 / 30.80 |
| atom Pearson correlation | -0.2437 | 0.3958 |
| bond MAE | 33.77 | 15.93 |
| bond under / exact / over | 55 / 4 / 41 | 99 / 0 / 1 |
| bond 误差范围 | -58 .. 102 | -44 .. 1 |
| bond 预测均值 / GT 均值 | 38.01 / 33.48 | 17.57 / 33.48 |
| bond Pearson correlation | -0.2253 | 0.3430 |

### 预注册门逐项判定

| 预注册门 | 清洗 B0 | 候选 A | 判定 |
|---|---:|---:|---|
| completed / grammar / scene 均 >=85，且不低于 B0 超过 2 | 67 / 67 / 67 | 100 / 100 / 100 | **PASS** |
| SDF valid >=10，且 >=B0+5 | 1 | 31 | **PASS** |
| count-pair exact >=5，且 >=B0+3 | 1 | 0 | **FAIL** |
| Skeleton exact >=3，且 >=B0+2 | 0 | 0 | **FAIL** |
| typed graph exact >=1，且 >=B0+1 | 0 | 0 | **FAIL** |
| grammar-masked `COMPONENT_END` >=85/100 | 未记录 | 100/100 | **PASS** |
| grammar-masked `LINK` >=75% | 未记录 | 269/368 = 73.10% | **FAIL** |
| non-forced decision accuracy >=77.5% | 未记录 | 75.63% | **FAIL** |
| element / chemical bond type >=95% / 96% | 96.98% / 97.46% | 74.38% / 8.51% | **FAIL** |
| 真实 TextObject byte >=80% | 未记录 | 48.85% | **FAIL** |

候选 A 只通过 completion、SDF 和 `COMPONENT_END` 三组门槛，其余七组失败。最关键的 count、
Skeleton 和 typed graph 全部没有形成 exact；这不是接近阈值的边界情况。

### 决策

候选 A 明确 **NO-GO**。它证明 grammar-aware loss 能让序列可靠结束并显著增加可写出的 SDF，
但当前平衡权重把模型推向系统性过早结束：97/100 atom under、99/100 bond under，最终 100/100
都是 count mismatch。与此同时，真实 TextObject、元素和键类型指标明显退化。

按预注册规则，不追加 epoch、不与候选 B 组合、不因 SDF valid 上升而放宽 graph exact 门槛，
也不读取 confirm400。confirm400 在本次审查中保持未消费。

## 对五类目标的最终判断

| 方案 | 当前判断 | 原因 |
|---|---|---|
| class-balanced per-token weighting | 候选 A NO-GO | completion 和 SDF 上升，但系统性过早停止且 graph exact 为 0 |
| focal loss | 暂不进入 | 会混合困难样本与不可观察/高噪声位置，A 之前没有增量证据 |
| scheduled sampling | 暂不进入 | 长 AR 串行 rollout 成本高、梯度有偏且实现变量多 |
| token corruption | 保留固定 5% | 单 forward、已有实现，可低成本检验 exposure bias |
| Professor Forcing | 拒绝本阶段 | 额外生成轨迹与对抗训练，不符合 capacity pilot 的低成本要求 |

参考方法背景为 Scheduled Sampling（Bengio et al., 2015）、Professor Forcing（Lamb et al.,
2016）、Focal Loss（Lin et al., 2017）和 Class-Balanced Loss（Cui et al., 2019）。本审查的
路线选择依据是本项目 P2 审计，而不是直接照搬这些方法。
