# ChemTraverse 解码路线复核与四臂实验矩阵

> 日期：2026-08-01
>
> 性质：架构调研和预注册实验设计，不是训练结果，不创建模型版本或 tag
>
> 数据边界：只使用冻结训练清单、`overfit_32`、`train_5k` 和 `dev_500`；不使用任何外部、
> 已暴露或封存评测集

## 结论

ChemTraverse 应当保留，但应当把它从“唯一的长自回归模型输出”降为统一的无损编译和诊断
协议。下一阶段不应直接启动 50K 长序列训练，也不应退回匿名 free-query DETR。优先比较两种
有明确归纳偏置的路线：

1. **record-level 分段 traversal**：保留 DFS、`down/up/link` 和显式引用，但一次预测完整记录，
   记录内字段并行输出，并显式预测 count 和维护栈状态。
2. **层级空间锚定集合图**：先预测 count-conditioned Atom slots，再预测以 Atom slots 为条件的
   Bond group queries；每根键直接输出两个 Atom pointer，最后确定性编译为 ChemScene、
   ChemTraverse 和 SDF。

推荐主候选是第二条中的层级模型 `G3`，分段 traversal `G1` 是必要对照。联合 Atom/Bond 的
空间集合 `G2` 用来判断“层级条件化”本身是否必要；当前 token 级长 AR `G0` 只作为锁定基线。

在四臂训练前，必须先完成一个共同修复：**模型不得预测 `molecule_id`**。它不是图片可观测量，
应由 sample sidecar 在编译后回填。短期兼容旧协议时，也必须强制注入并屏蔽对应 loss，不能让
decoder 消耗步骤去逐字节猜 ID。

## 现有证据

### 长自回归的实际负担

冻结 50K manifest 的 ChemTraverse token 审计如下：

| 分位 | P50 | P95 | P99 | 最大 |
|---|---:|---:|---:|---:|
| token 数 | 781 | 1,451 | 2,144 | 2,874 |

同一批 50K 目标按完整 ChemTraverse record 计数，结果为：

| 分位 | P50 | P95 | P99 | 最大 |
|---|---:|---:|---:|---:|
| record 数 | 98 | 183 | 270 | 349 |

这说明 record-level decoder 可以把串行决策步数缩短约 8 倍，同时不丢掉 ChemTraverse 的图
遍历语义。当前 50,165,232 参数模型中，视觉编码器为 13,658,080 参数，decoder 和其他部分为
36,507,152 参数。模型并不小到没有容量；主要问题是目标组织方式。

`train_5k -> dev_500` capacity 实验也给出了直接因果证据：

- 32 图可过拟合到 `30/32` sequence/count/Skeleton exact，证明协议可表达且模型有容量。
- 5K 训练 3 epoch 后 teacher token accuracy 为 81.43%，但自由生成只完成 71.2%。
- count-pair exact 为 0.4%，Skeleton exact 为 0，SDF valid 为 2.6%。
- 元素和 chemical bond type 的 teacher accuracy 已分别达到 96.3% 和 96.9%，主要错误集中在
  count、停止、`COMPONENT_END` 和 `LINK`。
- 把所有 traversal control token 统一乘 4 反而降低完成率，说明高频 `DOWN/UP` 会淹没稀有
  结束决策，不能继续做统一 loss 加权。
- 500 张自由生成耗时 1,398.24 秒，即约 2.80 秒/图 GPU。这个量级不适合作为未来 CPU 主路径。

额外的 token 分桶审查表明 numeric、text byte 和 bond orientation 仍明显低于元素/键型。
一次早期字段错误会改变后续 DFS 栈和引用编号，因此即使 grammar mask 排除了非法 token，长前缀
的 exposure bias 仍然存在。

### count 重要，但 count-only 尚未被证明充分

P2 的绝大多数完成样本发生 count mismatch，但仅修复 count 不能自动保证拓扑正确。下一步应先
运行便宜的 GT-count oracle gate：强制真实 Atom/Bond 数量，只让模型决定遍历动作、属性和
拓扑。如果 exact count 下 Skeleton 仍低于 `5/100`，就应淘汰“给现有长 AR 加一个 count head
即可”的路线，转向 record-level 或显式图集合。

DETR 官方实现中的 `cardinality_error` 只是日志项，不反向传播。因此本项目不能假设 Hungarian
matching 会自然学会数量；所有集合臂都需要独立、可反传的 Atom/Bond count 交叉熵。

### 匿名 free-query 已有反例

仓库已有联合 Atom/Bond/incidence capacity 模型。它用 128 Atom query、128 Bond query、
Hungarian matching、count head 和 endpoint pointer；后续还加入固定参考网格与两级局部 patch。
固定 100 张图训练到 5,000 step 后：

| 指标 | 结果 |
|---|---:|
| Atom count exact | 100% |
| Bond count exact | 100% |
| Atom 6px accuracy | 18.93% |
| Bond port pair 6px | 8.22% |
| Incidence endpoint accuracy | 19.78% |
| Skeleton exact | 0% |

问题不是缺 count head，而是 query 没有由当前图片的真实局部证据建立稳定身份。固定稀疏网格只是
初始化，不等于 image-conditioned proposal。`G2/G3` 必须从 dense heatmap、line evidence 或
多 slot cell 中选择空间 anchor，不能再让整组匿名 query 从全局自由争抢对象。

### 50K 容量适配

对冻结 manifest 的结构 JSON 做只读审计：

| 对象 | P50 | P95 | P99 | 最大 |
|---|---:|---:|---:|---:|
| Atom 数 | 29 | 54 | 79 | 100 |
| Bond 数 | 31 | 58 | 84 | 111 |

因此 128 Atom slots 和 128 Bond slots 能覆盖当前 50K 的全部样本，没有必要为了极少数样本增加
到 256 或 300 slots。容量必须由 commitment 审计决定，不能在评测后修改。

## 现有项目与论文能借什么

### StarVector

本地 `third_party/star-vector` 使用视觉 embedding 加 causal LM 生成 SVG。1B 配置的上下文为
8,192，8B 配置可到 16,000；这证明“图片到可执行矢量语言”可行，但它没有解决本项目已经观察到
的长 AR 停止、引用漂移和 CPU 延迟。化学图比通用 SVG 稀疏且类型固定，不需要照搬 1B/8B 的
通用代码生成器。

可借：统一可编译语言、grammar validation、结构化错误留存。

不借：大语言模型规模、逐 token 长序列作为唯一结构头。

### MolGrapher

复核本地 MolGrapher commit `33c8c4d244dd8485098f30d6697f92a4cf93d4c8`：

- KeypointDetector 输出 heatmap，再用阈值和区域极大值取 keypoint。
- GraphConstructor 每个 query 只主动加入最多 6 个近邻候选。
- 当前代码默认删除超过估计 bond length `2.5x` 的候选；bond length 来自最近距离的 75 分位。
- 还会用中心空白和中间 keypoint obstruction 规则预删候选。
- GraphClassifier 在原图 feature map 的原子位置和键上三等分位置采样特征，再分类 Atom/Bond。

可借的是**稳定空间对象接口、沿键采样视觉证据、对象后消息传递**。不能借的是近邻数量、键长、
中心空白和 obstruction 硬删除，因为真键一旦没进入候选集，后续分类器无法恢复。

### DETR 与 Relationformer

DETR commit `29901c51d7fe8712168b8d0d64351170bc0f83e0` 提供 fixed slots、no-object 和
Hungarian 一对一匹配。Relationformer road-network branch commit
`e5328e33948929d8de58586efbf6b84355d747cf` 在 object tokens 外加入 relation token，并用
Deformable DETR 的多尺度 reference points学习图节点。

Relationformer 的关系推理仍枚举所有已选 node pair，复杂度为 `O(V^2)`；训练时还对负边采样。
这对平均 30 个 Atom 尚可，但对 100 Atom 图会产生 4,950 个 pair，并重新引入“所有点对是否有
键”的决策负担。本项目应借它的联合 object/relation 表征和多尺度稀疏注意力，但把 relation
head 改成 `O(E x V)` 的 Bond queries + 两个 endpoint pointer。化学图稀疏，`E` 与 `V` 同阶。

## 共同前置协议

四个实验臂在模型差异之前必须共享以下处理：

1. `molecule_id` 从模型 target 中删除，由 dataloader sidecar 保存，编译 SDF 后回填。
2. width/height 是固定 768 时也不应让模型重复预测；作为 decoder condition 传入。
3. 开放文字识别使用同一个共享 TextDecoder。结构路线只预测 TEXT role、anchor 和 bbox；文字
   字符串不参与结构 checkpoint 选择，避免 OCR 差异污染架构结论。
4. 所有路线生成同一 ChemScene 字段，并使用同一确定性 serializer/validator 写
   ChemTraverse 和 SDF。
5. 双键、三键各是一条 BondObject，不展开为 4/6 个 endpoint。
6. 不使用键长上限、最近邻裁剪、中心空白硬删除或按特定分子家族写规则。

旧协议兼容期若暂时不能删除 meta ID 字段，应由 evaluator 强制注入固定 token，并将这些 token
从 loss、token accuracy、生成时延和首分歧统计中排除。让模型学习同一个固定字符串也不算完成
sidecar 化。

## 训练前的 O0 因果门

O0 不训练新模型，只在锁定的 P2 checkpoint 和固定 route100 上做两项干预：

由于旧 checkpoint 已经见过 meta ID 位置，O0 只能把 sidecar ID token 强制喂入 decoder cache，
但不调用模型预测、不计 loss/accuracy/latency；不能直接删 token 并移动全部后续位置。四臂重训时
才从 target schema 中真正移除该字段。

| 干预 | 内容 | 回答的问题 |
|---|---|---|
| O0-A | sidecar meta；其余保持当前 grammar greedy | 不可观测 ID 造成多少前缀污染 |
| O0-B | O0-A + GT Atom/Bond count 硬预算 | count/stop 修好后，现有语义和拓扑还有多少上限 |

O0-B 必须同时满足：

- completed `>= 90/100`；
- 输出 Atom/Bond 数服从 oracle budget `>= 80/100`；
- Skeleton exact `>= 5/100`。

若完成或预算服从不达标，现有 token AR 控制机制直接 NO-GO。若前两项达标但 Skeleton 仍小于
5，`long AR + count head` 这个 count-only 修补 NO-GO；仍继续比较 `G1/G3`，因为它们同时改变
暴露链和对象身份，而不是只改 count。

## 四臂 5K 公平矩阵

| 臂 | 模型 | 串行长度 | count | endpoint 建模 | 角色 |
|---|---|---:|---|---|---|
| G0 | sidecar-meta token AR | 不超过旧协议 2,874 token | 隐式 | DFS token/reference | 锁定基线 |
| G1 | record-level segmented traversal | 约 98 P50 / 349 max record | 显式 CE | DFS stack + pointer | 分段候选 |
| G2 | image-anchored joint Atom/Bond set | 固定 128+128 slots | 显式 CE + top-k | Bond endpoint pointer | 集合消融 |
| G3 | hierarchical Atom -> Bond group set | 预测 A slots 后预测 B slots | 显式 CE + top-k | 条件化双 pointer | 推荐候选 |

### G0：修正后的长 AR 基线

- 保留 token-level ChemTraverse 和 grammar-constrained greedy。
- 只接受共同的 sidecar-meta/TextDecoder 修复，不加 count head。
- 重新训练，不能直接拿旧 P2 数字当新共同协议的基线。

### G1：record-level segmented traversal

- action decoder 只预测 `component/atom/down/up/link/component_end/text/stereo/end`。
- 每一步的坐标、Atom 属性、Bond 属性、orientation 和 reference 由并行 field heads 输出。
- decoder 显式接收当前 DFS stack、已生成 Atom/Bond 数和预测总 count。
- `link/text/stereo` 使用 pointer distribution 指向已有 record state，不生成十进制引用 token。
- 分段边界必须是完整 record；禁止每 128 token 任意切块，那样仍会切断栈语义。

### G2：image-anchored joint set

- 从 stride-4/stride-8 dense evidence 选择 image-conditioned Atom/Bond anchor。
- 每个 cell 必须允许多 slot，避免拥挤区域同 cell 冲突。
- Atom/Bond 分别做 Hungarian matching；matching cost 包含几何、role/type 和 incidence。
- count head 决定 top-k，不用统一 presence threshold 猜整图数量。
- Bond endpoint 对 Atom slot 做 pointer softmax；不做最近邻吸附。

### G3：hierarchical anchored graph

- 第一阶段预测 Atom count、空间 Atom slots、role/anchor/bbox。
- 第二阶段在视觉 memory 和已选 Atom slots 上预测 Bond count 与 Bond group queries。
- 每个 Bond query 同时输出两个 endpoint pointers、两个可见 port、display/chemical type、方向和
  stereo；端点交换使用无序 matching。
- 复杂度为 `O(A + B + A*B)`，不枚举 `O(A^2)` AtomPair。
- Atom/Bond 全部确定后再并行预测属性并编译 ChemScene，不需要 DFS 生成才能得到 SDF。

G2 与 G3 的核心差别是：G2 的 Atom/Bond 身份同时竞争，G3 让 Bond 身份建立在已经选择的 Atom
slots 上。若 G2 失败而 G3 通过，证明层级条件化是必要归纳偏置；若二者都通过，选择更快、更小
且种子方差更低的一臂。

## 公平性冻结项

### 数据和优化

- train：同一 `train_5k`，3 epoch，共 15,000 图片曝光。
- dev：同一 `dev_500`；前 100 为 route，后 400 在 route 通过前不可读取。
- batch 4、gradient accumulation 1、3,750 optimizer updates。
- AdamW、LR `3e-4`、weight decay `0.01`，完全相同的 LR schedule。
- 图像 768x768、BF16、deterministic、同一无 shuffle 样本顺序。
- 同一 vision checkpoint SHA256：
  `65a443d26bbffd41359604ac8147fa62a66ee18ca877693ebb3dea0b0a637095`。
- seed 首轮固定 `20260713`；进入 50K 后使用三个预注册 seed。
- 固定 final step 评测，不按 teacher token loss 选 checkpoint。

### 参数和计算

- G0 当前总参数为 50,165,232。
- G1/G2/G3 目标范围为 48M-52M，总参数差不超过 5%。
- 现有 Atom/Bond set 配置改成 `d_model=384, layers=6, FFN=1536` 时约 49.39M，可作为
  G2/G3 参数预算起点，不需要添加无功能参数凑数。
- 所有臂报告训练 FLOPs、每秒图片、峰值显存、checkpoint 大小和单图生成时延。
- 主比较按相同图片曝光和 update；另报告共同最小 GPU 时间点的学习曲线，但不允许给快模型追加
  新图片来制造数据优势。
- batch 4 峰值显存必须小于 30 GiB。某臂 OOM 时不能只给它降低 batch；要么所有臂统一改
  accumulation，要么该臂因部署不可行而淘汰。

### 输出和指标

所有臂必须输出：

- Atom/Bond count exact、count-pair exact；
- Skeleton exact、Atom-typed exact、Bond-typed exact；
- count 正确条件下 Skeleton exact；
- ChemScene parse、SDF valid、SDF roundtrip、chemical match；
- incomplete、count mismatch、topology mismatch、type mismatch、sanitize failure 首因；
- GPU/CPU batch-1 P50/P95 latency。

sequence exact 只对 G0/G1 有意义，不得用于跨四臂排名。grammar valid 对 G2/G3 是 serializer
工程门，不是模型能力指标。

## 分阶段淘汰门

### P0：overfit_32

每臂固定最多 2,400 updates，不追加：

- count-pair exact `>= 30/32`；
- Skeleton exact `>= 30/32`；
- SDF valid `>= 30/32`；
- G0/G1 grammar/scene `32/32`；G2/G3 serializer/scene `32/32`。

任何一项失败，说明实现或表示容量不足，不进入 5K。G2/G3 不要求 sequence exact。

### P1-A：route100

最终 step 单次评测，通用硬门：

- count-pair exact `>= 20/100`；
- Skeleton exact `>= 5/100`；
- SDF valid `>= 20/100`；
- count 正确条件 Skeleton exact `>= 25%`。

额外门：

- G0/G1 completed `>= 90/100`；
- G2/G3 ChemScene 可编译 `>= 95/100`；
- 任一错误样本都必须保留序列或对象 JSON、失败 SDF 和 stage error，不得静默跳过。

未过门立即淘汰，不读取 confirm400。特别地，count-pair 达标但 Skeleton 为 0，不能解释成“再加
数据就行”；这说明 count 与拓扑没有形成联合表示。

### P1-B：confirm400

只有通过 route 的臂才能运行。合并成完整 dev500 后必须保持：

- count-pair exact `>= 15%`；
- Skeleton exact `>= 5%`；
- SDF valid `>= 20%`；
- count 正确条件 Skeleton exact `>= 25%`；
- route 到 confirm 的任一主要比例下降不得超过 5pp。

通过者按预注册词典序排名：

1. Skeleton exact；
2. count-pair exact；
3. Bond-typed exact；
4. SDF valid；
5. GPU P95 latency；
6. 参数量和三个 seed 的方差。

最多保留两臂进入 50K。若没有臂通过，停止扩量，回到视觉 anchor/标签协议审计。

### P2：50K 三种子

只训练 P1 前两名：每臂 3 seed，每个 seed 恰好看 50,000 张一次。需要从同一训练来源预先冻结
一个 molecule-disjoint internal dev，且不得与 `train_5k/dev_500` 重叠。固定保存 10K、25K、
50K exposure checkpoint，但只评测固定 50K 终点，避免多 checkpoint 选择偏差。

进入后续正式评测前，候选相对 G0 三 seed 中位数必须：

- count-pair exact `+5pp`；
- Skeleton exact `+5pp`；
- SDF valid 不低于 `-2pp`；
- count 正确条件 Skeleton exact 不低于 `-1pp`；
- 最差 seed 的 Skeleton exact 仍高于 G0 对应 seed；
- 推理不得依赖键长、近邻或按分子类别编写的规则。

## 多 GPU 调度与预算

当前机器只检测到 1 张 RTX 5090 32 GB，因此本机只能顺序执行。下表是有 4 张同规格 GPU 时的
公平调度，不使用 DDP；每个实验保持单卡 batch 语义。

| GPU | P0/P1 任务 | 最大预算 |
|---|---|---:|
| GPU0 | G0 | 1 GPU-hour |
| GPU1 | G1 | 1 GPU-hour |
| GPU2 | G2 | 1 GPU-hour |
| GPU3 | G3 | 1 GPU-hour |

P0 overfit 每臂上限 0.5 GPU-hour；P1 train5k + route100 每臂总上限 1 GPU-hour。通过 route 的臂
追加 confirm400，上限 0.5 GPU-hour/臂。现有长 AR 的 3,750 updates 训练约为分钟级，但自由生成
500 张已经需要约 23 分钟，因此预算主要被解码占用。

P2 是 2 臂 x 3 seed，共 6 个单 GPU job：

- 每 job 上限 2 GPU-hour；总上限 12 GPU-hours。
- 6 GPU 并行约 2 小时 wall time；4 GPU 为两波、约 4 小时；当前单卡上限约 12 小时。
- 每 job 最多保存 3 个约 0.6 GB checkpoint；模型、优化器、结果和失败样本总磁盘预算 20 GB。
- 三个 seed 要轮换 GPU 编号，避免把单卡性能差异与架构绑定。

任一实验超出 GPU-hour 或显存上限即停止，不通过“换更大模型”规避归纳偏置失败。

## 执行顺序

1. 完成 `molecule_id` sidecar 化，并写 serializer/evaluator 单测。
2. 跑 O0-A/O0-B，淘汰 count-only 假设。
3. 实现最小 G1 和 G3；G2 复用 G3 的 anchor、matching 和字段头，只移除层级条件化。
4. 四臂分别过 P0，随后并行 P1-A。
5. 只让过门臂读取 P1-B，冻结前两名。
6. 才准备 P2 internal dev commitment 和 2 x 3 seed 50K 训练。
7. P2 结束前不访问任何外部或封存评测数据，也不创建正式 tag。

如果资源只够先做一条新路线，顺序是 `O0 -> G3 -> G1`。G3 直接针对 count、空间身份和
endpoint incidence；G1 用来判断 traversal 语义在缩短暴露链后是否仍有独立价值。

## 参考实现与文献

- StarVector 本地实现：`third_party/star-vector`，源码归档对应上游 commit
  `0e083c1911760aa31bc576ca7f337a7f8ee605ec`。
- MolGrapher：<https://github.com/DS4SD/MolGrapher>，本地复核 commit
  `33c8c4d244dd8485098f30d6697f92a4cf93d4c8`。
- DETR：<https://github.com/facebookresearch/detr>，复核 commit
  `29901c51d7fe8712168b8d0d64351170bc0f83e0`。
- Relationformer：<https://github.com/suprosanna/relationformer>，road-network branch 复核
  commit `e5328e33948929d8de58586efbf6b84355d747cf`。
- Relationformer paper：<https://arxiv.org/abs/2203.10202>。
- Deformable DETR：<https://arxiv.org/abs/2010.04159>。
- ChemTraverse capacity 记录：
  `docs/model/training/20260801-chemtraverse-a1-capacity-pilot.md`。
- Free-query capacity 记录：
  `docs/model/training/20260723-v0.9.43-joint-free-query-capacity.md`。
