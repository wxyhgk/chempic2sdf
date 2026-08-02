# OCSR 路线审计：文字、键、风格与结构表示

> 日期：2026-07-31
> 范围：普通分子图片到 SDF；MR-TADF 为主应用 benchmark，USPTO 为 OOD 护栏
> 本轮决策：只调研和设计实验，不修改训练代码，不切换模型

## 结论

目前不是“纯 SVG 一定走不通”，而是当前流水线把黑色笔画过早拆成点、线和文字候选，
再用多个局部模型及规则恢复全图对象集合。这种分解会把一次文字/键混淆放大成原子数、
键数、拓扑和价态的连锁错误。

下一条主线不应继续给 v0.10.21 叠局部修补，也不应先押注一个独立 OCR。建议建立
`ChemTraverse` 试验路线：共享高分辨率视觉编码器直接生成**语义图遍历序列**，每一步交替
生成原子、可见文字、键及回边，随后确定性编译为 ChemScene 和 SDF。它保留“图元化、可解释、
直接输出 SDF”的目标，但不再采用“先预测全体顶点，再预测全体键”的独立集合范式。

这个判断同时来自本项目证据和最新文献：

- v0.10.21 在 MR-TADF 上 count-pair exact 只有 `49.09%`，但数量同时正确时 Skeleton exact
  达到 `97.96%`。连接规则不是第一瓶颈，整图对象的多检和漏检才是。
- Invalid-only OCR fallback 将 valid `677 -> 725`，Chemical exact 仍是 `264 -> 264`。
  OCR 是问题之一，但单独替换 OCR 没有修复主任务。
- 2026-07-30 发布的 OCSRGlyph 用普通 Swin-B/384 加六层序列解码器、约 94M 参数和约
  2.21M 有效训练样本，在其 USPTO benchmark 上报告 `96.2%` graph exact。它没有独立 OCR
  前端，说明普通分子并不必然需要先做 OCR。
- GTR-CoT 在同数据对照中，交替生成 atom-bond 的图遍历过程比“先全部 atom、再全部 bond”
  高 `2.86/3.11pp`。这与我们当前整图计数脆弱的现象直接相关。
- MolSight 发现原子定位与主序列联合优化会压制主任务收敛；坐标辅助头必须与主任务解耦。
- MolScribe、MolNexTR、MolParser 的消融一致表明：跨绘图风格的收益主要来自系统的渲染增强、
  真实图片和 curriculum，而不是简单增加同风格分子数量。

## 四类问题必须分开

| 问题 | 含义 | 当前证据 | 对策 |
|---|---|---|---|
| OCR 转写错误 | 已经找到正确文字区域，但 `N/O/Ph/CH3` 等内容读错 | v0.10.9 提高文字准确率；MR 仍有大量结构错 | 保留开放字符串识别，但不要把它当唯一主线 |
| Text/Bond 语义混淆 | 字符笔画被当作键，或键被当作 `-`、`=` | ChemicalOCR 明确报告 PaddleOCR 的同类错误；用户观察一致 | 视觉编码阶段学习语义对象，不能只靠通用 OCR 或 LSD |
| 对象集合计数错误 | 原子或键多一个/少一个，引起整图失败 | MR count-pair exact `49.09%`；条件 Skeleton exact `97.96%` | 改用交替图遍历生成，避免两个独立全局集合的基数耦合 |
| 风格域偏移 | 字体、字号、线宽、双键间距、布局、污染改变视觉分布 | 本项目四渲染 outcome 大量分裂；多篇消融均支持 | 分阶段增强和真实域微调，不从第一步同时加入全部强增强 |

这四类错误在最终 SDF 上可能表现相同。例如 `N` 的一条斜笔画被识别为键，会同时造成一个
假 BondObject、两个假端点、错误计数和无效价态。如果只按最终 SDF invalid 归因，会误判成
“化学 repair 不够强”。

## 关键文献证据

### 1. OCSRGlyph：简单端到端模型仍然很强

[MarkushGlyph and OCSRGlyph](https://arxiv.org/abs/2607.28532) 是 2026-07-30 发布的预印本，
尚无独立复现。其普通分子模型是 ImageNet 预训练 Swin-B、384 输入、六层 Transformer decoder，
字符词表 101、总参数约 94M，贪心解码 canonical SMILES。训练混合包含 PubChem-1M、
USPTO-680K 和立体化学过采样，有效曝光约 2.21M。

论文报告 USPTO canonical exact `93.8%`、graph exact `96.2%`、valid `99.6%`。错误中仍有
196 个连接错误、136 个纯立体错误和 12 个 invalid。它说明：

1. 约 50M--100M 的专用模型容量足够，不需要 1B 模型；
2. 大规模、正确对齐的数据可以让共享视觉编码器自己区分普通文字和键；
3. 其 benchmark 与 MR-TADF 风格、分子复杂度不同，不能把 `96.2%` 直接当作我们的预期值；
4. 论文输出 SMILES，不满足本项目的 SDF/坐标目标，但其编码器和训练规模仍是有效参照。

### 2. GTR-CoT：不要先独立数完所有点和所有键

[GTR-CoT](https://arxiv.org/abs/2506.07553) 使用图遍历形式交替输出原子和键，并将图片中显示的
缩写作为 superatom 真值保留，避免“图里是 Ph，标签却展开成苯环”的监督冲突。论文构建 1.3M
训练样本；在 351K USPTO 子集的受控实验中，图遍历 CoT 比直接 SMILES 高 `2.31pp`，比
atom-then-bonds 表示高 `2.86/3.11pp`。

这不是要求我们采用 3B VLM。真正可借鉴的是**输出顺序的归纳偏置**：新原子由一条已识别的键
引入，回边只引用已有原子，因此原子和键互相约束，不需要先独立决定两个全局数量。

### 3. MolScribe 与 MolSight：坐标有用，但优化方式很重要

[MolScribe](https://arxiv.org/abs/2205.14311) 在 200K 消融中，无增强时 Indigo/ChemDraw 为
`91.4/72.1`，有增强为 `95.9/89.7`。离散坐标 token 优于连续坐标回归。它还发现保留 SMILES
中的分支、环号等非原子 token 略有帮助，因为这些 token 提供图结构约束。

[MolSight](https://arxiv.org/abs/2511.17300) 则给出另一面：SMILES 与 bond classification 联合
有小幅正收益，但把 atom localization 一起联合训练会让模型过度优化位置、主序列难以收敛。
将坐标分支独立训练并增加专用 decoder layer 后才恢复。因此本项目可以保留量化坐标，但不能再
让独立坐标 loss 无条件主导共享编码器。

### 4. ChemicalOCR：通用 OCR 确实不适合化学图

[MarkushGrapher-2 / ChemicalOCR](https://arxiv.org/abs/2603.28550) 的 ChemicalOCR 基于约 256M
SmolDocling，先用 235K 合成化学 OCR，再用 7K 真实专利 crop 微调。论文明确观察到 PaddleOCR
会合并多个符号，并把化学键误作减号或等号。IP5-M 在 IoU > 0.5 时，PaddleOCR F1 为 `1.9`，
ChemicalOCR 为 `86.5`；合成预训练后的 OCR accuracy `27.6`，加入 7K 真实样本后为 `69.5`。

但该论文的消融也表明 OCR 分支主要补充 Markush 长文字和布局：纯视觉分支在普通 USPTO OCSR
为 `89.1%`，纯 OCR/VTL 为 `84.4%`，融合为 `89.8%`。所以 ChemicalOCR 支持“通用 PaddleOCR
会误导”，不支持“普通分子必须先完全 OCR 再识别结构”。

### 5. 风格增强与真实数据不是可选项

[MolNexTR](https://arxiv.org/abs/2403.03691) 的 ACS 消融从无增强 `60.8`，依次加入 render、
image、molecule、contamination 后为 `65.0/68.9/75.1/81.9`。Render augmentation 包括绘图器、
字体/字号、线宽、双三键间距和显式氢；image augmentation 包括缩放、模糊、压缩和噪声。

[MolParser](https://arxiv.org/abs/2411.11098) 使用 7.7M 合成样本和约 400K 人工复核真实 crop。
WildMol-10K 从无增强无 curriculum 的 `40.1`，提高到有增强 `69.5`，再到 curriculum `76.9`。
其课程先用短序列、无增强图片，随后逐步增加分子复杂度和增强强度。这正好回答“增强会不会导致
不好收敛”：如果一开始全部打开，确实可能；正确做法是课程式增加，而不是永久关闭。

### 6. 整图目标和错误分桶

[COMO](https://arxiv.org/abs/2604.23546) 用 minimum risk training 直接优化 validity、结构相似度
和 exact。相同额外数据仅继续 MLE 时，10 个 benchmark 中 5 个回退、2 个不变；使用 MRT 后
真实域明显提高。这说明相同数据量下，逐 token/逐对象损失未必对应最终分子指标。

[MolRecBench-Wild / MOSAIC](https://arxiv.org/abs/2605.05832) 定义 18 类视觉困难，其中包含
Decorated Text、Decorated Bond、Bond Crossing、Large Font、Long/Thick/Thin Bond、Long
Functional Group Name 和 Blurry Image。我们应复用这套思想给 MR-TADF 失败分桶，而不是把
所有错误统称为 OCR 或 hallucination。

### 7. 线段检测只能作为证据，不能独立决定化学键

[M-LSD](https://arxiv.org/abs/2106.00186) 证明轻量网络可以直接预测线段的起点、中心、终点，
并通过线段匹配、junction、长度和角度损失训练；长线可用重叠子段增强。HAWP、DeepLSD 等也
提供类似几何证据。

但是 `N/H/A` 的笔画本身同样是合法线段。没有 TextObject/BondObject 语义上下文时，任何 LSD
都会忠实检测这些笔画。因此 LSD 最适合作为视觉 encoder 的辅助通道或候选证据，不能成为最终
BondObject 判决器。

## 推荐的新表示：ChemTraverse v0.1

目标不是输出 SMILES，而是输出可无损编译成 SDF 的、唯一化的语义遍历序列。示意形式如下：

```text
<mol>
  <atom> C <xy> 412 731
  <bond> SINGLE <atom> N <xy> 508 640
  <branch> <bond> DOUBLE <atom> O <xy> 603 640 </branch>
  <bond> SINGLE <atom_text> Ph <xy> 508 522
  <ring> 0 <bond> AROMATIC
</mol>
```

实际协议需要做到：

1. 根节点按图片坐标确定，分支按固定方向排序，保证一个图只有一个训练序列；
2. 每引入一个原子必须同时说明来自哪条键，回边只能引用已出现原子；
3. 双键、三键和楔形键始终是一个语义 BondObject，不把内部物理线数当顶点数；
4. 无标签骨架顶点输出隐式 `C`；可见 `Ph/CH3/R1/A` 原样输出开放字符串；
5. 坐标采用离散 token，并可在第一阶段降低 loss 权重或停止其梯度进入共享 encoder；
6. 解析器先生成 ChemScene，再写 SDF；评价直接比较规范化分子图，不用 SMILES 作为真值协议。

这条路线同时保留 StarVector 的“语义图元序列”思想、MolScribe 的离散坐标、GTR-CoT 的
atom-bond 交替约束，但不照抄任何一种模型，也不引入 1B 级通用 SVG/VLM。

## 下一轮只做三个可证伪实验

### 实验 A：错误分桶，不训练

从 MR-TADF 固定抽取 200 个错误和 100 个正确样本，按 MOSAIC 风格标签及以下因果桶复核：

- 文字笔画生成假键/假点；
- 键生成假文字；
- 隐式碳漏检或重复；
- 单/双/三键对象数量错误；
- 数量正确但拓扑错误；
- 原子类型、键级、立体或编译错误；
- 字体、字号、颜色、布局、密度、键长分桶。

门槛：至少 80% 的失败能归入明确的首因；否则当前日志不足以指导新架构。

### 实验 B：同编码器、同数据、同计算量的表示 A/B

- A0：当前 Atom-set + Bond-set 目标；
- A1：ChemTraverse 主序列，不加 OCR/坐标辅助头；
- A2：ChemTraverse + 低权重量化坐标 token + Text/Bond 语义 mask 辅助。

先用 50K 图片做容量/收敛对照，再用独立 fresh-dev 选方向。比较 count-pair exact、Skeleton
exact、Chemical exact、valid、每种错误桶和 CPU 解码耗时。若 A1/A2 在相同计算量下不能明显
提高 count-pair exact，说明图遍历表示假设不成立，停止扩量。

### 实验 C：课程式扩量

只有实验 B 通过后才使用现有 `webdataset_full_v0`、`style_train_100k` 和合法的真实训练域：

1. clean 黑白、短序列、简单分子；
2. 增加复杂环系、大分子及完整 720K 合成图；
3. 逐步加入字体、字号、线宽、双键间距、颜色和布局变化；
4. 最后加入真实 crop 或强污染，始终保留 clean replay 防止遗忘；
5. MLE 收敛后才试图加入 SDF-valid、count、graph similarity、exact 的分子级 risk loss。

MR-TADF 继续只做主应用 benchmark，不用于训练、阈值选择或 checkpoint 选择。USPTO 保留为
风格差异很大的 OOD 护栏，不再单独决定模型方向。

## 暂停的方向

- 暂停继续调近邻删除阈值、局部 line-band 阈值和更多确定性 repair；它们不能修复漏对象。
- 暂停把 PaddleOCR 重新接成教师；文献已直接证明其化学笔画定位存在系统偏差。
- 暂停单独扩大同一绘图器、同一风格的数据；先验证课程和表示。
- 暂停纯 LSD 直接生成键；它没有文字语义，必然把部分字形笔画当线。
- 暂停 1B/3B VLM 或 MoE。现有证据支持 50M--100M 专用模型先验证归纳偏置，5090 足够训练。

## 最终判断

用户关于“文字和键混淆、风格多样性”是对的，但它们不是两个孤立分类器精度问题。更深层的
问题是当前流水线先独立建立对象全集，再恢复关系，导致局部语义错误通过基数和组合关系放大。

数据也确实还不够多样：当前 100K style 数据远少于最新强模型的百万级曝光，并缺少大规模真实
crop；但过去“同表示直接扩量”没有稳定收益，说明不能把下一步简化成继续堆图片。应先用 50K
受控 A/B 证明 ChemTraverse 表示能提高整图计数，再按 curriculum 扩到现有全量数据。这样可以
把“路线错了”和“数据不足”两个假设真正区分开。
