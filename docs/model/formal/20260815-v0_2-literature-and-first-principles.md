# v0.2 文献核对与第一性原理映射

日期：2026-08-15

本记录只用于确定表示与训练边界，不用论文 benchmark 替代本项目的可辨识性证明。

## 核对的工作

1. LETR，*Line Segment Detection Using Transformers without Edges*，CVPR 2021，
   DOI `10.1109/CVPR46437.2021.00424`，arXiv `2101.01909`。
   论文直接用 tokenized queries 和 endpoint distance loss 预测线段，跳过 edge/junction/region
   启发式中间处理。它支持“Lane/Bond primitive 作为集合对象”，但不提供同像异图消歧定理。
2. Relationformer，*A Unified Framework for Image-to-Graph Generation*，ECCV 2022，
   DOI `10.1007/978-3-031-19836-6_24`，arXiv `2203.10202`。
   论文用 set-based object prediction 和 relation token 联合预测对象与关系，支持整组图目标而非
   独立 pair veto。它仍假设训练标签对输入定义了单一目标。
3. SelectiveNet，*A Deep Neural Network with an Integrated Reject Option*，ICML 2019，
   arXiv `1901.09192`。
   论文联合优化预测与拒识，并报告 risk-coverage；它支持把 `Ambiguous/Reject` 作为正式输出，
   但其置信度拒识不能替代逻辑碰撞证明。
4. *A review of optical chemical structure recognition tools*，Journal of Cheminformatics 2020，
   DOI `10.1186/s13321-020-00465-0`。
   该综述覆盖传统线条、OCR、规则与学习式 OCSR，说明图元解析长期是化学识图的核心；它没有
   证明 raster 到化学图在一般域上为单射。
5. MolNexTR，*A generalized deep learning model for molecular image recognition*，Journal of
   Cheminformatics 2024，DOI `10.1186/s13321-024-00926-w`。其参考链包含 MolScribe、
   Relationformer、ChemGrapher 与传统 rule-based OCSR，说明 image-to-graph 与图像序列两条路线
   都存在；模型效果仍不能解除相同像素对应不同语义的反例。

文献元数据由 arXiv API、OpenAlex 和 Crossref 在上述日期核对。

## 与本项目证据的对应

本项目历史实验已经给出比架构直觉更直接的约束：

- v0.8.13 的 Bond-first whole-set selector 在固定 500 张有容量，但 9k 泛化低于原阈值；说明
  “集合建模”必要但不充分。
- v0.9.22 只提高 junction negative 权重，删除 4 个 negative 的同时误删 21 个 positive；
  target 类别强调不能替代可见证据。
- v0.9.24 给 pair 加有序邻域后仍被旧方法严格支配；逐 pair 决策没有优化整个 lane component。
- v0.9.84 的全图 LSD 对极端长键 coverage 提升显著，但无条件补边 NO-GO；线段检测解决
  carrier 缺失，不解决 grouping 和 semantic interpretation。

因此各方法的职责必须严格拆开：

```text
LETR/LSD 类能力       -> Visible Lane proposals
Relationformer 类能力 -> LaneBundle/Port/Junction 的集合与关系预测
Selective prediction  -> 证据不足或逻辑碰撞时 Ambiguous/Reject
化学绘图 prior        -> 明示地从候选解释中选择协议答案
Lean + finite audit    -> 证明每一层允许声称什么、禁止声称什么
```

## 不可混淆的两种不确定性

```text
epistemic/model uncertainty
  同一真值存在，但模型或数据不足；可以通过训练、数据和校准改善。

logical/non-identifiability
  同一 observation 在允许域中存在不同图真值；任何确定性模型都不能同时正确。
```

SelectiveNet 一类方法主要处理前者。本项目 `double vs parallel singles` 和
`crossing vs junction` 的 exact raster collision 属于后者，必须通过缩域、公开 prior 或
`Ambiguous` 输出处理。

## 对 v0.2 的架构约束

- 先证明 `VisibleEvidence = extract(Observation)`，再讨论模型如何预测它。
- GT lane ownership、Bond bundle、Atom identity 不能伪装成 visible target。
- candidate enumeration 必须报告 completeness；只生成 prior 选中的候选会掩盖歧义。
- graph chooser 必须是 evidence/candidate-set 的确定性公开函数，不读 GT 或 RDKit 成败。
- 训练目标最终可采用 set prediction，但训练前必须先完成 Evidence factorization 和 Oracle
  decoder 的条件左逆。
