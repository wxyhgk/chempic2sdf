# ChemMolSeq v0.3 协议

ChemMolSeq v0.3 面向图片到化学结构生成，目标是让主化学序列依赖图片中可推导
的顺序，同时保留 SDF 所需的二维布局。

v0.2 仍然保留，现有 checkpoint 不受影响。训练时通过
`--sequence-version 0.3` 显式启用新协议。

## 核心变化

1. 原子不再使用 RDKit canonical rank 顺序，而是按当前图片的 `y、x、原 ID`
   排序。
2. 原子和键的记录位置隐含 record ID，不再输出显式 atom ID 和 bond ID。
3. 删除化学键记录中的 `display_type` 和 `direction`，这些字段继续由视觉辅助
   任务学习。
4. 坐标移动到全部原子和键之后，不再成为后续化学键生成的前缀。
5. 四面体 `CW/CCW` 相对于按空间原子 ID 排序的邻居规范化，避免 bond 插入
   顺序改变绝对手性。

## 文档结构

```text
<mol v=0.3 order=spatial-yx>
<a ...>
<b ...>
<coords>
<p x y>
<coords_end>
<mol_end>
```

## 原子记录

```text
<a element charge isotope aromatic explicit_h no_implicit radical canonical_chiral>
```

原子 ID 是记录出现顺序，从 0 开始连续递增。

## 化学键记录

```text
<b begin end type stereo stereo_atom_0 stereo_atom_1>
```

bond ID 是记录出现顺序。`begin/end` 引用空间顺序原子 ID。bond record 保留 RDKit
在空间重编号后的邻接插入顺序，以保证复杂四面体和 E/Z 结构无损重建。

## 坐标区

坐标记录数量必须等于原子数量，并与空间原子顺序一一对应，数值范围为 0 到
1023。tokenizer 只编码一个坐标区开始标记、连续的 `x/y` 数字和一个结束标记。

## 两层验收

化学验收不应用 Render 坐标：

```text
ChemMolSeq v0.3
    -> 无坐标 RDKit Mol
    -> sanitize
    -> canonical isomeric SMILES / InChIKey
```

SDF depiction 验收再应用终端坐标，并要求写出、重新读取后的 InChIKey 不变。
如果增强布局已经把 E/Z 几何移动到与化学标签冲突的位置，SDF 导出必须拒绝。

首个训练 shard 的实测结果：

- 4,992/4,992 化学往返通过。
- 发现 13 张原始 `stress/coordgen` 图片存在 depiction-stereo 冲突。
- 平均长度从 v0.2 的 767 token 降到 639 token，减少 16.71%。
- P99 从 2,006 降到 1,658，最大长度从 2,481 降到 2,062。

详细报告见 `chemmolseq_v0_3_shard_validation.json`。
