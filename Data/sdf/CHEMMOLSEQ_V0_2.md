# ChemMolSeq v0.2 协议

ChemMolSeq 是模型实际生成的紧凑文本语言。SDF 仍是权威结构真值，
ChemMol JSON 和 Render JSON 是数据交换与辅助标注格式。

训练加载器根据 ChemMol JSON 与当前图片的 Render JSON 动态生成
ChemMolSeq v0.2，不直接使用数据分片中旧的 v0.1 序列。

## 文档结构

```text
<mol v=0.2>
<a ...>
<b ...>
<mol_end>
```

所有原子必须位于所有化学键之前。原子和键的编号都必须从 0 开始、连续并
按照编号排序。

## 原子记录

```text
<a id element charge isotope aromatic explicit_h no_implicit radical chiral x y>
```

- `id`：规范化原子编号。
- `element`：元素符号。
- `charge`：形式电荷。
- `isotope`：同位素质量数；第一版数据通常为 0。
- `aromatic`：是否为芳香原子，取值为 0 或 1。
- `explicit_h`：原子属性中的显式氢数量，例如 `[nH]` 中为 1。
- `no_implicit`：是否禁止 RDKit 自动补充隐式氢，取值为 0 或 1。
- `radical`：自由基电子数。
- `chiral`：完整的 RDKit `ChiralType` 枚举名称。
- `x`、`y`：当前图片中的量化坐标，范围为 0 到 1023。

## 化学键记录

```text
<b id begin end type display_type stereo stereo_atom_0 stereo_atom_1 direction>
```

- `id`：规范化化学键编号。
- `begin`、`end`：两端原子编号。
- `type`：化学键的 RDKit `BondType`。
- `display_type`：当前图片实际显示的键型，例如芳香键可能显示成单双键。
- `stereo`：RDKit `BondStereo` 枚举名称。
- `stereo_atom_0`、`stereo_atom_1`：定义双键立体化学的参考原子；无引用时均为 -1。
- `direction`：当前图片中的 RDKit `BondDir`，包含楔形键和虚线键方向。

## 验收条件

每个进入训练的数据样本必须满足：

```text
ChemMol JSON + Render JSON
    -> ChemMolSeq v0.2
    -> RDKit Mol + 2D conformer
    -> sanitize
    -> canonical isomeric SMILES / InChIKey
    -> 与原始结构一致
```

模型推理结果也必须通过相同的解析和化学往返流程，之后才能导出 SDF。

`direction` 是当前图片的显示属性，不参与默认的化学结构重建。直接将图片中的
楔形/虚线方向与原子手性标签同时写回 RDKit，可能导致复杂分子的 InChI 立体层
发生变化。默认重建以 ChemMol 的 `chiral` 和 `stereo` 为化学真值；显示方向只
用于视觉辅助损失和重新渲染，需要时通过显式选项应用。
