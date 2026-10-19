# ChemTraverse v0.1 协议

ChemTraverse v0.1 是图片到 SDF 新路线的第一版原子、键交替遍历协议。模型不再先独立生成
全部 AtomObject，再独立生成全部 BondObject，最后依靠全局集合选择器拼图；它沿分子图逐步
输出“原子 -> 一根键 -> 邻接原子”，闭环时显式引用已经出现的原子。

```text
图片
-> 高分辨率共享视觉编码器
-> ChemTraverse 原子/键交替序列
-> ChemScene
-> SDF
```

协议是训练目标和诊断载体，不是 SMILES。最终产物仍为 SDF；SMILES 不作为模型输出协议。

## 记录格式

```text
<chemtraverse v=0.1 units=q1023>
<meta 768 768 utf8:6d6f6c6563756c652d31>
<component>
<atom 100 100 IMPLICIT C 0 0 0 0 0 0 ->
<down F SINGLE SINGLE NONE STEREONONE>
<atom 200 100 TEXT O 0 0 0 0 0 0 ->
<up>
<link 0 F SINGLE SINGLE NONE STEREONONE>
<component_end>
<text 1 205 98 190,84,220,112 utf8:4f>
<chemtraverse_end>
```

上例中 `->` 是 Markdown 把末尾 `-` 与闭合符号 `>` 连续显示的结果；实际字段值是 `-`。

记录定义：

- `meta width height molecule_id_utf8`：画布和开放 UTF-8 分子 ID。
- `component/component_end`：一个连通分量；允许多组分。
- `atom x y role element charge isotope aromatic explicit_h no_implicit radical chiral`：一个原子对象。
- `down orientation display chemical direction stereo`：树边及随后首次出现的邻接原子。
- `up`：返回父原子。
- `link previous_atom_ref ...`：闭环边，只能引用已经出现的原子。
- `text atom_ref center_x center_y bbox utf8:text`：可见文字对象；`Ph`、`R1`、`A`、任意 Unicode
  字符串都按原文保存，不在 OCR 阶段展开化学含义。
- `stereo bond_ref atom_ref0 atom_ref1`：双键立体参考原子。

`orientation` 为 `F/R`，保留 ChemScene 中有方向的 begin/end 语义。`display` 与 `chemical`
分别保存图片显示键型和化学键型。双键、三键始终各是一条 bond 事件，不会展开成 4 个或
6 个端点。

## 唯一遍历

1. 连通分量根节点按图像坐标和原子/文字属性唯一排序。
2. 邻居按相对角度、距离、邻居属性和键属性唯一排序。
3. 第一次到达原子使用 `down + atom`；已经出现的原子只能使用 `link`。
4. Text 和双键 stereo 引用放在遍历后的 postamble。
5. `Scene -> Traverse -> Scene -> Traverse` 必须逐字节一致。

这里不存在键长上限、最近邻建边或“长键候选删除”。一根键是否存在由一个完整 bond 事件
表示，而不是由端点距离启发式决定。

## 四面体手性

RDKit 的 `CHI_TETRAHEDRAL_CW/CCW` 依赖原子邻居排列，不能直接放入会重排原子的遍历序列；
单独使用 CIP `R/S` 也不无损，因为部分环系只有相对四面体手性，InChI 可以区分，但 RDKit
不会给每个中心分配 CIP。

ChemTraverse 因此使用：

- `-`：未指定四面体手性。
- `TCW/TCCW`：相对于邻居 `(y, x)` 图像坐标排序的四面体奇偶性。

解析器根据完整邻接图把该稳定奇偶性转换回 ChemScene 的原子编号奇偶性。该过程不调用
sanitize，因此即使模型预测出无效价态，序列仍可保存、解析和复现，便于错误诊断。

## 源目标门禁

训练目标进入 ChemTraverse 前必须依次通过：

1. ChemMol + Render + Layers 构建 ChemScene。
2. ChemScene 编译结果与 ChemMol 的完整 InChIKey 一致。
3. ChemScene 可无损写入并读回 SDF。
4. ChemTraverse 两次序列化逐字节一致。
5. ChemTraverse 编译 SDF 的完整 InChIKey 与 ChemMol 一致。

源坐标碰撞、布局立体冲突和源 Scene 无法无损写 SDF 属于 `source_rejected`，必须确定性替换，
不能计作协议失败，也不能混入训练。

## 当前验收

`style_train_100k` 两个独立分片共检查 10,000 张图片：

| 分片 | 原始样本 | 源拒绝 | 合格样本 | 协议失败 |
|---|---:|---:|---:|---:|
| train-000000 | 5,000 | 16 | 4,984 | 0 |
| train-000001 | 5,000 | 40 | 4,960 | 0 |
| 合计 | 10,000 | 56 | 9,944 | 0 |

另对 v0.10.21 的 1,100 个 MR-TADF 实际预测 ChemScene 做 canonical-only 审计，包含无效价态
预测：`1,100/1,100` 可表达并逐字节规范化，未调用 SDF sanitize。

冻结 50K 训练清单的 token 长度范围为 261 到 2,874，P99 为 2,144；当前 3,072 上下文覆盖
全部样本，无需截断。

验收结果：

- `outputs/evaluations/chemtraverse-v01-style-train-shard000-validation.json`
- `outputs/evaluations/chemtraverse-v01-style-train-shard001-validation.json`
- `outputs/evaluations/chemtraverse-v01-v01021-mr-tadf-primary-canonical-audit.json`

运行：

```bash
.venv-train/bin/python -m pytest tests/data_sdf/test_chemtraverse_v0_1.py -q

.venv-train/bin/python -m Data.sdf.validate_chemtraverse_v0_1 \
  Data/sdf/chemvector_joint_v1/style_train_100k/shards/train-000000.tar \
  --output outputs/evaluations/chemtraverse-v01-style-train-shard000-validation.json

.venv-train/bin/python -m Data.sdf.audit_chemtraverse_v0_1_predictions \
  outputs/evaluations/mr-tadf-v1-v01021-invalid-only-ocr-fallback-live/primary_candidates \
  --output outputs/evaluations/chemtraverse-v01-v01021-mr-tadf-primary-canonical-audit.json
```
