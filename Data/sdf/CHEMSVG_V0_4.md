# ChemSVG v0.4 协议

ChemSVG v0.4 是图片到 SDF 路线的第一版坐标图元协议。它不让模型直接输出
`begin_atom/end_atom` 形式的抽象 graph，而是输出图片中的视觉顶点和 stroke。
确定性编译器根据 stroke 的起止坐标建立连接，再构造 RDKit Mol 和 SDF。

现有 75GB WebDataset 不需要重建。加载器读取已经保存的 ChemMol JSON 和
Render JSON，运行时动态生成 ChemSVG 标签。

## 文档结构

```text
<chemsvg v=0.4 units=q1023>
<v 100 500 C 0 0 0 0 0 0 CHI_UNSPECIFIED>
<v 300 500 O 0 0 0 0 0 0 CHI_UNSPECIFIED>
<s 100 500 300 500 SINGLE SINGLE NONE STEREONONE>
<chemsvg_end>
```

顶点记录：

```text
<v x y element charge isotope aromatic explicit_h no_implicit radical chiral>
```

stroke 记录：

```text
<s begin_x begin_y end_x end_y display_type bond_type direction stereo>
```

带双键立体参考坐标时追加：

```text
<s ... stereo ref0_x ref0_y ref1_x ref1_y>
```

## 与 v0.3 的根本区别

v0.3 的键记录是：

```text
<b 2 17 SINGLE ...>
```

v0.4 的连接记录是：

```text
<s 315 228 507 411 SINGLE SINGLE NONE STEREONONE>
```

解析器只允许 stroke 端点引用此前出现的视觉顶点坐标。它不会根据原子间距离
枚举候选边，也没有最大键长阈值。一根很长的可见线仍然只编译为一条 stroke。

`display_type` 描述图片中的线型，`bond_type` 保存无损化学键型。芳香键在图片
中可能显示为 `SINGLE/DOUBLE` 交替，也可能显示为 `AROMATIC`，因此第一版同时
保留两者。后续版本可以在视觉编译器成熟后进一步减少隐藏化学字段。

## 坐标和不可表示图片

坐标范围为 0 到 1023。顶点按 `(y, x)` 排序，但 stroke 不引用排序产生的原子
编号，只引用坐标。

如果两个不同原子在 Render JSON 中具有完全相同的量化坐标，图片本身无法
唯一恢复两个视觉顶点。加载器会跳过这类样本，不会添加隐藏 atom ID 来绕过
视觉歧义。

## 确定性编译

ChemSVG 到 SDF 的编译步骤：

1. 按顶点记录创建 RDKit Atom。
2. 建立 `坐标 -> Atom` 映射。
3. 每条 stroke 直接在两个端点对应的 Atom 之间创建 Bond。
4. 应用芳香性、键型、手性、双键立体和显示方向。
5. 执行 RDKit sanitize。
6. 写出 SDF 后重新读取，并比较 InChIKey。

## 训练

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.4 \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --validation-shards Data/sdf/webdataset_full_v0/shards/validation-000000.tar \
  --batch-size 4 \
  --gradient-accumulation 1 \
  --workers 0 \
  --epochs 1 \
  --output-dir outputs/chemvector-v4
```

首个训练 shard 的验收结果：

- 排除 13 张已知 E/Z 描绘冲突图片后，共检查 4,979 张。
- 4,956 张完成 ChemSVG -> RDKit Mol -> InChIKey 无损往返。
- 23 张因为两个不同原子具有完全重合的视觉坐标而被跳过。
- 平均长度 702 token，P99 为 1,784，最大为 2,274。
- 词表大小为 2,153，当前 3,072 token 上下文可以完整覆盖。

详细结果见 `chemsvg_v0_4_shard_validation.json`，重合坐标 key 清单见
`chemsvg_v0_4_coordinate_collision_keys.txt`。

## 测试

```bash
.venv-train/bin/python -m unittest -v \
  Data.sdf.test_chemsvg_sequence_v4 \
  Data.sdf.test_chemsvg_tokenizer_v4 \
  chemvector.test_generation_v4
```
