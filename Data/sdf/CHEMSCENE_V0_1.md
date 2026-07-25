# ChemScene v0.1 对象协议

ChemScene v0.1 是 v0.8 Bond-First 路线的对象级中间表示。它不是把图片中的每一段
黑色像素都拆成点和线，而是把化学结构图视为由少量可组合对象构成的场景：

```text
ChemScene
  ├─ Junction       真正的化学顶点
  ├─ TextObject     与顶点绑定的开放字符串
  └─ BondObject     一整根单键、双键、三键、芳香键或楔形键
```

一根双键仍然是一个 `BondObject`，不是两根独立直线和四个线端点。键线之间发生
视觉交叉也不会自动产生 `Junction`。这直接针对“平行双键被拆成多个原子”和
“交叉线被错误建联”的失败模式。

## 示例

```json
{
  "schema_version": "chemscene-0.1",
  "molecule_id": "chembl:CHEMBL123",
  "units": "q1023",
  "canvas": {"width": 768, "height": 768},
  "junctions": [
    {
      "coordinate": [180, 500],
      "role": "IMPLICIT",
      "atom": {
        "element": "C",
        "formal_charge": 0,
        "isotope": 0,
        "aromatic": false,
        "explicit_h_count": 0,
        "no_implicit": false,
        "radical_electrons": 0,
        "canonical_chiral_tag": "CHI_UNSPECIFIED"
      }
    }
  ],
  "text_objects": [],
  "bond_objects": [
    {
      "endpoints": [[180, 500], [700, 500]],
      "display_type": "DOUBLE",
      "chemical_type": "DOUBLE",
      "direction": "NONE",
      "stereo": "STEREONONE",
      "bbox": [175, 490, 705, 510]
    }
  ]
}
```

## 对象语义

`Junction` 只允许两种角色：

- `IMPLICIT`：通常是不显示字符的碳顶点。
- `TEXT`：由一个 `TextObject` 占据的原子位置。

`TextObject.text` 是开放字符串，允许 `N`、`Cl`、`Ph`、`R1`、`A` 等任意非空
内容。OCR 阶段只负责转写，不在这里做缩写展开或化学理解。文字框和几何连接相互
独立；更改字符串不会改变 `BondObject` 的端点。

`BondObject.endpoints` 直接引用两个 Junction 坐标，不引用隐藏 atom ID，也不通过
最近邻或键长阈值寻找连接。一个跨越整幅图片的可见长键仍然是一根键。

`display_type` 描述图片中画出的对象，`chemical_type` 描述输出 SDF 的键型。芳香键
的视觉表示可能是交替单双键，也可能是芳香线型，因此暂时保留两个字段。

## 确定性编译

```text
ChemScene v0.1
    -> 严格对象和端点检查
    -> ChemSVG v0.4
    -> RDKit Mol
    -> SDF
    -> 重新读取 SDF
    -> InChIKey 比较
```

SDF 默认保留 Scene 的二维坐标。少数源图在手性中心附近存在重叠方向，V2000 无法
在这些原坐标上无损编码楔形键时，编译器只为 SDF 重新生成规范二维坐标；ChemScene
仍保留模型看到的原始视觉坐标。

以下输入会被显式隔离：

- 两个 Junction 量化后坐标完全相同；
- 源绘图几何与结构标签的 E/Z 立体信息冲突；
- BondObject 端点不是已声明 Junction；
- 同一对 Junction 出现多个 BondObject；
- 立体参考点与键端点不相邻。

## 当前验收

ChemVision pilot validation：

- 总计 500 张；
- 498 张完成 Scene -> ChemSVG -> RDKit -> SDF 无损往返；
- 2 张同坐标冲突被显式隔离；
- 未知失败为 0。

首个训练分片：

- 总计 3,000 张；
- 2,983 张完成无损往返；
- 8 张源 E/Z 描绘冲突；
- 9 张同坐标冲突；
- 未知失败为 0。

验收结果分别见 `chemscene_v0_1_validation.json` 和
`chemscene_v0_1_train_shard_validation.json`。

## 运行

```bash
.venv-train/bin/python -m Data.sdf.validate_chemscene_v0_1 \
  Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar \
  --output Data/sdf/chemscene_v0_1_validation.json
```

实现位于 `chemscene_v0_1.py`。现有 75GB WebDataset 不需要重建，训练加载器可从
`.chemmol.json`、`.render.json` 和 `.layers.json` 动态构造 ChemScene 标签。
