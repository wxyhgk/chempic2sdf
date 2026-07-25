# ChemScene v0.2 联合对象协议

ChemScene v0.2 是 v0.9.42 开始使用的联合 `AtomObject + BondObject` 目标协议。
它保留 v0.1 已验证有效的完整键对象，但删除独立 `Junction`、独立 `TextObject`
以及推理末端的最近邻 OCR 附着。

```text
ChemScene v0.2
  ├─ AtomObject
  │    ├─ chemical state
  │    ├─ IMPLICIT / TEXT role
  │    ├─ chemical anchor
  │    ├─ optional text / center / bbox
  │    └─ incident bond ports
  └─ BondObject
       ├─ begin/end AtomObject 引用
       ├─ 两个可见端口
       ├─ display / chemical bond type
       └─ direction / stereo
```

## 为什么升级

USPTO clean 1k 的 v0.9.41 分解结果为：

- 纯骨架 exact：72.7%；
- 原子类型骨架 exact：29.3%；
- 键级骨架 exact：59.1%。

这说明完整 BondObject 已经明显强于原子身份层。旧路线先生成匿名 Junction，再将
OCR 文字按距离附着到节点，会把文字框偏移、原子 anchor、键端口和化学身份拆成
多个互相补救的阶段。任何一个阶段错一处，都可能产生错误价态或把文字挂到相邻节点。

v0.2 将一个原子的全部视觉和化学状态收回同一个 AtomObject。模型以后必须直接预测
“这个对象是什么、在哪里、文字区域在哪里、哪些键端口属于它”，而不是先找匿名点再猜
文字归属。

## 核心不变量

1. AtomObject ID 和 BondObject ID 必须从 0 开始连续排列。
2. 每个 BondObject 固定引用两个不同 AtomObject。
3. 每个 BondObject 端口必须且只能由对应 AtomObject 的一个 incident port 拥有。
4. AtomObject port 与 BondObject port 的坐标必须逐项完全一致。
5. `ANCHOR` 端口必须等于 AtomObject 的 chemical anchor。
6. `TEXT_BBOX_RAY` 只能属于 `TEXT` AtomObject。
7. 同一对 AtomObject 不允许重复 BondObject。
8. 双键、三键、芳香键和楔形键各自仍是一个 BondObject；内部墨线不产生额外原子。
9. 不使用键长阈值删除连接，不在编译阶段做最近邻文字附着。

## 文字端口

文字原子的 chemical anchor 通常位于字形内部，但可见键在文字框边界停止。因此 v0.2
同时保存两类坐标：

- `anchor`：化学原子位置，用于 SDF 二维坐标和 AtomObject 匹配；
- `port`：图片中键真正接触该对象的位置，用于 BondObject incidence 训练。

Oracle 构建器从 anchor 朝相邻原子发射射线，与文字 bbox 求交得到
`TEXT_BBOX_RAY`。隐式原子的可见端口直接等于 anchor，来源为 `ANCHOR`。

## 示例

```json
{
  "schema_version": "chemscene-0.2",
  "molecule_id": "chembl:CHEMBL123",
  "units": "q1023",
  "canvas": {"width": 768, "height": 768},
  "atom_objects": [
    {
      "id": 0,
      "anchor": [240, 512],
      "role": "TEXT",
      "chemical": {
        "element": "N",
        "formal_charge": 0,
        "isotope": 0,
        "aromatic": false,
        "explicit_h_count": 0,
        "no_implicit": false,
        "radical_electrons": 0,
        "canonical_chiral_tag": "CHI_UNSPECIFIED"
      },
      "text": {
        "value": "N",
        "center": [240, 512],
        "bbox": [220, 488, 260, 536]
      },
      "incident_ports": [
        {
          "bond_id": 0,
          "endpoint": 0,
          "coordinate": [260, 512],
          "source": "TEXT_BBOX_RAY"
        }
      ]
    }
  ],
  "bond_objects": [
    {
      "id": 0,
      "atom_refs": [0, 1],
      "ports": [[260, 512], [600, 512]],
      "display_type": "SINGLE",
      "chemical_type": "SINGLE",
      "direction": "NONE",
      "stereo": "STEREONONE",
      "bbox": [258, 506, 602, 518]
    }
  ]
}
```

## 确定性编译

当前编译路径为：

```text
ChemScene v0.2
    -> 严格 Atom/Bond/port 所有权检查
    -> 无损投影到 ChemScene v0.1
    -> ChemSVG v0.4
    -> RDKit Mol
    -> SDF 写入与重新读取
    -> InChIKey 比较
```

v0.2 目前先复用已经验收的 v0.1 化学编译器，避免同时修改模型目标和 SDF 语义。
后续模型可以直接输出 v0.2，但稳定 SDF 编译行为保持不变。

## 当前验收

ChemVision validation 500：

- 498 张无损完成 Scene -> SDF -> InChIKey 往返；
- 2 张量化坐标冲突被显式隔离；
- 未知失败 0。

ChemVision train 首 1000：

- 996 张无损往返；
- 2 张源 E/Z 描绘冲突被显式隔离；
- 2 张量化坐标冲突被显式隔离；
- 未知失败 0。

结果见 `chemscene_v0_2_validation.json` 和
`chemscene_v0_2_train_1000_validation.json`。

## 运行

```bash
.venv-train/bin/python -m Data.sdf.validate_chemscene_v0_2 \
  Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar \
  --limit 500 \
  --output Data/sdf/chemscene_v0_2_validation.json
```

现有 WebDataset 不需要重建，也不需要重新渲染 PNG。加载器可以从已保存的
`.chemmol.json`、`.render.json` 和 `.layers.json` 动态构建 v0.2 标签。

