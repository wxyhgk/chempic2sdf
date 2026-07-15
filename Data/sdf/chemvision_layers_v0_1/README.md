# ChemVision Layers v0.1 数据准备规范

本目录是 ChemPic2SDF v0.8 的语义分层数据交付目录。数据人员应先完成
`pilot_10k/`，通过验收后才能生成 `full/`。本任务不修改、不覆盖现有数据：

```text
Data/sdf/webdataset_full_v0/
```

现有 18 万个分子、72 万张图片继续作为结构来源和切分真值。

## 一、要解决的问题

当前图片中的黑色像素同时可能表示：

- 原子文字，例如 `N`、`B`、`Cl`、`NH2`、`Ph`；
- 单键、双键、三键、芳香键、楔形键或虚线键；
- 无文字隐式碳原子的线条交点。

大号 `N` 可能在像素上表现为三条独立笔画，不能用连通域数量判断它是几个
对象，也不能让模型把其中的斜线当成化学键。本数据集必须从同一次语义 SVG
渲染中，准确分离文字和键线。

## 二、交付目录

```text
chemvision_layers_v0_1/
├── README.md
├── schemas/
│   └── chemvision-layers-0.1.schema.json
├── examples/
│   └── example.layers.json
├── manifests/
│   └── README.md
├── pilot_10k/
│   └── README.md
├── full/
│   └── README.md
└── reports/
    └── README.md
```

生成数据后，`pilot_10k/` 和 `full/` 内部统一使用：

```text
pilot_10k/
├── shards/
├── previews/
├── samples.csv
├── molecules.csv
├── fonts.csv
├── rejected.csv
├── summary.json
└── validation.json
```

## 三、第一批规模

第一批固定生成 10,000 张图片：

- 2,500 个分子；
- 每个分子生成 `original`、`rdkit`、`coordgen`、`stress` 四种布局；
- train：2,250 个分子，9,000 张图片；
- validation：125 个分子，500 张图片；
- internal_test：125 个分子，500 张图片；
- 必须沿用 `webdataset_full_v0` 原有 molecule/scaffold split；
- 不允许把同一个分子移动到其他 split；
- MR-TADF 数据不得进入训练数据。

选择分子时进行分层覆盖。以下为图片中可见文字实例的最低数量，不要求类别
互斥：

- `B`：不少于 2,000 个实例；
- `N`：不少于 2,000 个实例；
- `F`：不少于 1,000 个实例；
- `Cl/Br/I` 合计：不少于 1,000 个实例；
- `P/S/Si/Se` 合计：不少于 1,000 个实例；
- 无文字隐式 C hard-negative anchor：不少于 10,000 个。

隐式 C hard negative 必须覆盖：二键端点、三叉交点、四叉交点、芳环顶点、
双键附近、长键端点和靠近文字的碳中心。

## 四、第一批渲染范围

第一批只验证几何、字号和文字/键线分层，不加入困难图像退化：

- 画布：`768 x 768`；
- 输出：RGB PNG，白色不透明背景；
- 颜色：仅黑白；
- 不加模糊；
- 不加扫描噪声；
- 不加透明背景；
- 不做 JPEG 压缩；
- 键线宽度：`1.4–2.6 px`；
- 旋转角度：`-12–12` 度；
- 保留现有四种布局和 stress 操作；
- RDKit 版本固定为 `2026.03.3`，并写入元数据。

10,000 张图片的字号数量必须精确满足：

| 字号区间 | 图片数 | 比例 |
|---|---:|---:|
| 8–12 px | 2,500 | 25% |
| 13–19 px | 4,500 | 45% |
| 20–28 px | 2,500 | 25% |
| 29–36 px | 500 | 5% |

至少使用 5 个许可允许训练和再分发的字体家族。字体文件、字体家族、许可证、
SHA256 和 split 必须记录到 `fonts.csv`：

- train 使用 3 个字体家族；
- validation 使用 1 个 train 未见字体家族；
- internal_test 使用 1 个 train/validation 未见字体家族；
- 同一字体家族的 regular、bold、italic、condensed 版本必须属于同一 split。

## 五、唯一允许的生成方式

每张图片必须从一份带语义 class/id 的 SVG 产生。推荐流程：

```text
规范化 RDKit Mol
  -> 使用确定的布局和渲染参数生成一次语义 SVG
  -> 按 atom-{id} / bond-{id} 等语义信息拆分 SVG 图元
  -> 使用同一个 rasterizer、画布、scale 和抗锯齿设置栅格化
  -> combined/text/bond/semantic
```

禁止：

- 从已有 PNG 使用 OCR、连通域、形态学或阈值算法猜文字层；
- 分别重新绘制 combined、text 和 bond，导致坐标或抗锯齿不一致；
- 隐藏文字后重新绘制 bond-only，使键线延伸到原文字内部；
- 根据原子距离、最近邻或最大键长猜测真实键；
- 将多段文字笔画错误拆成多个文字实例；
- 遇到异常时静默修改结构或忽略失败样本。

`bond.png` 必须保留 combined 中原本因文字标签而产生的键端截断和留白。
真实键来自 ChemMol/SDF 的 bond id，不来自像素几何猜测。

## 六、每张图片的文件

WebDataset 中每个 sample key 包含：

```text
000000001.png
000000001.text.png
000000001.bond.png
000000001.semantic.png
000000001.layers.json
000000001.chemmol.json
000000001.render.json
000000001.meta.json
```

其中：

- `.png`：由同一语义 SVG 栅格化的完整黑白图；
- `.text.png`：8-bit 灰度文字 alpha，背景 0，文字 1–255；
- `.bond.png`：8-bit 灰度键线 alpha，背景 0，键线 1–255；
- `.semantic.png`：8-bit 单通道索引图；
- `.layers.json`：原子文字实例、文字框、键和渲染来源；
- `.chemmol.json`：复用现有 ChemMol 标签；
- `.render.json`：与本次新渲染对应的 Render JSON；
- `.meta.json`：来源、split、sample key、版本和 hash。

`semantic.png` 的像素值固定为：

```text
0 = background
1 = bond
2 = text
3 = text/bond overlap
```

不得使用调色板颜色代替上述索引值。

## 七、文字标签规则

- `layers.json.atoms[].text` 保存图片中实际显示的 UTF-8 字符串；
- 不进行元素周期表校正或化学含义纠正；
- `Ph`、`R1`、`A`、`X` 等未来都应作为普通字符串保留；
- 隐式碳使用 `text=""`、`visible_text=false`、`bbox=null`；
- 显式 `C` 使用 `text="C"`、`visible_text=true`；
- `NH2`、`[N+]` 等保存可读顺序，不把每个 glyph 当成独立原子；
- bbox 是该 atom_id 所有文字矢量路径的并集；
- 即使 `N` 的三笔不连通，也只能产生一个 atom 文字实例和一个 bbox；
- bbox 顺序固定为 `[x_min, y_min, x_max, y_max]`，使用图片像素坐标。

JSON 必须通过：

```text
schemas/chemvision-layers-0.1.schema.json
```

## 八、分片和命名

- 每个 tar 建议 2,500–5,000 个图片样本；
- tar 内文件名只使用 sample key，不建立子目录；
- sample key 在整个数据集内唯一；
- `source_sample_key` 指向 `webdataset_full_v0` 的原始 sample key；
- molecule_id、split 和 variant 必须与现有数据一致；
- tar 创建顺序和 sample 顺序必须由固定 seed 决定；
- 相同输入、代码版本和 seed 必须生成相同 hash。

## 九、必须生成的清单和报告

具体 CSV 字段见 [`manifests/README.md`](manifests/README.md)。交付至少包含：

- `samples.csv`；
- `molecules.csv`；
- `fonts.csv`；
- `rejected.csv`，没有失败也必须有表头；
- `summary.json`；
- `validation.json`；
- `previews/index.html`，随机展示至少 500 个样本的四层左右对照。

任何 SVG 解析失败、文字路径无法归属、bbox 越界、结构不一致或 mask 对不齐的
样本都进入 `rejected.csv`。不得自动换成另一个分子来掩盖错误；第一批先报告
真实失败率。

## 十、验收标准

`pilot_10k` 必须全部满足：

1. 2,500 个分子、10,000 个图片 sample，数量准确。
2. 每个 sample 的 8 个成员均存在且可读取。
3. 所有图片均为 `768 x 768`。
4. `text/bond/semantic` 与 combined 使用相同坐标和栅格化参数。
5. `text ∪ bond` 与 combined 前景 IoU 不低于 `0.999`。
6. 未归类前景面积不超过 combined 前景的 `0.1%`。
7. visible text 的 alpha 非空，bbox 包含其全部非零像素。
8. 隐式 C 必须 100% 满足空字符串、无 bbox、无文字实例。
9. 大号 `N/B` 即使包含多个像素连通域，也只有一个 atom 文字实例。
10. bond id、begin_atom 和 end_atom 与 ChemMol 100% 一致。
11. `display_type` 与 Render JSON 一致，`chemical_type` 与 ChemMol 一致。
12. ChemMol 重建 RDKit Mol 后 sanitize 成功，InChIKey 往返 100% 一致。
13. molecule/scaffold 跨 split 重叠为 0。
14. train、validation、internal_test 的字体家族无交叉。
15. 字号、元素、线宽、布局、stress 类型和字体分布均写入报告。
16. 随机 1,000 个样本重复生成后，所有成员 SHA256 完全一致。
17. 人工检查 `previews/index.html` 中至少 500 个四层对照样本。

`validation.json` 顶层必须包含：

```json
{
  "schema_version": "chemvision-validation-0.1",
  "status": "passed",
  "samples": 10000,
  "molecules": 2500,
  "rejected": 0,
  "foreground_iou_min": 0.999,
  "scaffold_overlap": 0,
  "font_family_overlap": 0
}
```

如果存在 rejected，`status` 应为 `failed` 或 `review_required`，不能填写
`passed`。

## 十一、全量数据何时生成

在模型人员确认以下三个实验结果前，不要启动 72 万张全量生成：

1. 真值 bond layer 能显著减少大字母造成的伪边；
2. 多尺度 OCR 在最差字号桶明显优于固定 20px ROI；
3. raw image + soft semantic probability 优于硬删除文字区域。

通过后，`full/` 才按 72 万张现有图片规模生成。全量仍保持现有 18 万分子和
scaffold split，不增加 MR-TADF 分子。
