# Public OCSR Benchmark v1 数据交付规范

本目录用于交付公开、可复现、与其他 OCSR 模型可横向比较的测试集。第一阶段只
处理 MolGrapher 发布的 `USPTO-30K`，官方来源：

```text
https://huggingface.co/datasets/ds4sd/USPTO-30K
```

MR-TADF 不再承担主测试集角色，只保留为项目内部复杂结构 challenge。

## 一、第一阶段范围

USPTO-30K 包含三个互不替代的子集：

| 子集 | 官方说明 | 当前用途 |
|---|---|---|
| `uspto_10k` | 10,000 个不含缩写的结构 | 主公开 benchmark |
| `uspto_10k_l` | 10,000 个超过 70 个原子的结构 | 大分子公开 benchmark |
| `uspto_10k_abb` | 10,000 个包含 superatom/缩写的结构 | 先完整收集，等待缩写 SDF 协议 |

第一版 headline 指标只使用 `uspto_10k` 和 `uspto_10k_l`。`uspto_10k_abb` 不得
丢弃，也不得擅自展开缩写；在模型输出协议能够无损表达 superatom、alias 或
S-group 前，只报告数据审计结果，不报告不公平的完整 SDF 准确率。

## 二、交付目录

数据人员应生成以下目录，不要把文件放到本目录根部：

```text
public_ocsr_v1/
├── README.md
├── sources/
│   ├── README.md
│   └── uspto_30k.json
├── raw/
│   ├── README.md
│   └── uspto_30k/
│       └── <官方原始下载内容>
├── prepared/
│   ├── README.md
│   ├── uspto_10k/
│   │   ├── images/
│   │   └── ground_truth/
│   ├── uspto_10k_l/
│   │   ├── images/
│   │   └── ground_truth/
│   └── uspto_10k_abb/
│       ├── images/
│       ├── ground_truth/
│       └── source_structures/
├── manifests/
│   ├── README.md
│   ├── uspto_10k.csv
│   ├── uspto_10k_l.csv
│   └── uspto_10k_abb.csv
└── reports/
    ├── README.md
    ├── summary.json
    ├── validation.json
    ├── rejected.csv
    ├── duplicates.csv
    └── previews/
        └── index.html
```

## 三、绝对禁止

- 不得重新渲染官方图片。
- 不得 resize、裁边、旋转、去噪、二值化、锐化或改背景。
- 不得根据结构重新生成“更干净”的测试图片。
- 不得删除难例、超大分子、带电结构或不认识的元素来提高通过率。
- 不得因为重复结构而改变官方 benchmark 样本数量。
- 不得把任何公开 benchmark 样本加入训练、微调、threshold 选择或 checkpoint
  选择。
- 不得用规范 SMILES 代替 ground-truth SDF 作为正式交付标签。
- 不得擅自把 `Ph/Me/tBu/R1` 等 superatom 展开成另一份化学结构。
- 不得覆盖或修改 `raw/` 下任何文件。

## 四、原始数据保留

`raw/uspto_30k/` 必须完整保留官方数据。下载完成后，在
`sources/uspto_30k.json` 记录：

- Hugging Face dataset ID；
- 实际下载时间；
- dataset revision/commit SHA；
- 下载工具和版本；
- 官方 README 或 dataset card 的 SHA256；
- 每个原始文件的相对路径、字节数和 SHA256；
- dataset card 声明的许可证；
- 实际字段名和三个子集的映射关系。

不能只记录 `main` 或 `latest`，必须固定到具体 revision。

## 五、图片处理

正式图片必须是官方原始图片的逐字节复制：

```text
raw official image bytes == prepared image bytes
```

每张图片至少检查：

- PIL 可以完整读取并执行 `load()`；
- 宽高大于 0；
- 不是全白、全黑或空文件；
- SHA256 与 raw 对应文件完全相同；
- 样本 ID 与官方结构标签一一对应。

不同尺寸、灰度、RGB、RGBA、JPEG 或 PNG 都允许，评测器负责输入适配。数据人员
不得统一格式或尺寸。

## 六、真值结构与 SDF

### 6.1 官方 Molfile/SDF 存在时

官方 Molfile/SDF 是唯一结构真值。`prepared/*/ground_truth/<sample_id>.sdf` 可以
进行容器级规范化，但必须满足：

- 原子元素、形式电荷、同位素、自由基和显式氢不变；
- 键端点、键级、芳香性和立体化学不变；
- 分子片段数量不变；
- 原子数和键数不变；
- 原始二维坐标尽量保留；
- V2000 无法无损表示时使用 V3000；
- 写出后重新读取并通过结构往返验证。

原始结构文件仍必须保留在 `raw/`。不要只留下转换后的 SDF。

### 6.2 官方只有字符串结构时

如果某个子集没有 Molfile/SDF，只有 SMILES、CXSMILES 或其他字符串标签，不得
静默转换后宣称为官方 SDF。应在 `reports/validation.json` 标记：

```json
{
  "status": "blocked_missing_official_structure_file",
  "subset": "...",
  "available_structure_field": "实际字段名"
}
```

并暂停该子集，先让模型人员确认比较协议。

### 6.3 superatom/缩写

`uspto_10k_abb` 必须同时保留官方结构文件和转换后的候选 SDF。需要统计官方结构
使用的 alias、S-group、R-group、dummy atom、atom label 和缩写字段。任何转换
发生信息丢失时，该样本不得标记为 `ready`，也不得自行展开缩写修复。

## 七、Manifest

三个 CSV 使用 [manifests/README.md](manifests/README.md) 中定义的同一字段顺序。
路径全部相对于 `public_ocsr_v1/`，不得写机器绝对路径。

每个官方记录必须有且只有一行。无法读取或无法转换的样本仍需出现在 manifest，
并通过 `status` 和 `notes` 说明，不得直接删除。

## 八、重复与交叉污染审计

公开 benchmark 保持官方样本数量，不做去重，但必须生成 `duplicates.csv`，报告：

- 子集内部重复图片；
- 子集内部重复结构；
- 三个子集之间的重复结构；
- 与 `webdataset_full_v0` 训练分子的结构重叠数量；
- 与 ChemVision pilot 训练分子的结构重叠数量；
- 与 MR-TADF 的结构重叠数量。

结构重叠审计可以内部使用 InChIKey 或规范图 hash，但正式准确率仍按 SDF 图比较。
发现训练重叠时不要删除官方测试样本，必须在报告中单独给出：

```text
all_samples
train_novel_only
```

两个统计口径。

## 九、验收标准

`reports/validation.json` 只有满足以下条件才能写入 `"status": "passed"`：

1. 官方 dataset revision 已固定并记录。
2. 三个子集的官方记录数量与 dataset card 一致。
3. 所有 raw 文件都有 SHA256。
4. 所有 prepared 图片与 raw 图片 hash 完全相同。
5. 图片与结构标签一一对应，无 ID 串位。
6. `uspto_10k` 和 `uspto_10k_l` 的 ready 样本 SDF 往返通过率为 100%。
7. 所有失败样本进入 `rejected.csv`，没有静默丢弃。
8. `uspto_10k_abb` 的缩写表示类型和信息损失已完整统计。
9. `duplicates.csv` 已完成所有规定的重叠审计。
10. 随机抽取至少 500 个样本生成图片/真值 SDF 对照预览。
11. 使用相同 revision 重复准备 1,000 个样本时，输出 hash 完全一致。

## 十、模型人员交付入口

数据人员完成后，只需要通知模型人员以下文件：

```text
Data/benchmarks/public_ocsr_v1/reports/validation.json
Data/benchmarks/public_ocsr_v1/reports/summary.json
Data/benchmarks/public_ocsr_v1/manifests/uspto_10k.csv
Data/benchmarks/public_ocsr_v1/manifests/uspto_10k_l.csv
Data/benchmarks/public_ocsr_v1/manifests/uspto_10k_abb.csv
Data/benchmarks/public_ocsr_v1/reports/previews/index.html
```

模型人员会在验收通过后编写统一 public benchmark 加载器，不要求数据人员修改
任何模型训练或推理代码。

## 十一、后续公开数据集

完成 USPTO-30K 后，再单独评估 CLEF、UOB 和 JPO 的官方来源、许可和标签质量。
它们不得在同一个下载任务中混入 USPTO-30K，也不得使用第三方重新打包版本冒充
官方原始数据。
