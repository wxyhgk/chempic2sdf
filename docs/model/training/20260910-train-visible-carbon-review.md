# 2026-09-10 Train-only 可见碳标签确认

## 本轮工作

用户要求多个 subagent 先整理代码，再推进。三个分工分别梳理数据生成、推理依赖和
审计/修订代码；实际整理新增通用 renderer 诊断、独立 train 输入层，并加固未冻结的
修订纯函数。旧生成器、生产推理、冻结历史审计源文件和数据没有修改。

本轮不是重新训练，也不自动将诊断结果转换为标签修订。

## 冻结范围

- 输入固定为 `Data/sdf/chemvector_joint_v1/style_train_100k`：25,000 分子、100,000 图。
  它是 `STYLE_TRAIN_SHARDS` 与 OCR 的 style_train domain 使用的数据来源；不因此
  声称当前某份 checkpoint 实际对本轮每个样本做过梯度更新。
- 校验 commitment、自校验 SHA、samples.csv 和选中 train tar 的 SHA/大小；禁止其他
  root、路径穿越、符号链接和非 train 成员。只读取入选样本的 PNG/layers/chemmol/render。
- 在读取实际标签前，按 `sha256("visible-carbon-review-v1:" + molecule_id)` 排序，固定
  取 250 分子，各保留 original/rdkit/coordgen/stress 四布局，共 1,000 图。
- 检查全部可见中性、无同位素/自由基的碳，包含原标签 C 的对照，不仅筛选 CH2。
- noFreetype SVG 文本节点只提供 diagnostic：当节点恰为 `["C"]`，而 layer text 非 C，
  标为待复核。不同文字后端不是原始栅格重放，不能自动升级为同次渲染权威记录。
- 无 sealed test、MR-TADF、USPTO 内容读取；没有修订 shard、模型加载或训练。

## 结果

| 项目 | 数量 |
|---|---:|
| 入选分子 / 图片 | 250 / 1,000 |
| layers 标记为可见文字的对象 | 8,144 |
| 可见中性碳对象 | 80 |
| 原标签 C | 12 |
| 原标签 CH2、诊断节点为 C | 68 |
| 疑似不一致涉及图片 / 分子 | 22 / 12 |
| 未解决的诊断对象 | 0 |
| 已应用修订 / 训练次数 | 0 / 0 |

68 个诊断差异按布局为：original 19、rdkit 21、coordgen 1、stress 27。
其中两个分子分别贡献 24 个对象，因此对象数不是独立分子证据数。
不能将 68/80 解读成全训练集文字污染率；该分母是可见中性碳子集。

## 原图视觉复核

按照固定规则“每个疑似分子取 object key 最小的一个对象”，选择 12 个代表。
图版同时显示无标注的原图派生放大 crop 和带 bbox 的上下文，避免边框遮挡字符。
主 agent 与独立 subagent 分别查看图版，12 个目标均观察为单个 C，未见目标 H2。
第4个样例邻近 H 字符，但与目标 C 之间有明确连接和空间间隔，不是 C 的后缀。

这属于 **AI 视觉复核，不是人类审核签署**。已另存逐对象观察和证据哈希，没有把
主诊断报告的其余 56 个疑似对象标为已审核，也没有触发修订接口。

独立验证了全部80个原始PNG副本与训练manifest SHA一致，重新从源图在内存裁剪，
clean crop和context与保存图片逐像素一致。图版与selection绑定哈希；视觉结果不是
直接照抄替代渲染器输出。

这将此前的“fresh-dev 中存在同类问题”推进为：**在实际训练数据来源中，也找到
原图可见 C 与文字标签 CH2 不一致的样例**。仍不能断言训练集完整影响范围或当前
模型错误中有多少由此造成。

## 代码整理结果

- `rendered_text_evidence.py`：新数据工具不再 import 某次 audit 来获取渲染诊断。
  返回 `same_source_raster=false`、`revision_authorized=false`，不把节点顺序拼成化学式。
- `train_visible_text_inputs.py`：manifest/选择/tar成员身份独立于渲染、OCR与展示。
- `train_visible_text_report.py`：纯候选判定、分母统计与样例输出；CLI只编排各层。
- `visible_text_revision.py`：拒绝缺失sample identity、缺失旧文字、bool bbox坐标，
  防止已有provenance被覆盖；仍保持纯函数，只返回新版本文档，不写数据。
- `train_visible_text_contact_sheet.py`：从冻结证据包生成确定性代表图版，不分配审核结论。

旧审计保留历史helper实现是有意的：它们被源哈希冻结，不能为清目录而破坏重放。
生产OCR loader、SDF writer、scene compiler 的历史耦合列入后续项，本轮不迁移。

## 下一步

优先进入**新版生成器的可见文字契约修复**：让同次渲染的可见字符成为文字监督来源，
而不是在发现bbox后从隐式氢回填字符串。必须保留图片真正显示 CH2 的合法情况，
不能全局替换；化学字段和连接几何保持独立。

修复应先用小型合成回归样例验证，包括共线C、真正画出的CH2、带电/同位素碳及
无文字隐式碳。随后才决定版本化重建或受控修订，不覆盖旧shard，不顺手开训。

## 产物与复现

```bash
.venv-train/bin/python -m Data.sdf.audit_train_visible_text --output-dir outputs/evaluations/train-visible-carbon-review-20260910 --molecules 250
.venv-train/bin/python -m Data.sdf.train_visible_text_contact_sheet --review-dir outputs/evaluations/train-visible-carbon-review-20260910 --output-dir outputs/evaluations/train-visible-carbon-review-examples-20260910
```

这两个命令拒绝覆盖非空输出目录。`visual-review.json` 是运行后另行记录的 AI 视觉
观察，不是以上命令自动生成的“正确答案”。

| 产物 | SHA256 |
|---|---|
| review/protocol.json | `71ac628adb4269c8f660753ebe9a7d5fc7c447f23e55f7bb219feadf0b7dfab4` |
| review/report.json | `b205c682e13eae59a1818a3fb7df20580286732ebec7ae57c552d5b0d6e71943` |
| review/records.jsonl | `99b1f432d4ed65dc3d772834d3929590f010d404dba3fd66ffe851d7963b8e57` |
| examples/selection.json | `9975621fab2103442704315d638883d6e39a302c44a289aa95c14c242d9689df` |
| examples/contact-sheet.png | `309fa0d30294a7d5f15dab9449565da77eb399b30aea7aacce209eed5cd8534e` |

`review/`和`examples/`分别指上方命令中的两个输出目录。

## 最终验证与剩余边界

- 新工具、修订接口、图版生成和原文字审计/推理回归合计 **167 passed**。
- 1,000图/250分子/四布局各250图、80对象、68诊断差异均经独立重算。
- 原比较、失败归因、标签探针绑定的旧源文件哈希保持不变；新train证据包全部
  产物及源文件哈希复核通过，80组原PNG/crop/context像素绑定检查通过。
- 代码复核发现一个需显式确认的序号前提：RDKit按chemmol原子列表顺序建图，
  而诊断按atom_id关联。已重新读取全部选中训练样本，确认1,000图的chemmol ID均为
  从0开始的连续列表，render/layer ID完整且唯一，故本次不存在编号错位解释。
  后续输入模块版本应将这一检查内置；本轮保留冻结代码并单独记录验证。
- 8,144是层标签标记数量，不是逐对象视觉普查；12个源图代表的AI视觉检查不等于
  人类审核批准，也不授权修改其余56个未看的候选。
