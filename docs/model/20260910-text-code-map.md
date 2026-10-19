# 2026-09-10 文字相关代码职责与整理边界

状态：代码边界梳理；受控 train-only 检查按下表分工推进。生产模型、默认推理入口、
旧生成器和冻结实验源文件不在本轮迁移范围内。本文不声称训练标签已修复或已完成普查。

## 目标与允许依赖

当前先解决“明确训练样本的可见文字标签是否值得复核”，不同时推进模型训练或架构替换。
输入审核、渲染诊断、人工证据与标签修订必须分开：诊断差异只能产生候选，不能自动
成为“原 PNG 实际画了什么”的答案。

| 层 | 文件/入口 | 职责与允许依赖 | 不应依赖或执行 |
|---|---|---|---|
| 训练输入 | `Data/sdf/train_visible_text_inputs.py` | 处理显式训练输入、验证身份与 split、读取受控样本；由 CLI 调用 | 历史 fresh-dev selection、模型 checkpoint、自动遍历所有数据目录 |
| 通用渲染诊断 | `Data/sdf/rendered_text_evidence.py` | 对显式提供的化学图与渲染参数提取诊断文字；供训练检查使用 | 某次冻结审计目录、训练/评测路径选择、把替代渲染当作原 PNG 真值 |
| 检查编排与报告 | `Data/sdf/audit_train_visible_text.py` | 组合输入与诊断、绑定样本与文件哈希、输出待核验候选与覆盖范围 | 自动改标签、覆盖 shard、启动训练、读取未授权 split |
| 候选统计与呈现 | `Data/sdf/train_visible_text_report.py`、`train_visible_text_contact_sheet.py` | 纯候选规则、分母统计、原图派生crop、确定性图版 | 模型加载、训练目录扫描、自动填写视觉审核结论 |
| 标签修订纯函数 | `Data/sdf/visible_text_revision.py` | 接收同图有效证据与明确修订请求，验证旧值、返回版本化 layers | 文件系统扫描、模型推理、从隐式氢推定字符、接受替代后端诊断作修订证据 |
| 冻结历史审计 | `chemvector/evaluation/audits/ocr/text_entry_*`、`text_failure_*`、相关 `audit_*` | 重放各自协议、处理已绑定产物、诊断历史模型失败 | 成为 train-only 检查的数据入口或通用渲染库 |
| 生产推理 | `chemvector/inference/infer_direct_atom_pair_bond.py`、`chemvector/direct_scene_inference.py` | 加载既定组件、候选筛选、连接预测、OCR、scene/SDF 编译 | 导入训练标签检查或历史文字审计以决定线上输出 |

这些边界是模块职责约定，具体命令参数、输入路径和执行结果以对应实现、测试和
独立运行报告为准。新检查工具没有自动获得训练数据写入或模型训练授权。

## 已有组件应当复用到哪里

- 原始 layers 的文字修订继续走 `visible_text_revision`，不要在 CLI 中再写一套
  `CH2 → C` 替换规则。未经过同图证据复核的项只进入候选报告。
- 原有生成器可提供几何恢复等底层能力，但不能把其化学属性拼接结果再当成独立
  的可见文字证据；新版诊断与旧标签来自同一路径时，应明确证据并不独立。
- `audit_text_failure_label_probe.py:visible_svg_labels` 是历史审计内的诊断实现。
  新的通用诊断独立落到 `rendered_text_evidence.py`，不通过导入该历史审计取得能力，
  也不为了复用而迁移其冻结源文件。
- `text_entry_predictions.py:selected_images` 只服务冻结比较的样本协议，不是通用
  训练读取器；`text_entry_protocol.load_protocol` 也不是 train-only 数据门禁。
- 生产已有 `direct_scene_inference` 承担 direct 模型加载、scene tensor 转换、OCR
  写入和可选抑制。训练标签确认不需要导入这些 GPU/模型组件。

## 生产中实际存在、但本轮暂缓的整理

以下优先级针对将来的纯重构，不是当前训练方向问题的优先级。

### P1：直接引用已有 SDF 输出底层函数（小）

`infer_direct_atom_pair_bond.py` 通过 `benchmark_mr_tadf` 导入
`write_prediction_sdf`，而该 benchmark 已从 `chemvector.diagnostic_sdf` 导入同名实现。
可改为直接引用底层模块，减少生产到 benchmark 的依赖；不必修改 benchmark 本身。
这不意味着当前推理读取了该 benchmark 数据集。

### P2：将 OCR 加载与裁剪从 evaluator 提取（中）

生产入口从 `evaluate_direct_atom_ocr` 导入 `load_role`、`load_bbox`、`load_ctc`、
`crop_tensor`，从而带入 evaluator 顶层的训练与评测依赖。
将来可放入独立 runtime helper，并由 evaluator 保留历史 re-export。需验证 checkpoint
schema、严格加载行为、冻结/eval 状态、4px context、裁图尺寸和归一化保持一致。
不能借此改变 OCR 数值精度、阈值或生产预测行为。

### P3：通用 scene 编译与 oracle 评测分层（大）

入口从 `shadow_chemscene_v02_oracle` 获取 scene builder、芳香一致性、渲染和空 SDF
输出 helper。该大模块还包含 oracle matching、缓存训练、MILP 和 benchmark 依赖。
生产传入空的 `placeholder_scene`，不是评测真值 scene；名字不能作为发生真值泄漏的证据。
将来可提取通用编译层，把 oracle 特有控制留在评测层，需要 scene/SDF 与退化路径的
充分回归验证，不应与当前标签确认一同改动。

三项暂缓的共同原因：不是本次标签核验的必要条件；现有文件被旧实验记录了精确
源哈希；大范围重构会把训练标签问题与推理行为回归混在一起。本轮只新增分层工具
和职责文档，不以“代码整理”名义改变生产入口或历史报告。

## 冻结实验与新源码的边界

`text_entry_protocol.source_files()` 在冻结时记录所有 `chemvector/**/*.py` 和顶层
`Data/sdf/*.py`；`load_protocol()` 校验协议已记录的文件。因此修改这些旧文件，即使
只是移动函数，也会导致原协议源校验失败。新增未被旧协议记录的文件不会自动使
旧校验失败，但不能据此宣称新工具已被旧协议覆盖。

不重写旧协议哈希来迁就整理，不覆盖历史产物。未来真正实施上述重构，需要明确
新源版本与旧协议固定源的重放办法；若有新实验，独立冻结它的输入、源文件和产物。

## 受控检查的解释边界

训练检查的结果应区分：扫描范围、检查成功数、诊断差异候选数、已经原图复核数、
允许修订数。未扫描样本、诊断失败和未经人工确认的差异均不能算作标签已验证。
即使出现 `C/CH2` 差异，也不能据此宣称训练集污染率或识别器错误率。

本轮输出待核验材料之后，再凭同图证据决定独立版本的修订；不自动训练，不拿
fresh-dev、sealed test 或外部 benchmark 样本补足训练确认数量。

本轮实际运行与源图复核结果见 `training/20260910-train-visible-carbon-review.md`。
