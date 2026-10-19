# 2026-09-19 基础原子与键环境扩量

用户要求先让多个 agent 造数据。本轮仅进行自研 ChemVector 的 CPU 合成数据准备，
不启动模型训练、GPU 推理、外部身份查询或封存评估访问。

**结果：生成 20,904 张，通过全量产物审计；保守内部质量提案保留 20,289 张不同图片／
3,468 个结构。16 个冲突身份的 96 图整体隔离，519 张同身份重复图列入排除提案。
原始文件全部保留，仍未训练准入。**

## 固定方案

三个 agent 独立完成原子/基本键、官能团环境、立体环境目录，主 agent 合并去重与绘图。
只新增 `Data/sdf/chemvector_curriculum_v0_2/` 和对应新测试，不修改旧 renderer、训练或历史实验源码。

| 项目 | 数量 |
| --- | ---: |
| 原子/基本键目录 | 1,722 个结构 |
| 官能团环境目录 | 1,245 个结构 |
| 立体环境目录 | 594 个结构 |
| 跨目录 canonical isomeric 去重后 | 3,484 个结构 |
| 固定绘图任务 | 20,904 = 3,484 × 6 |

六种样式为 reference/font_small/font_large/bond_short/line_thin/serif，使用既有
`stereo-aware-v1`。原图 768×768，只有干净绘图，不加噪声；字符框、图/键身份、图像和 SDF
保持既有同源合同。不同图样不视为独立分子。

合并覆盖审查发现 N–N/N–O/S–S 与 C=N/C=S 初始不足，固定计划之前在原子/键模板中补充
89 个结构，保留原先全部 1,633 个 ID/SMILES；没有借助开发集逐样本身份选择新分子。

## 已完成的预检

- 12 个固定 ID 前缀结构 × 六样式 smoke：**72/72 合格，0 拒绝**，72 个不同 PNG，
  无跨身份相同 PNG。耗时 15.484 秒，CPU，GPU=0。
- 全量生成使用 16 CPU workers，每分子批次 120 秒上限，外层总预算 1,800 秒，不自动重试。
- 来源源码及字体哈希写入计划，在任务结束复核；同一分子的六样式是日志提交单位，
  超时未返回批次为 pending，不能伪装已完成分数。
- 三路新目录、集成、独立审计、旧渲染/立体排版/CPU运行保护的合并回归
  **138 passed / 60.06 秒**。此前单独的 38/96/21 项结果与此重叠，不再相加。
- 独立 agent 对 72 张 smoke 全量核验：diagnostic loader/collator 各 72/72，
  558 原子目标、522 键目标、空间同格碰撞 0；training loader/collator 全部拒绝。
- 另一 agent 对 3,484 个源结构做全量 SDF 往返与索引审计：28,850 原子、27,197 键，
  化学身份、无向端点与已指定立体 0 丢失；512 分子的 544 个指定四面体中心、
  82 个 E/Z（各 41）均保留。175 分子仍有 181 个未指定四面体中心，不宣称立体全指定。
  80 条键的序列化方向翻转符合原无向合同；172 个未指定双键 NONE→ANY 是未知态表示差异。

## 外层监控中断及恢复

原生成 launcher 在执行期间退出，工具报告 exit 143；原因尚未定位，不能声称正常退出。
独立 session 内的原 CPU worker 继续运行，已提交图像未丢失，也未重启 worker 或重试任务。
原 `runtime.json` 未生成，不能补造为正常成功凭据。

新增独立 `tools/watch_curriculum_cpu_job.py`，由临时 user service 托管，核对精确 PID、
进程启动 ticks、命令、CPU 环境和独立进程组后，恢复**原 1,800 秒预算**。
截止以原 worker 启动时刻计算并再保守减去 2 秒，恢复时不重新起算预算。
凭据为 `supervision-recovery.json` 与最终 `runtime-recovery.json`；无法取回原子进程退出码时
保持 null，必须另外独立核验完成产物，不能以“进程消失”替代成功检查。

## 全量结果

固定任务已全部生成并保存图库，`summary.json` 已完成：

| 指标 | 数量 |
| --- | ---: |
| 完整处理的固定成图任务 | 20,904 / 20,904 |
| 通过原有渲染合同 | 20,904 |
| 渲染质量拒绝 | 0 |
| 不同 PNG 字节 | 20,316 |
| 全部六样式成图的分子 | 3,484 / 3,484 |
| 已重核样本文件 SHA | 229,944 |
| 生成源码/字体绑定 | 33 |

生成进程结束时恢复监控记录距原 worker 启动 **1,276.441 秒**，在原预算内；
`runtime-recovery.json` 为 `worker_exited_exit_status_unavailable`，明确 `worker_exit_code=null`，
没有伪造原 launcher 的正常完成码。完整产物随后通过下面的独立审计。

实际成图含化学 SINGLE 112,962、DOUBLE 10,770、TRIPLE 3,780、AROMATIC 35,670 条实例；
这些是按图片重复计算的标签实例，不是不同分子数或模型可达正例数。
实际实楔、虚楔各 1,632 条；双键方向笔画另计。化学芳香与画法单双线没有混为同一标签。

数据：`outputs/datasets/curriculum-v0.2-expanded-20260919/`。
运行记录：`outputs/datasets/curriculum-v0.2-expanded-20260919-runtime/`。

## 成图过程中发现的跨身份像素冲突

在已提交样本中，出现未指定 E/Z 的烯烃和已指定构型具有完全相同 PNG 的情况：
例如 `CC=CCC` 与 `C/C=C/CC`，以及 `CC=Cc1ccccc1` 与 `C/C=C/c1ccccc1`。
六种样式中若分子没有文字，换字号/字族还会产生同一身份的重复 PNG。

前者是不能直接训练的标签歧义，后者是不能当作额外不同图片的重复。原数据和固定任务不改，
全量独立审计后按所有冲突身份整体隔离，再对其余同身份 PNG 做确定性去重提案。
不从冲突两边挑一个“赢家”，不自动从绘图几何改写源分子的 E/Z；后续需要单独改进生成语义。
隔离/保留提案仍不等于训练准入，完整分母与冲突证据全部保留。

完整生成清单记录 **27 组跨身份相同 PNG，涉及 16 个化学身份、96 条样本**；
独立解码 RGB 与 PNG 字节核验得到相同冲突范围。

## 独立产物审计

`tools/audit_curriculum_expansion_v02.py` 在生成进程退出后单独执行，耗时 **97.752 秒**：

- 20,904/20,904 图、229,944/229,944 样本文件 SHA、33/33 源码和字体绑定通过。
- SDF 严格立体身份、原子/键索引、graph/layers/text 合同、双层内部哈希、journal 与分母对账通过。
- 逐图合计 173,100 原子、163,182 键；化学计数与实际绘图方向分别复核。
- 不同解码 RGB 与不同 PNG 均为 20,316；未发现额外的同像素不同 PNG 编码冲突。
- 去立体、保同位素/电荷/键级的连接身份为 **3,165**，不是 scaffold 数或新划分。
- 审计状态为 `passed_artifact_integrity_only`，且 `blocking_cross_identity_pixel_conflicts=true`：
  产物完整性通过**不代表**原始批次可以直接训练。

报告：`outputs/audits/curriculum-v02-expanded-independent-20260919.json`。
全量 target tensor collation 和预测端点可达性没有在此次审计中执行；只有 72 图 smoke 做过 collator。

## 内部质量交接

质量交接工具 `tools/prepare_curriculum_v02_quality_handoff.py` 绑定通过的审计与完整源 manifest，
重新核对 PNG/RGB 冲突并输出新的保留/隔离/重复三分区提案，不删除、改写原样本。
相关新增独立文件测试 **19 passed / 10.44 秒**，包括真实微型生成→审计→交接。
完整交接结果已成功落盘，状态 `proposed_internal_quality_handoff_only`：

| 互斥分区 | 图片数 |
| --- | ---: |
| 建议保留的不同 PNG／RGB | **20,289** |
| 冲突身份整体隔离 | 96 |
| 其余同身份 PNG 重复排除提案 | 519 |
| 合计，等于原始成图分母 | **20,904** |

保留候选来自 **3,468 个结构**；冲突隔离涉及 **16 个身份**，全部六样式均计入。
剩余 173 个同身份重复组固定按 sample_key 字典序保留一个，519 条为额外重复。
去重后 PNG 与 decoded RGB 的不同图片数均为 20,289，没有将重复编码冒充不同训练图片。

交接文件：`outputs/audits/curriculum-v02-quality-handoff-20260919.json`，绑定独立审计、
源 manifest、审计实现及依赖哈希，并重新核验全部 PNG/RGB；未删除或改写样本。

## 浏览和下一步

- [原始隔离图库](../../../outputs/datasets/curriculum-v0.2-expanded-20260919/index.html)：包含所有原始样本，
  **包括**有冲突和重复的图，不能直接当成筛选后的训练集。
- [首张参考联系页](../../../outputs/datasets/curriculum-v0.2-expanded-20260919/reference-contact-01.png)。
- [独立审计](../../../outputs/audits/curriculum-v02-expanded-independent-20260919.json)。
- [保守质量交接提案](../../../outputs/audits/curriculum-v02-quality-handoff-20260919.json)。

下一步先对保留候选做全量目标适配/必要的人审、身份准入，再检查真实预测候选中的键可达性。
新身份不继承前批已经消耗的单次核验授权。没有创建 train/val/test 划分，也未因此追加训练预算。

## 不变的边界

全部生成物仅 quarantine，`training_authorized=false`、`external_overlap_status=unchecked`。
化学合法/通过渲染不是识别准确率、外部去重通过或训练有效性证据。部分合成结构只经过
模板/价态检查，没有实验稳定性保证。配位/氢键/金属/波浪键未加入。

训练前还需要独立准入、处理像素重复/冲突、目标适配、真实预测端点可达性，
以及能够学习相应键类型的新训练方案；不能原样沿用只训练 presence 的小试训来声称学会键型。
