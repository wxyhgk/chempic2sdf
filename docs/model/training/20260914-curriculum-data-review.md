# 原子、基本键与官能团课程数据试点：独立审查

日期：2026-09-14。状态：已完成统一 builder 首轮独立审查及关键修订复查；产物仅限隔离预览。

本轮目标是制作可检查的隔离合成数据，不启动训练、不改生产流程，不替换被冻结的
生成器或实验。没有读取 sealed test 逐样本、历史训练图片或 benchmark 图片。

## 推荐复用的同源渲染链

优先在新的独立绘图 profile 中复用以下接口，保持旧文件原样：

- `Data/sdf/text_source_renderer/v2/renderer.py:prepare_molecule`：复制输入分子，检查
  原子和键身份、连接及属性；源 chemical bond 与绘图 drawing bond 分开记录。
- `Data/sdf/text_source_renderer/v2/svg_records.py:attributable_svg`：检查最终 SVG 的
  文字、键与辅助 junction 归属；未知或歧义路径拒绝，不能猜作新的化学键。
- `Data/sdf/text_source_renderer/v2/layers.py:compile_layers`：从同一份最终 SVG 得到
  PNG、逐字形 alpha bbox、实际字符、原子 anchor、键端点和分层 mask；记录 SVG/PNG
  哈希。要求前景可被已归属墨迹完全覆盖，文字与几何重叠、边界裁切等失败保留为拒绝。

文字来自 SVG 实际发出的 text 节点，不能从原子的隐式氢、电荷或化学式重新拼接为
“可见标签”。源分子属性与可见字符串是两套事实：图上只画 C 时允许结构仍含隐式氢；
没有文字的碳节点也必须保留在完整图中。字符按 SVG emission order 记录；化学规范化
或 OCR 转写顺序若另有约定，应新增派生字段，不覆盖原始记录。

显式 display override 必须同时影响最终图像和文字记录，记录来源及意图；不能只改标签。
默认不借课程数据引入任意化学别名或把官能团缩写当完整原子标签。

`text_source_layout` 的 v0.3 可以作为布局候选参考，但不能无条件当作数据质量已通过：
其既有矩阵 127/128 严格接受不代表训练可用，README 已记录大画布中文字到键的过大
空白。新 profile 应独立检查实际间距，而不是通过增大 padding 消除一切重叠。

v0.2 semantic mask 是 bitset：bond=1、text=2、auxiliary=4，bond/auxiliary 同像素为5。
后续训练消费者不能把它直接当旧 categorical map。辅助 junction 不增加化学原子或键数。

## 字号、键长和线宽的试点设计

现有 `TextSourceConfig` 可设字体参数、线宽、旋转和画布，但没有独立键长参数。
新 profile 若使用 RDKit `fixedBondLength`，不能把请求值直接当实测键长或已经验证的
像素上限。本轮 CCO 的请求值48对应实测 anchor 间距72px，与原始2D键长1.5的缩放
相符。固定 `scaleBondWidth=False` 后也应记录实际 SVG stroke-width。

建议直接渲染在最终模型输入分辨率，例如 768×768；若确有 resize/letterbox，必须用
相同变换处理图像、glyph mask、bbox 和 anchor，报告最终模型输入坐标系而非生成画布值。

每个接受样本至少记录：

- 请求 font size、各实际 SVG glyph 的 font size、最终 alpha>0 bbox 的高度/宽度；
  原子文字 union bbox 单列。上下标字形单列，不能用整段 union 高度代替主体字高。
- 全部化学键的 anchor-to-anchor 实测距离，及样本分布；不要把为文字截短后的 stroke
  长度当化学键长。无键单原子样本记空列表，不伪造零长度样本参与统计。
- SVG 实际线宽与渲染配置，文字/键或文字/辅助的重叠、裁切、最近文字到关联键墨迹间距。

首轮计划9个配对配置：reference、font_small/font_large、bond_short/bond_long、
line_thin/line_thick、serif、rotated，而不是立即全笛卡尔大规模扩增。
对同一分子保持2D构型及其他参数不变，验证一个旋钮确实只
改变预期尺度；实际字高与期望区间不符时记录失败或实测偏差，不换一个容易样本补数。
字族限于显式校验的 DejaVu Sans 与 DejaVu Serif，两者均记录实际字体文件哈希；
这仍不能宣称一般多字体或真实扫描图鲁棒性。已有 attribution 接口会将字体强制写成
DejaVu Sans，因此 serif profile 必须在 attribution 后更改最终 SVG，并同步 glyph 的
`svg_attributes`；之后不能再调用该接口将字体覆盖回来。改字族造成的排版重叠应按
原合同拒绝，不能因为已绑定文字身份就放宽墨迹质量检查。

严格拒绝、意外 error/timeout、接受应分别记录。失败不进入训练材料，也不从生成尝试
分母消失；radical 等已知超出 renderer 合同的例子应明确为负控制或未覆盖，不静默换例。

## 隔离与重叠边界

全部产物保持 quarantine，`training_authorized=false`、`overlap_status=unchecked`；
不创建可被现有训练 loader 自动发现的正式 train shard，不借目录名暗示已完成发布。
手写或简单分子不等于无污染：乙醇、芳环、常见官能团很可能与历史数据同分子或近邻。
本轮不读取受限评测逐样本，因此不能声称已排除这些重叠。

试点内部可对新生成分子计算精确、非立体及 scaffold 标识并标记重复，防止同一分子的
不同渲染样式被当作独立验证样本；这只解决内部组织，不解决外部污染。不能用完整图像
哈希不重复代替分子身份去重。后续若要晋升训练集，需要另行授权的重叠检查或受信索引
查询、明确统计结果和发布清单，在此之前不接训练、不声称泛化收益。

## 官能团和完整图目标

官能团是课程采样标签及可选辅助监督，不是最终识别单位。保留原子 ID、完整原子属性、
键 ID、端点、类型和立体信息；输出不折叠为官能团超级节点，不要求模型先识别官能团
才能预测普通原子或键。每个官能团实例引用本分子的 atom/bond IDs，支持重叠、多实例，
标记命名/SMARTS规则版本及适用边界。保护被 SMARTS 命中的局部结构不意味着整图已正确。

这套课程可以检验低复杂度样本的文字、节点和连接监督是否一致，但合成接受率、标签
保真及编译成功率都不是模型准确率，也不能证明课程学习优于现有采样或某种架构。

## 发布前最小检查

1. 所有接受样本：图节点/边 ID 与同源 layers 相互一致，完整图编译及立体往返正确；
   字符串与同源 text 节点、alpha bbox、图像及哈希一致。
2. 显式与隐式碳、上下标、电荷、同位素、单双三键、芳香/Kekulé、楔线、盐的连接分量
   各有小测试；不支持的输入明确拒绝。完整图与 drawing 表示不能混用为训练目标。
3. 报告全部尝试、接受、拒绝及意外错误；所有 artifact 保持隔离授权状态，输入目录
   和旧冻结源码/工件不变。并发 worker 有有限超时，不让单个分子卡死整轮。
4. 检查最终模型输入像素尺度、真实字形高度、实际键长及线宽，而非只列配置参数。
5. 官能团引用无越界、符合固定规则，内部重复和外部重叠未检查状态如实保留。

## 统一 builder 首轮代码审查

已只读审查 `Data/sdf/chemvector_curriculum_v0_1/` 下的 `catalog.py`、`rendering.py`、
`build.py`、`gallery.py`，未修改这四个文件。确认以下实现方向符合本轮隔离试点：

- 跨课程目录按 canonical isomeric SMILES 合并相同结构，保留多阶段成员关系；没有
  建 train/validation/test 划分，没有读取旧语料。
- 从最终修改字体后的同一SVG编译图像、字形框和字符，Serif 同步 glyph 属性与字体
  文件哈希；完整化学图独立保存，不将官能团当超级节点。
- 输出目录 `exist_ok=False`，样本文件以 exclusive `xb` 写入；quarantine 与未授权
  状态在 catalog、layers、graph、meta、summary 保留。联系页的前景裁剪明确仅供人工
  预览，原始768px图不变。

只用手写合成分子进行了不落盘的微型核对：CCO、苯、带正电胺、碳同位素及E/Z烯烃
通过SDF身份往返；CCO的Sans和Serif同源渲染通过，声明24px字体对应18px主体字形墨迹
高度，两种字体均保持2px实际线宽，训练授权为false。

首轮向主 agent 报告的具体问题：

1. `F[C@H](Cl)Br` 的SDF化学和立体SMILES不变，但楔线序列化把第一条键端点从
   `(0,1)`反转为`(1,0)`。原 `sdf_bytes` 的有序端点检查将其误拒绝为bond index变化。
   正确检查应保留键ID、无向端点集合及类型，同时保留整图isomeric身份检查；不能忽略
   真实立体变化。
2. 首版进程池`map`没有单任务超时，且`shutdown(wait=True)`会继续等待卡住的native
   任务。建议每任务隔离、有限超时、终止卡住的子进程，不让一个绘图挂起整轮。
3. 首版将所有`ValueError`归入普通拒绝，包括源分子被改写、SDF身份变化、字体回退等
   不变量失败。应只对白名单绘图合同拒绝计rejected，其他异常单列error并标记构建不完整。
4. 实际键长可以大于`bond_length_cap`字段数值；请求参数与实际像素结果必须分开说明。

### 修订复查与最终现状

已只读确认：SDF键端点改为无向集合验证，isomeric身份检查保留；两种四面体构型和
E/Z烯烃的不落盘微型往返全部通过。sample ID新增安全字符校验。普通质量拒绝改为
显式白名单；其他ValueError重新抛出并中止，不写成功summary，不再冒充正常拒绝。

已读取本轮新生成目录 `outputs/datasets/curriculum-v0.1-pilot-20260914/summary.json`
核对结果：108条课程记录合并为101个唯一分子，计划909张，接受751张，拒绝158张。
其中57张字形之间重叠、101张文字/几何重叠；没有把它们换例补数。98个结构至少有一张
接受图，49个结构九个样式全部接受，3个结构所有样式均拒绝。阶段归属重叠，不能把
阶段图片数相加当总图片数。

各profile已报告实际墨迹尺寸、anchor键长及SVG线宽。参考键长请求48对应实测72px，
短/长请求32/64对应约48/96px；因此应继续用实测字段解释数据尺度。字形高度列表含
上下标和电荷符号，最小值不是主体元素字母高度。接受率也不是文字可读性或模型性能。

**仍未实现单任务超时**；本轮909次已完成，未观察到挂起不等于该风险已消失。扩量前
需补进程隔离超时与失败记录。当前不扩量、不训练、不声称外部无重叠；全部仍为
quarantine、external_overlap_status=unchecked、training_authorized=false。

本独立审查验证的是上述代码、aggregate摘要及少量手写微型用例；完整751套文件的
逐工件哈希与像素验收由主 agent 的验证记录负责，不能将本段当作独立全量验收证明。
