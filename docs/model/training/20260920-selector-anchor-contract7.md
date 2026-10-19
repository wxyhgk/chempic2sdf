# 7图锚点监督核验：未发现热图锚点混用，文字中心另有契约差异

## 结论和下一步

本轮已完成旧原子头来源、旧/新目标适配路径、保存热图的损失方向核验。
**未发现将文字框中心当成原子热图锚点的直接接线错误；现有热图损失对10个漏点的输出梯度方向也正确。**
这排除了所检查范围内的两种简单解释，不是修好了模型，也不是已找到完整训练退步的原因。

- 同7张图、226个原子，经两条真实适配路径和显式ID对齐后，锚点、热图、角色和文字掩码完全一致。
- 73个可见文字目标的中心定义确实不同：旧路径取化学锚点，当前路径取墨迹框中心。
  这个变化属于文字几何辅助监督，未改变热图目标；不能直接归因为10个漏峰的原因。
- 10/10漏点所在真值格的热图损失导数为负，10/10最强邻格的导数为正：独立修改输出时，
  梯度下降会抬高正确格、压低竞争格。竞争格均不是其他原子的真值正格。
- 四份checkpoint的40个原子头张量完全相同；当前initial的原子头仍沿用joint30k阶段。
  后续阶段总样本计数增加，不能当成原子头又学了相应数量的样本。

下一步建议只检查 **`atom.heatmap` 最后一个1×1卷积的共享参数方向**，不立即训练：
该层为128个权重加1个偏置，共129参数。固定其余网络，检查同一个参数梯度方向是否改善
10个漏点的真值格/邻格分差，并记录对其余216个标注原子的影响。
现有快照没有该层完整输入特征，执行时需要新的、单独限额的特征观察，不能声称本轮已经验证。
可沿用固定7图、CPU最多120秒、每图最多一次前向、零优化器步、不扫学习率的诊断范围；
若需要额外图像或更新参数，应另列范围和预算。

只有共享参数方向检查支持继续，才讨论极小热图层试训及旧图端点/可达键回归保护。
固定7图都是已观察错误样本，不是泛化验收集；即使它们改善，也不能自动替换默认模型。
继续冻结OCR、文字角色和文字几何头，不把已发现的中心定义差异混入这项定位检查。
此前失败的selector配方与exact-one目标不因本轮结论而重新获得训练准入。

## 范围与资源

沿用上一轮dense观察的固定7张旧训练/回放图、10个raw缺失目标。
每张图使用全部原子标注构造热图，避免将未列入10个漏点的其他原子误当负例。
没有新增图片、开发集逐图读取、封存测试读取、身份核验或数据发布。

真实样本审计工作进程耗时 **1.716秒**，硬上限120秒，CPU双线程，GPU为0、重试为0。
仅以CPU `weights_only=True, mmap=True` 读取四份checkpoint的已存张量，
不构造真实模型或优化器，不运行真实图片前向，不计算模型参数梯度、不更新参数。
求导对象仅为上一轮已保存logits的独立副本。
真实图像像素没有加载；适配器所需图像字段由白色占位图提供，不用于推理。
单元测试中的合成输入及小模型检查不属于真实图片审计前向。

## 两种中心必须分开理解

| 监督字段 | 旧AtomBondScene路径 | 当前Curriculum路径 | 固定7图结果 |
| --- | --- | --- | --- |
| 原子锚点 | render中的化学坐标量化后，经scene传递 | layers中化学锚点量化 | 226/226一致 |
| 热图 | 由原子锚点生成，半径1格 | 由原子锚点生成，半径1格 | 7/7整图逐值一致 |
| 文字中心 | layers中的center量化，即此处化学锚点 | 墨迹框两角先量化，再取框中心 | 73/73可见目标不同 |
| 文字掩码/原子角色 | 可见文字与隐式节点区分 | 可见文字与隐式节点区分 | 全部一致 |

代码证据：

- `Data/sdf/build_chemvision_layers.py`：旧renderer的layer center和render坐标均来自RDKit绘图坐标。
- `Data/sdf/chemscene_v0_1.py`、`chemscene_v0_2.py`：化学锚点与文字几何分别传递。
- `chemvector/training/graph/scenes/train_atom_bond_scene.py`：旧适配器分别填充`atom_anchors`与`atom_text_center`。
- `chemvector/training/curriculum/data.py`：当前适配器的锚点取`visual["center"]`，文字中心取bbox中心。
- `chemvector/components/atoms/objects/spatial_atom_object.py`：热图只由`atom_anchors`构造，文字中心另有辅助损失。

本轮桥接把**同一批当前标注**转成旧schema，再运行真实旧适配器；
明确逆转旧路径的空间排序，按源atom ID逐个比较。
这不是读取、重放或证明原始历史训练数据完全无误；旧训练分片未打开，也未重新渲染。
如果未来解冻原子头共享层，文字辅助损失仍可能间接影响热图，不能因直接锚点一致就忽略这一风险。

## 参数来源及历史代码边界

核对链：

1. `chemvector-v0971-joint-proposal-objects-full50k/samples-030000.pt`：samples_seen=30000，原子学习率2e-5、损失权重1。
2. `chemvector-v0972-bond-presence-calibration-10k/best.pt`：samples_seen=35000。
3. `chemvector-v0101-set-first-style10k/selected.pt`：samples_seen=45000。
4. `chemvector-selected-matching-trial-20260915/initial.pt`：本次bundle优化步数0，记录的来源指向第3项。

四份原子头40个张量逐值相等，排序名称与张量字节摘要均为：
`cf4ac0d399cad6653e34bc0c251878f8dfdb56d45097ca57eb0a4e6325c272be`。
这只证明原子头参数链，不单凭它宣称所有上游视觉参数的历史都相同。

第3项记录的git版本为`1141b40d621f4707b4aaad550d6d71a91eafeb0f`。
该版本中8个相关renderer/scene/collator/loss/joint-training源文件与当前对应文件字节一致。
但第1项joint checkpoint没有记录自己的代码版本；不能据后续版本推断每次历史运行的源码都已被完整证明。
每份checkpoint和每个代码文件的完整SHA见机器报告。

## 梯度解释与独立复核

现有focal热图损失把目标值为1的格子当正例；邻域高斯值小于1的格子仍是降权负例，
不是要求模型在邻格输出高斯值的回归目标。
10个竞争格目标值分别属于约0.011109和0.00012341两类，无其他真值正格。

独立NumPy复核不调用本轮producer或PyTorch求导：
从全部源锚点重新量化、生成热图，再对保存logits计算sigmoid、完整图loss和解析导数。
对正例，令p为sigmoid、N为整图正格数，导数为：

`[2p(1-p)^2 log(p) - (1-p)^3] / N`

对目标y<1的负例，导数为：

`(1-y)^4 [-2p^2(1-p) log(1-p) + p^3] / N`

20个被核对的格子均未进入概率clamp饱和区。
解析导数与float32自动求导最大绝对差为`6.468001922249611e-09`；
7图完整loss最大绝对差为`1.1265816501904169e-07`。
独立复核同时通过73个输入SHA、7个逐图SHA、8个历史源文件字节比较及226/73/10计数。
复核没有重新反序列化checkpoint；原子参数逐值比较由主审计执行，复核核验其输入hash和链记录。

**输出logit的正确导数不等于共享参数更新的正确效果。**
卷积权重由所有格子共享，真值格、邻格和全图负格的参数梯度可能相互抵消；
本轮不计算这部分，不宣称已救回任何候选或可达键。
同样，7图样式被主动选中，不能从中估计全数据的字体/版式偏移比例。

## 工程交付与回归

新增`tools/selector_anchor_contract.py`和`tools/audit_selector_anchor_contract.py`，
合成测试`tests/chemvector/test_selector_anchor_contract.py`共21项。
覆盖ID对齐、锚点/文字中心分离、移动文字框不改变热图、协议拒绝条件、
其他真值邻格保护、必须使用全图标注以及只对独立logits求导。

相关11个测试文件合并回归：**155 passed，16 subtests passed / 5.09秒**。
这是工程回归，不是全仓测试或模型准确率验收。

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv-train/bin/python -m pytest -q \
  tests/chemvector/test_selector_anchor_contract.py \
  tests/chemvector/test_curriculum_data.py \
  tests/chemvector/test_selector_localization_contract.py \
  tests/chemvector/test_selector_coordinate_contract.py \
  tests/chemvector/test_selector_dense_observation.py \
  tests/chemvector/test_selector_dense_review.py \
  tests/chemvector/test_spatial_atom_object.py \
  tests/chemvector/test_vertex.py \
  tests/data_sdf/test_chemscene_v0_1.py \
  tests/data_sdf/test_chemscene_v0_2.py \
  tests/data_sdf/test_chemsvg_sequence_v4.py
```

审计入口：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv-train/bin/python -m tools.audit_selector_anchor_contract`。
结果已经存在，入口拒绝覆盖，不应删除旧记录以强制重跑。

结果目录：`outputs/audits/chemvector-selector-anchor-contract7-20260920/`。
报告SHA：`7029a3fafc813f36343155da12641cdaba8dd5e7fae049538f4b975a33ce174f`。
上一轮dense报告SHA：`7ba69b5f75241fced837611a9dbdef236a1855fdbbd290f01ff50154f78ec1ff`。

生产模型代码、checkpoint、标签、阈值、匹配半径与默认initial均未改变。
本轮没有训练或新增准确率；此前另21个“raw有候选但selector没保留”的缺口也尚未解决。
