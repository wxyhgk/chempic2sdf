# ChemVector

ChemVector 是面向化学结构图片识别的轻量视觉到结构序列模型。当前主开发
路线是 ChemSVG v0.4：模型生成视觉顶点和 stroke，确定性编译器再生成 SDF。
项目同时保留 ChemMolSeq v0.2/v0.3 作为直接 graph 生成基线。

项目版本从 `v0.7` 重新开始，以后每次发布增加 `0.0.1`：`v0.7.1`、
`v0.7.2`、`v0.7.3`。根目录 `VERSION` 是当前开发版本的唯一来源。

### v0.7.1 模块边界

推理代码不再从 CLI 或 generation 模块反向导入工具函数：

```text
image_io.py             图片准备、PIL/Tensor 转换
detection.py            vertex 检测与锚点 OCR
stroke.py               点对监督与线段分类
parallel_v6.py          无 decoder 并行结构预测
repair_v4.py            墨线拓扑与化学图修复
sdf_io.py               SDF 写入及立体往返校验
evaluation_metrics.py   纯图比较、叠加图与汇总指标
infer_*.py              参数解析与结果落盘
evaluate_detected_v5.py 数据迭代与 pipeline 调度
```

核心模块不再依赖 `infer_*` CLI；`repair_v4` 不再依赖 generation；并行推理只调用
公开的 `detect_vertex_records`，不再引用 generation 内部私有函数。

## 当前基线

- 参数量：49,974,768
- 输入图片：768×768 RGB
- 视觉 token：24×24，共 576 个
- ChemMolSeq 词表：2,153
- decoder 上下文：3,072
- decoder：8 层、隐藏维度 512、8 个注意力头

全量数据中 ChemMolSeq 平均长度为 779 token，P99 为 2,036，最大为 2,613，
因此 3,072 的上下文可以覆盖全部第一版样本。

## 环境

训练环境使用系统中已经支持 RTX 5090 的 PyTorch，并单独安装数据依赖：

```bash
python -m venv --system-site-packages .venv-train
.venv-train/bin/pip install -r requirements-train.txt
```

## 冒烟训练

```bash
.venv-train/bin/python -m chemvector.train_smoke \
  --image-size 768 \
  --batch-size 4 \
  --steps 3
```

## 单卡训练

```bash
.venv-train/bin/python -m chemvector.train \
  --train-shards 'Data/sdf/webdataset_full_v0/shards/train-*.tar' \
  --validation-shards 'Data/sdf/webdataset_full_v0/shards/validation-*.tar' \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --image-size 768 \
  --epochs 1 \
  --output-dir outputs/chemvector-v0
```

第一轮正式训练建议从 100 个固定样本过拟合开始，再扩展到 1,000、10,000
和全量训练集。

`--visual-aux-weight` 启用 24×24 patch 密集监督。16 个通道分别表示 12 种
元素的原子位置和 `SINGLE/DOUBLE/TRIPLE/AROMATIC` 四种实际绘制键线。该任务
训练视觉 encoder 保留点线信息，但不使用距离规则创建候选边。

训练器也保留图片错配对比损失参数用于后续实验，但高权重 hinge 对比损失在
100 样本实验中不稳定，不属于下面已经验证的训练课程。

## 已验证的 100 样本课程

单纯 teacher-forcing 会忽略图片。实测正确图片和打乱图片的 loss 几乎完全
相同，模型自回归时会固定生成训练集中的另一个分子。当前通过以下三阶段解决：

1. 视觉接地：50% 前缀遮蔽，`visual_aux_weight=5.0`，学习率 `3e-4`。
2. 前缀恢复：10% 前缀遮蔽，`visual_aux_weight=1.0`，学习率 `3e-5`。
3. 自回归精调：不遮蔽前缀，`visual_aux_weight=1.0`，学习率 `1e-5`。

100 样本最终 checkpoint：

```text
outputs/chemvector-grounded-final-100/last.pt
```

最终实测结果：

- 100 张图 teacher-forcing token accuracy：100%。
- 100 张图 exact sequence accuracy：100%。
- 前 5 个分子共 20 张四布局图片：20/20 自回归完成。
- 20/20 通过 RDKit sanitize，20/20 SMILES/InChIKey 与真值一致。
- 错配其他分子图片后，序列 loss 增加到正确图片的 11.34 倍。
- 错配图片后，视觉 patch loss 增加到正确图片的 91.03 倍。

详细评估结果位于：

```text
outputs/chemvector-grounded-final-100-eval/teacher_forcing_metrics.json
outputs/chemvector-grounded-final-100-eval-20/generated/generation_metrics.json
```

这组结果只证明训练、视觉接地、自回归和 SDF 往返链路正确，不代表未见分子
的泛化准确率。下一步应将相同课程扩大到固定 1,000 样本，再使用独立 scaffold
validation 检查泛化。

## ChemMolSeq v0.3

开发中的 v0.3 使用图片空间原子顺序，去除显式 record ID 和渲染专用键字段，
并把坐标移动到化学图之后。现有数据无需重建：

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.3 \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --no-shuffle --workers 0 \
  --batch-size 4 --gradient-accumulation 1 \
  --train-samples-per-epoch 1000 \
  --epochs 10 \
  --token-corruption-probability 0.5 \
  --visual-aux-weight 5.0 \
  --output-dir outputs/chemvector-v3-grounded-1k
```

第二阶段从第 10 个 epoch 继续到第 20 个 epoch：

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.3 \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --no-shuffle --workers 0 \
  --batch-size 4 --gradient-accumulation 1 \
  --train-samples-per-epoch 1000 --epochs 20 \
  --token-corruption-probability 0.1 --visual-aux-weight 1.0 \
  --resume outputs/chemvector-v3-grounded-1k/last.pt \
  --resume-learning-rate 3e-5 \
  --output-dir outputs/chemvector-v3-finetune-1k
```

第三阶段从第 20 个 epoch 继续到第 25 个 epoch：

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.3 \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --no-shuffle --workers 0 \
  --batch-size 4 --gradient-accumulation 1 \
  --train-samples-per-epoch 1000 --epochs 25 \
  --token-corruption-probability 0 --visual-aux-weight 1.0 \
  --resume outputs/chemvector-v3-finetune-1k/last.pt \
  --resume-learning-rate 1e-5 \
  --output-dir outputs/chemvector-v3-final-1k
```

首 shard 的 4,992 张图片已全部通过 v0.3 化学往返，平均序列缩短 16.71%。
同时检测到 13 张增强图片的二维 E/Z 几何与化学标签冲突；正式全量训练前需要
扫描并替换这类样本。

## ChemSVG v0.4

v0.4 不再输出 `<b atom_id atom_id>`。每条 stroke 直接携带可见起点和终点
坐标，SDF 连接关系由端点确定性编译：

```text
<v 315 228 N 0 0 0 0 0 0 CHI_UNSPECIFIED>
<v 507 411 C 0 0 0 0 0 0 CHI_UNSPECIFIED>
<s 315 228 507 411 SINGLE SINGLE NONE STEREONONE>
```

现有 WebDataset 已包含所需 Render JSON，不需要重新生成图片：

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.4 \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --validation-shards Data/sdf/webdataset_full_v0/shards/validation-000000.tar \
  --batch-size 4 --gradient-accumulation 1 \
  --workers 0 --epochs 1 \
  --output-dir outputs/chemvector-v4
```

协议细节见 `Data/sdf/CHEMSVG_V0_4.md`。完全重合的视觉顶点无法从图片唯一
恢复，v0.4 加载器会跳过这类样本，而不是用隐藏 atom ID 掩盖歧义。

## checkpoint 评估与 SDF 导出

评估命令先在指定样本上计算 teacher-forcing 指标，再对少量样本执行真正的
语法约束自回归生成。生成结果必须通过 ChemMolSeq 解析和 RDKit sanitize，
成功结果会写成 SDF：

```bash
.venv-train/bin/python -m chemvector.evaluate \
  --checkpoint outputs/chemvector-grounded-final-100/last.pt \
  --shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --samples 100 \
  --generate-samples 20 \
  --output-dir outputs/chemvector-grounded-final-100-eval
```

语法约束只负责保证记录顺序、字段类型、原子/键编号和端点引用合法，不会根据
距离补边，也不会替代模型判断图片中是否存在化学键。

从 checkpoint 继续训练时默认保留其中的 optimizer 学习率。如需进入低学习率
精调阶段，应显式传入例如 `--resume-learning-rate 3e-5`。

## ChemVector v0.5：几何与文字解耦

v0.5 在共享视觉编码器上增加独立 OCR head。ChemSVG decoder 继续处理顶点、
stroke 和键型；OCR head 根据顶点锚点读取局部文字。OCR 使用 UTF-8 byte 级
输出，不把词表限制为固定元素，因此可以表示 `R1`、`Ph`、`Me`、查询符号和
后续出现的其他文字。

现有 72 万张完整分子图片只包含
`B/Br/C/Cl/F/I/N/O/P/S/Se/Si`。对于主数据中不存在的缩写和字符，使用在线
合成的 128×128 黑白局部图预训练。生成器会加入单键、双键、三键和斜键干扰，
但第一阶段不加入模糊、透明背景或扫描噪声：

```bash
.venv-train/bin/python -m chemvector.synthetic_ocr \
  --samples 64 \
  --output-dir outputs/synthetic-ocr-preview
```

局部图预训练必须冻结共享视觉编码器，只更新 OCR head：

```bash
.venv-train/bin/python -m chemvector.pretrain_ocr \
  --init-checkpoint outputs/chemvector-v4-replay-interp-010/model.pt \
  --output-dir outputs/chemvector-v5-ocr-frozen \
  --steps 10000 --batch-size 64 --workers 2 \
  --ocr-learning-rate 1e-4 \
  --vision-learning-rate 0
```

不要在纯局部图预训练中调高 `--vision-learning-rate`。实验证明，即使视觉层
只使用很小学习率，局部图分布也会拉低原 ChemSVG 几何准确率。冻结后，原模型
的 267 个基础 tensor 保持逐位相同。

预训练完成后，再在完整分子图上联合训练。此时几何损失提供全局约束，才允许
共享视觉层以低学习率适配 OCR：

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.4 \
  --init-checkpoint outputs/chemvector-v5-ocr-frozen/last.pt \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --validation-shards Data/sdf/webdataset_full_v0/shards/validation-000000.tar \
  --no-shuffle --workers 0 \
  --batch-size 4 --gradient-accumulation 1 \
  --train-samples-per-epoch 1000 --epochs 1 \
  --learning-rate 1e-6 \
  --ocr-weight 1.0 --ocr-learning-rate-multiplier 10 \
  --synthetic-ocr-replay-weight 0.5 \
  --synthetic-ocr-replay-batch-size 64 \
  --visual-aux-weight 1.0 \
  --output-dir outputs/chemvector-v5-joint-1k
```

完整分子数据没有 `R1/Ph/Me` 等标签。若联合训练时不加入合成 replay，OCR head
会快速遗忘预训练得到的开放字符。Replay 分支只更新 OCR head，其视觉特征在
`no_grad` 下提取，因此不会再次用局部图改变共享视觉编码器。

PaddleOCR v5 只用于真实论文图片和非标准文字的补充伪标签。合成分子图已经有
ChemMol 元素真值和 Render 锚点，不需要再由 PaddleOCR 重复标注。

### 视觉优先生成实验

`--visual-first --ocr-fusion --stroke-scoring` 会在第一个 stroke 之前用 OCR
改写非碳标准元素、重建 decoder cache，并让学习式线段 scorer 从全部顶点对中
保留有视觉线条支持的候选：

```bash
.venv-train/bin/python -m chemvector.evaluate \
  --checkpoint outputs/chemvector-v5-joint-replay-10k-stage1/last.pt \
  --shards Data/sdf/webdataset_full_v0/shards/validation-000000.tar \
  --samples 20 --generate-samples 5 --beam-width 4 \
  --visual-first --ocr-fusion --stroke-scoring \
  --stroke-threshold 0.5 \
  --output-dir outputs/chemvector-v5-visual-first-eval
```

scorer 使用 stride-4 特征沿点对之间采样带状视觉区域，不设置键长上限。10k
训练后的全点对验证在阈值 0.5 时得到 99.72% edge recall 和 84.82% precision。

### 生成后按真实墨线修图

固定顶点生成仍可能选中“两个端点之间经过了其他线条”的伪边。`repair_v4`
直接检查每个顶点对之间是否存在连续墨线，不使用键长上限；若直线上存在中间
顶点，则只保留相邻线段，避免把两条共线键合并成一条跨点长键。修复器还会
合并同一文字内部产生的相邻重复峰，并根据小环中的芳香预测传播芳香键型。

检测式推理使用 20px OCR 取样半径。它在 500 张验证图片、15,220 个原子锚点
上的元素融合准确率为 97.12%，高于 checkpoint 原始 24px 半径的 96.98%，并
消除了 GD906 中邻近 N 泄漏到上方碳顶点的问题。完整自动推理命令直接接受
带透明边界的原始图片：

```bash
.venv-train/bin/python -m chemvector.infer_detected_v5 \
  --checkpoint outputs/chemvector-v5-vertex-10k/last.pt \
  --image Data/GD906.png \
  --output-dir outputs/chemvector-v5-auto-gd906-raw
```

命令会依次完成透明背景白底合成、等比例画布准备、顶点检测、OCR、stroke
生成、连续墨线修图、重复文字峰合并、SDF 写入和重绘预览。已经是 768×768
训练画布的图片可加 `--image-is-prepared` 跳过外部图片准备。

当前 GD906 输出为 84 个原子、104 条键、1 个连通分量，分子式
`C80H44BFN2`，RDKit sanitize 通过。该结果不再需要手工元素覆盖，也没有针对
键长或分子模板进行人工补边。

### 500 张 scaffold 独立端到端盲测

`evaluate_detected_v5` 会逐样本保存结果并支持断点续跑：

```bash
.venv-train/bin/python -m chemvector.evaluate_detected_v5 \
  --checkpoint outputs/chemvector-v5-vertex-10k/last.pt \
  --shards Data/sdf/webdataset_full_v0/shards/validation-000000.tar \
  --samples 500 \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --output-dir outputs/chemvector-v5-e2e-validation-500
```

这 500 张图片对应 125 个未见分子，每个分子四种布局。当前结果：

- 生成完成率：100%。
- 合法 SDF：48.6%。
- 完整拓扑准确率：17.2%。
- SMILES/InChIKey 完全一致率：12.4%。
- 至少一种布局能生成合法 SDF 的分子比例：85.6%。
- 至少一种布局完全化学一致的分子比例：31.2%。
- 在修复成功的样本中，顶点 precision/recall 为 98.55%/98.67%。
- 在修复成功的样本中，边 precision/recall 为 96.95%/97.24%。
- 正确边上的键型准确率：96.27%。
- 完整拓扑已经正确时，化学完全一致率为 65.12%，剩余差异主要是 stereo。

四种布局的完全化学一致率分别为：RDKit 18.4%、CoordGen 15.2%、Original
13.6%、Stress 2.4%。257 个修复失败中，120 个是墨线图断开，112 个是价态
冲突，25 个是芳香 Kekulé 化失败。因此现阶段瓶颈已经从视觉定位转移到布局
鲁棒的连线恢复、价态约束键型求解和立体化学恢复。详细结果见
`outputs/chemvector-v5-e2e-validation-500/summary.json`。

## ChemVector v0.6：无 decoder 并行 CPU 推理

v0.5 的主要延迟不是视觉模型，而是自回归 decoder。GD906 的视觉编码只需要
约 10ms，但 decoder 为 99 条边串行生成约 992 个 token，需要约 4.4 秒。
v0.6 删除部署路径中的 decoder、token embedding 和序列投影，直接并行分类
全部顶点对。实际参与部署推理的参数从 51.49M 降到 15.18M。

CPU 推理命令：

```bash
.venv-train/bin/python -m chemvector.infer_parallel_v6 \
  --checkpoint outputs/chemvector-v5-vertex-10k/last.pt \
  --image Data/GD906.png \
  --output-dir outputs/chemvector-v6-parallel-cpu-gd906 \
  --device cpu --cpu-threads 16
```

在 Ryzen 9 9950X 上，GD906 的完整推理耗时为 0.735 秒，其中顶点与 OCR
0.122 秒、全部 3,570 个点对分类 0.538 秒、墨线修复 0.075 秒。输出的分子式、
SMILES 和 InChIKey 与 CUDA 版本一致。CUDA 并行版本耗时约 0.389 秒，不再有
近千次串行 token 调度。

同一批 500 张 scaffold 独立验证图片的 CPU 结果：

- 平均推理时间：0.202 秒/张，比 v0.5 GPU 自回归的 1.324 秒快 6.6 倍。
- 完整拓扑准确率：20.6%，v0.5 为 17.2%。
- SMILES/InChIKey 完全一致率：13.2%，v0.5 为 12.4%。
- 修复成功样本的顶点 precision/recall：99.01%/98.98%。
- 修复成功样本的边 precision/recall：98.37%/98.19%。
- 墨线图断开失败从 120 个降到 12 个。
- 价态冲突从 112 个增加到 223 个，说明下一瓶颈是全局键型约束求解。

详细结果见 `outputs/chemvector-v6-parallel-cpu-validation-500/summary.json`。
当前命令加载完整训练 checkpoint 后再从内存移除 decoder；后续部署导出需要只
保留 15.18M 前端参数，并进一步进行 ONNX/OpenVINO INT8 量化。

## v0.7.2 候选实验：因子化 Edge/Bond head（未发布）

旧 stroke scorer 用一个五分类同时预测 `NONE/SINGLE/DOUBLE/TRIPLE/AROMATIC`。
候选实验将其拆成两个目标：Edge head 判断两点之间是否存在可见连线，Bond
head 只在真实边上区分可从局部线条观察到的 `SINGLE/DOUBLE/TRIPLE`。

实验中发现，Render JSON 的 `AROMATIC` 是化学键属性，不一定是局部可辨识的
绘图类型。RDKit 即使保留 aromatic bond，也会在图片中画出环内交替单线和双线；
因此不能要求局部线段分类器从完全相同的像素中区分“普通单键”和“芳香环中的
单线”。芳香性继续由顶点芳香预测、环结构和修复器全局恢复。

5,000 张训练图片、冻结视觉 stem 并解冻 pair encoder 后的局部验证结果：

- Edge precision/recall/F1：85.90% / 99.77% / 92.32%。
- 可观察键型准确率：96.23%。
- SINGLE/DOUBLE/TRIPLE recall：97.25% / 92.75% / 90.63%。
- 3,144 条芳香边只参与 Edge loss，不参与局部 Bond loss。

训练命令：

```bash
.venv-train/bin/python -m chemvector.pretrain_relation \
  --init-checkpoint outputs/chemvector-v5-vertex-10k/last.pt \
  --output-dir outputs/chemvector-v072-visible-relation-5k \
  --train-samples 5000 --batch-size 4 --learning-rate 1e-4 \
  --negative-ratio 4 --anchor-jitter-pixels 12 \
  --stroke-band 9 --stroke-band-radius 12 \
  --sample-shuffle-buffer 4096 --train-pair-encoder
```

但同一批 500 张 CPU 端到端盲测没有改善：完整拓扑仍为 20.6%，化学完全一致
仍为 13.2%，合法 SDF 从 43.0% 降到 41.2%；价态失败为 236，Kekulé 失败为
46。详细结果见
`outputs/chemvector-v072-visible-relation-cpu-validation-500/summary.json`。

因此该实验只作为后续全图求解器的概率输入接口保留，不更新 `VERSION`，也不打
`v0.7.2` tag。下一阶段应在全图范围内联合选择边和键级，显式满足原子价态、
连通性和芳香环一致性，而不是继续单独扩大局部 Bond head。
