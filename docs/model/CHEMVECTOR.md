# ChemVector

ChemVector 是面向化学结构图片识别的轻量视觉到结构模型。当前主线直接从图片预测
视觉顶点、文字和可见 stroke，生成 ChemSVG v0.4，再由确定性编译器写出 SDF。

本页只维护当前状态、架构说明和训练档案索引。每次训练的完整命令、指标、失败
配置和结论统一放在 [`training/`](training/README.md) 中，不再追加到本页。

## 当前版本

- 稳定 tag：`v0.7.1`。
- 当前稳定开发基线：`v0.7.10`。
- 当前 shadow-mode 实验版本：`v0.7.12`，尚未打 tag。
- 版本从 `v0.7` 开始，每次发布增加 `0.0.1`。
- 根目录 `VERSION` 是开发版本的唯一来源。

## 当前路线

```text
768×768 图片
    -> 共享视觉编码器
    -> 顶点 heatmap + subcell offset
    -> 多尺度 Pixel OCR（12/20/32px）
    -> 全点对 Edge/Bond 分类
    -> 基于真实墨线的图修复
    -> ChemSVG v0.4
    -> RDKit 校验与 SDF
```

关键约束：

- 不使用键长阈值删除候选边。
- 不通过隐藏 atom ID 恢复连接关系。
- OCR 输出开放 UTF-8 字符串，不在 OCR 阶段执行化学理解。
- `Ph/R1/A/X` 等都作为普通字符串处理。
- MR-TADF 不进入训练，只作为外部 benchmark。
- 正确、错误和无效预测都保留对应 SDF。

## 模型规模

- 早期自回归 ChemMolSeq 基线：49,974,768 参数。
- v0.5 完整训练 checkpoint：约 51.49M 参数。
- 删除 decoder 后的并行部署路径：约 15.18M 参数。
- 输入：768×768 RGB。
- 早期共享视觉 token：24×24，共 576 个。

v0.6 之后的部署推理不再需要自回归 decoder，CPU 500 张验证平均约 0.202 秒/张。
当前优先提升模型能力，量化和进一步加速后置。

## 当前最佳结果

v0.7.10 沿用 v0.7.9 checkpoint：

```text
outputs/checkpoints/chemvector-v079-relation-all-pairs-9k/last.pt
```

SHA256：

```text
553fbb80a823ce08b338d1c2b2263a5084ce1b42d714eb4deca7e4728e2427c9
```

在 1,100 个完全未进入训练的 MR-TADF 分子上：

- 有效 SDF：92.09%。
- 化学完全一致：79.73%。
- 原子数一致：82.18%。
- 键数一致：82.18%。
- 平均 CUDA 推理：0.118 秒/张。

v0.7.9 在 v0.7.8 上训练 146,308 个 relation/pair 参数。全点对骨架训练将真值
顶点下的骨架完全一致从 37.35% 提高到 49.40%。v0.7.10 不训练新权重，保留全部
点对概率用于断图桥接；USPTO clean 1k 的有效 SDF从 177 提高到 192，完全一致
保持 22，MR-TADF 完全一致保持 79.73%。

v0.7.11 结构 MoE 已复核为 NO-GO。USPTO/MR-TADF 的结果来自完全回退 v0.7.10，
MolGrapher routed detected skeleton exact 只有 1.3%，不能视为通用能力提升。

v0.7.12 在固定 100 张审计后 MolGrapher 图片上验证集合式顶点：纯 heatmap head
过拟合的 vertex exact 为 87%，解冻一个 stride-4 block 为 90%；Atom Query 负责
计数、dense peak 负责一对一坐标吸附后，Count exact 为 100%，4px/6px vertex exact
为 95%。接入现有 relation 后 skeleton exact 最高 23%，当前第一瓶颈已转为检测后
关系判断。该分支仍处于 shadow mode，不替换 v0.7.10 推理。

v0.7.7 字体扩展实验将内部可见 OCR 提高到 96.89%，但整体微调 OCR head 导致
MR-TADF 完全一致降至 71.27%；插值和新旧字体混合训练也未通过回归门槛。因此
三个 v0.7.7 checkpoint 均已否决。v0.7.8 已改用冻结基线和零初始化残差适配器。
角色头继续保持影子模式。当前优先改进骨架补边和双键横截面特征，OCR 暂时冻结。

## 模块边界

```text
image_io.py             图片准备、PIL/Tensor 转换
detection.py            vertex 检测与锚点 OCR
stroke.py               点对监督与线段分类
parallel_v6.py          无 decoder 并行结构预测
repair_v4.py            墨线拓扑与化学图修复
sdf_io.py               SDF 写入及立体往返校验
evaluation_metrics.py   纯图比较、叠加图与汇总指标
benchmark_diagnostics.py 图级错误分类与回填
infer_*.py              参数解析与结果落盘
evaluate_detected_v5.py 数据迭代与 pipeline 调度
```

核心模块不依赖 `infer_*` CLI；repair 不依赖 generation；并行推理通过公开检测 API
运行，不引用 generation 内部函数。

## 数据与协议

- [ChemMolSeq v0.2](../../Data/sdf/CHEMMOLSEQ_V0_2.md)：最早直接 graph 序列。
- [ChemMolSeq v0.3](../../Data/sdf/CHEMMOLSEQ_V0_3.md)：空间排序短序列基线。
- [ChemSVG v0.4](../../Data/sdf/CHEMSVG_V0_4.md)：当前主协议。
- [WebDataset 数据说明](../../Data/sdf/README.md)：训练样本和加载器协议。
- [MR-TADF benchmark](../benchmarks/MR_TADF_V1.md)：外部测试、历史结果和错误诊断。

## 训练档案

完整时间线见 [训练档案索引](training/README.md)。主要阶段：

| 阶段 | 简要说明 |
|---|---|
| [001](training/001-grounding-100.md) | 100 样本视觉接地，证明模型确实读取图片 |
| [002](training/002-chemmolseq-v03-1k.md) | ChemMolSeq v0.3 短序列和空间排序 |
| [003](training/003-chemsvg-v04-initial.md) | ChemSVG v0.4 点线端点协议 |
| [004](training/004-v05-ocr-decoupling.md) | 几何与开放字符串 OCR 解耦 |
| [005](training/005-v05-visual-relation-repair.md) | 全点对连线与真实墨线修复 |
| [006](training/006-v06-parallel-cpu.md) | 无 decoder 并行 CPU 推理 |
| [007](training/007-v072-factorized-relation.md) | 因子化 Edge/Bond head，端到端未通过 |
| [008](training/008-v072-replacement-ocr.md) | 平衡字符替换 OCR |
| [009](training/009-v073-pixel-ocr.md) | 原图高分辨率 Pixel OCR |
| [010](training/010-v075-chemvision-multiscale-ocr.md) | ChemVision 三尺度 OCR |
| [011](training/20260715-v0.7.6-vertex-recall-pilot.md) | 9k vertex-head pilot，完成 |
| [012](training/20260715-v0.7.7-font-expanded-ocr.md) | 字体扩展有效，但整体微调遗忘旧分布，checkpoint 全部否决 |
| [013](training/20260715-v0.7.8-ocr-residual-adapter.md) | 冻结旧 OCR 的 29k 残差适配器，完成 |
| [014](training/20260715-v0.7.9-skeleton-all-pairs.md) | 全点对 relation 与概率约束 repair |
| [015](training/20260715-v0.7.10-probability-bridge.md) | 使用连续点对概率恢复断图，当前最佳 |
| [019](training/20260715-v0.7.11-conservative-structure-moe.md) | 风格 gate 只实现回退，已判定 NO-GO |
| [020](training/20260716-v0.7.12-atom-query-vertex.md) | Atom Query 集合计数与 dense 几何吸附，顶点方向 GO |

## 常用入口

创建训练环境：

```bash
python -m venv --system-site-packages .venv-train
.venv-train/bin/pip install -r requirements-train.txt
```

运行全部测试：

```bash
.venv-train/bin/python -m unittest discover -v
```

查看 benchmark：

```bash
.venv-train/bin/python frontend/server.py
```

服务默认监听 `0.0.0.0:8765`；浏览器使用服务器 IP 和端口 `8765` 访问。

## 后续文档维护

新训练不得继续扩写本页。使用
[`training/TEMPLATE.md`](training/TEMPLATE.md) 新建独立记录，完成后只更新：

1. [`training/README.md`](training/README.md) 的时间线和当前下一步；
2. 本页“当前版本”“当前最佳结果”和训练档案表中的一行摘要；
3. 对应 benchmark 文档中的外部评测结果。
