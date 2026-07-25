# 2026-07-17 Kimi：v0.7.36 模型路线评审

## 基本信息

- 提供方：Kimi Code CLI
- 日期：2026-07-17
- 项目版本：v0.7.36
- 会话 ID：`session_4b3d14f4-24b3-4687-b3dd-e49ea1ba9389`
- 咨询方式：在项目根目录使用 `kimi -p` 非交互评审
- 文件修改：无，Kimi 只读取代码和文档

继续该会话：

```bash
bash -ic 'source /home/zhenghe/.bashrc; kimi -S session_4b3d14f4-24b3-4687-b3dd-e49ea1ba9389'
```

## 提供背景

要求 Kimi 阅读项目 README、ChemVector 主文档、v0.7.34 至 v0.7.36 训练记录，以及
端点条件化 Bond Object 的实现与评测代码。提供的关键事实包括：

- v0.7.34 双顶点分支：Vertex F1 约 92.6%，vertex exact 约 9.6%；
- v0.7.35 自由 Bond Queries：100 图有容量，9k endpoint pair recall 仅 4.62%；
- v0.7.36 GT 端点：Edge F1 99.16%，skeleton exact 78.31%；
- v0.7.36 预测端点：Vertex F1 92.85%，vertex exact 9.84%，strict skeleton exact
  7.83%；
- 匹配端点条件下 Edge recall 98.70%，顶点完全正确时 Edge exact 79.59%；
- GT 端点下 display accuracy 79.43%，chemical accuracy 87.97%。

硬约束：不使用最近邻截断、键长阈值或长键预删除；MR-TADF 不参与训练和 checkpoint
选择；OCR 输出开放字符串且不在 OCR 阶段做化学理解；最终输出以 SDF 为主。

## Kimi 主要判断

### 1. 第一瓶颈是顶点集合决策

当前两个顶点分支分别通过固定阈值解码，再使用文字 bbox 几何规则合并。局部 F1 高，
但任何一个重复点、漏点或错误计数都会使整图 exact 失败。下一步应把计数从逐点阈值
的副产品改为显式全局预测，并用集合级排序选择最终顶点。

### 2. 边存在性已泛化，完整 Bond Object 尚未泛化

GT 端点下 Edge F1 99.16% 说明连边读取器已经成立，但 Bond Object exact 只有
12.25%。因此“键集合已经泛化”应限定为“边存在性已经泛化”。芳香与 Kekule 表示
需要环级上下文，局部二维线带不存在足够信息独立完成判断。

### 3. 双键问题只在键读取层得到证明

一对端点只生成一个 Bond Object，已经解决“多条 stroke 被读取成多条键”。但当前
评测没有统计隐式顶点分支是否仍会在双键副线端点产生伪顶点。因此不能宣称顶点侧的
多 stroke 问题已经完全解决。

### 4. 优先级

```text
顶点集合 > 环级键型 > OCR 接入 > 扩大数据
```

### 5. 暂不扩大到 72 万图片

当前失败更符合表示、训练协议和组件组合问题，而非数据不足。应先完成集合级顶点和
环级键型实验，再用 2k/4.5k/9k scaling curve 判断最终架构是否真正受数据量限制。

## Kimi 建议的版本实验

### v0.7.37 集合级顶点解码

- 使用低阈值文字与隐式候选，不增加距离候选过滤；
- 候选特征加入局部 patch 和 incident edge support；
- 使用全局 Count、top-k 集合选择、Hungarian 标签和 whole-set margin；
- 建议 9k GO 门槛：Count exact 不低于 50%，strict vertex exact 不低于 20%，
  跨字体 F1 不低于 91%。

### v0.7.38 环级键型层

- 先在 GT 端点和 GT 边上隔离键型问题；
- 对键对象加入一至两跳图上下文和环级一致性监督；
- 建议 GO 门槛：display accuracy 不低于 90%，chemical accuracy 不低于 94%，
  Bond Object exact 不低于 40%。

### v0.7.39 端到端组合

- 使用预测顶点分布微调键读取器，覆盖假顶点与坐标噪声；
- 接入现有多尺度 Pixel OCR；
- 输出 ChemSVG，经过 RDKit 校验生成 SDF；
- 在 internal 与 USPTO 上完成同口径验收，MR-TADF 只作最终盲测。

## 项目侧评估

### 采纳

- 顶点集合是 strict skeleton exact 的第一瓶颈；
- v0.7.37 应先做全局 Count 与集合选择，而不是继续扩大现有 head 的曝光；
- 将“键集合泛化”修正为“边存在性泛化”；
- 双键副线伪顶点需要独立审计；
- 芳香/Kekule 键型需要环级上下文；
- 72 万扩量必须晚于结构实验和 scaling curve。

### 保留

- incident edge support 是否直接进入顶点选择，需要防止早期错误边反向强化假顶点；
- 环级消息传递的邻接构建必须保持全点对或概率加权，不得变相加入距离截断；
- v0.7.37 的具体 GO 数值是咨询建议，正式门槛需要结合当前 validation 分布确认。

### 不直接采纳

Kimi 建议不要使用化学规则修正键型。项目仍保留 RDKit sanitize、芳香性校验和确定性
SDF 编译；这些属于输出协议和合法性验证，不等同于针对 benchmark 的手写视觉修复。

## 后续实验

当前计划首先建立 v0.7.37 的错误审计与集合级顶点基线：

1. 统计双键/三键副线端点附近的隐式 FP；
2. 统计文字 bbox 合并规则造成的漏点；
3. 统计 Count error 为 -2、-1、+1、+2 的分布；
4. 在相同候选池比较固定阈值、Oracle count 和预测 Count + top-k；
5. 通过诊断后再训练集合选择器。

## 结果回填

待 v0.7.37 实验完成后填写。
