# 连键端点扰动：单遍匹配对照试训协议（待新授权）

承接 [排程／接线与原图复核](20260924-pair-anchor-wiring-result.md)。
本轮仅准备代码、CPU 合成测试和保存元数据复核。不会创建授权、预约训练、加载真实模型或使用 GPU。

## 单一问题

固定上游时，在相同原监督上增加一个固定偏移的 typed loss，是否改善**正常原位置**连键识别，
同时不损坏原正确键、产生新的确认假连接或损坏 SDF？

这不是八方向稳健性试验，不是增加分子种类，不是独立泛化测评。
两臂均不添加上一轮的跨图非边回放，不与历史回放结果直接作单变量比较。

## 两臂与固定范围

- 同一 initial，SHA `09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`。
- 原 fixed58 training_samples，manifest SHA `f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111`。
  新用途仅拟定为 `local_pair_anchor_ab58_once`，原数据发布及 selector 用途不改写。
- control：原 typed pair loss。augmentation：原 loss + 1.0 × 一个固定偏移的 typed loss。
- 两份独立但相同初值的 pair head、独立全新 AdamW；学习率 2e-5、betas(.9,.999)、eps1e-8、
  weight_decay0、gradient_clip1。FP32，无 autocast／TF32，全模块 eval，上游、角色／裁框、OCR 冻结。
- 每张图只访问一次，沿用逐图计数的首偏移 `(-4,-4)`，不挑方向、不扫系数。
  同一图的两臂顺次更新后才进入下一图；空原监督两臂一起跳过，空增强不重抽。
- 原采样只在 fresh cache 上做一次，CPU seed=`20260921+ordinal`、negative_ratio6、minimum_negatives48。
  两臂共享相同原行号，未知不当负例；原正例、纯负例和空图分母不得丢失。
- 新增视图保持原归属、6px／唯一近邻／图内保护，不重匹配、不注入 GT 坐标。

## 硬预算：整个 worker 共用，而不是每臂各算一次时间

| 项目 | 上限 |
| --- | ---: |
| GPU | 明确指定单卡 0，无 CPU 回退 |
| 总墙钟，包括授权检查、加载、缓存、OCR 评估、更新和保存 | 600 秒 |
| 每臂尝试／实际更新 | 58 / 54 |
| 两臂总更新 | 108 |
| 冻结结构前向 | 58 |
| 全量 58 图评估 | 6 轮：两臂各初始、attempt29、attempt58 |
| 采样接口等价性检查 pair 前向 | 108 |
| 训练 pair 前向（原位置与额外分支计入） | 162 |
| CPU 保存 tensor cache、plan、像素张量 | 2 GiB |

一次调用，自动重试／续训 0。外层独立计时器到限终止本次进程组；worker 提前 30 秒留收据空间。
独占 attempt 文件和 worker-started 文件均不可覆盖，启动失败也不自动重用许可。
显存不足、CUDA 不可用、非有限数、冻结内容变化或任何预算超限均退出，不能改设备／协议补跑。

## 首次更新前必须通过

1. 新用途和单次运行授权精确绑定最终协议、准备报告、实现哈希、数据与 initial。
   已消耗的旧授权不可复用。`--check-only` 只核对来源，不能授予运行权。
2. 在指定执行设备上重新生成 58 图候选、特征、原匹配和标签；不读取旧 row_id 当新监督。
   目标锚点从同一批原目标复制到 CPU 作增强保护，原 GPU 匹配不在 CPU 上重做。
3. 每个新 cache 绑定图像／目标、候选／标签／特征、增强计划、设备 UUID 与版本环境。
   缓存中不保存可训练的 pair hidden/ribbon encoding。
4. 若新缓存非空分支超过 54，直接退出，不能截断最后几张以适配预算。
   历史预期为 54，但新设备不必伪装成旧 CPU 候选。
5. 对每个非空样本，在执行设备上比较 weight0 adapter 与原 direct pair 调用的 logits／loss，逐值一致。
6. 两臂初始全 58 图指标／文字签名完全一致，上游、OCR、initial pair 状态未变，之后才创建优化器。

## 验收、停损与比较

每个检查点都用原位置、全部 58 图及全部候选对；不是训练采样行或 GT 合格偏移子集。
原芳香提升、编译、OCR 和阈值不改动。

分别保留两套正确类型键 ID：raw head argmax 和芳香提升后的键型。
这两套不是最终 SDF 的逐键对齐：最终结构另以严格 SDF identity、有效性和 fallback 验收，
不能把 raw／promoted 局部正确数称为最终分子准确率。

任一臂相对共同 initial 出现以下任一情况，在 attempt29 或58检查后停止两臂：

- 丢失／错型任何原正确 raw 或 promoted 目标键。
- 出现**具体新确认假 pair**。即使假边总数没增加，换成另一条也判失败。
- 原 exact SDF 不再 exact、原 valid SDF 变 invalid、出现新 fallback。
- 候选／目标／文字签名或评估分母变化（此类属于契约错误，直接中止）。

unknown 的具体新增存在行独立报告，不伪称假边、不作为训练负例。
最终 augmentation 还须保住 control 正确键集合与 SDF、无 control 之外新确认假 pair，
且 raw/promoted/SDF exact 总量均不下降、至少一项严格增加，才算 train-fit 可行性通过。
净增不能抵消损坏；loss 不决定成功。任一臂中途失败，都不把不等长度结果当完整成功对照。

原图交叉的 C12—O13 与 C9=O10 属于全量逐键保护集合，不因困难而删除；
原位置和增强各臂保存逐图完整 logits、promoted 键型、原 pair 行、编译 trace 与指标。
54 上限只保证一遍原监督机会，不保证全部 1,103 条可达键均获额外增强；
历史单偏移只新增 650 条真键观察，其余仍保留原位置监督。

## 本次 CPU 准备

新 adapter 只用人工合成 CPU features／候选运行测试（含 toy 参数更新）；
不读取真实图作模型前向，不触碰 CUDA 可用性。运行完整新旧回归、绑定旧 wiring/one_sweep 元数据，
保存默认拒绝的 protocol-draft 和报告。不创建 authorization、attempt 或真实实验输出目录。
CPU 准备上限 120 秒，合成测试子进程上限 60 秒。

准备通过仍不代表已经验证 GPU 数值、证明增强有效或获准训练；
GPU 等价性／fresh cache 是未来获准的同一次 600 秒调用内的前置门槛，失败立即停止且不自动重试。
