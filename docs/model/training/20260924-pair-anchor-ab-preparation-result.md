# 单遍有／无增强对照：入口和 CPU 准备完成，待新授权

## 当前结论

**对照入口、设备缓存适配和停损规则已实现，274 项 CPU 回归通过；尚未训练。**
没有加载真实 checkpoint 作模型运行、调用 GPU 或创建本次训练授权／attempt／真实实验目录。
initial checkpoint 只流式核对 SHA。CUDA 可用性和 GPU 数值等价性均未测试，不能据此声称驱动有问题或 GPU 验证已通过。

接续 [上一轮排程与原图复核](20260924-pair-anchor-wiring-result.md)。
完整方案见 [事前协议](20260924-pair-anchor-ab-protocol.md)。

## 已实现

1. 新的独立两臂入口及单次授权检查，不修改旧的哈希绑定实验实现或现有默认推理入口。
   旧单次训练许可不能复用；新授权须绑定新用途、协议、准备报告及 42 个实现／测试／协议文件。
2. 执行设备缓存适配器：在当前设备重新生成候选与匹配之后，才生成偏移计划。
   不把历史行号复制到新缓存；原采样只生成一次，两臂共用。
   输入冻结，只有 pair head 可更新；不增加跨图非边回放或改 OCR／selector。
3. 一遍完整 58 图、两臂按同样的逐图顺序配对执行，避免 50 步截断最后 4 张回放图。
   各最多 54 次有效更新，总计最多 108 次，**单卡整个调用合计 600 秒**。
   缓存、校验、OCR／SDF 评估、更新与保存都计入总时间，不是两臂各 600 秒。
4. 每臂初始全量评估必须一致；每个非空样本的 weight0 adapter 与原直接调用须在执行设备逐值相同。
   新缓存若不能在更新上限内完成完整 58 图，首次更新前停止，不能截断或自动加预算。
5. 在 attempt29 和58，对两臂都评估原位全 58 图、全部候选对；任一臂保护失败则停止两臂。
   每张图只访问一次，因此本轮实际只测试固定 `(-4,-4)` 偏移，不称为八方向稳健性训练。

## 保护规则比只看总数严格

- 保留每一条 initial 正确的 raw 键型及芳香提升后正确键型，分别记录，不能相互掩盖。
- 确认假连接按**具体候选对行号**判断新增；一条旧假边消失、另一条新假边出现，即使总数不增也失败。
- 保留 initial 正确／有效 SDF，无新增 fallback；候选、标签、文字签名或分母变化直接中止。
- unknown 的新增存在行单独报告，不伪装成确认假连接，也不进入负例监督。
- 终点还要求增强组相对无增强组保住其正确集合与 SDF，至少一个预定指标严格改善；
  loss 下降、净增抵消损坏或中途不等长度的两组结果均不能判成功。

raw/promoted 两套局部正确键并不等同于最终 SDF 逐键匹配；完整分子另用严格 SDF identity、
有效性和 fallback 检查。即使通过也只是 58 图 train-fit 可行性，不是独立准确率或部署许可。

## 验证结果

- 新增对照／设备适配测试 44 项，连同相关新旧回归，**274 passed / 2.70 秒**，无跳过。
- 合成测试覆盖：新授权边界、按候选对识别新假边、raw/promoted 保护、对照损失等价、
  多步 toy AdamW 等价、非空纯负例／空分支、缓存／计划篡改、设备变化与双层步数上限。
- 另做未写入主 JUnit 计数的合成 CPU 交叉检查：随机初始化实际 `DirectAtomPairBond` 小配置，
  weight0 和 weight1 下，新设备适配器与上一轮 CPU 接线入口的 loss 及所有参数梯度逐值一致。
  未加载训练权重、未使用真实图片或优化器更新；不是 GPU parity 或真实模型效果测试。
- 准备进程 **4.719 秒**，低于 CPU 120 秒上限。
- 准备后再次核对 **42 个实现绑定、3 个产物绑定**及协议摘要，全部一致。
- `--check-only` 返回：来源检查通过、58 张训练图、authorization 不存在、attempt 未消耗、训练未执行。

CPU 准备输出在 `outputs/audits/chemvector-pair-anchor-ab-preparation-20260924`。
真实实验路径 `outputs/experiments/chemvector-pair-anchor-ab-20260924` 尚未创建。

## 尚需的明确授权

仅对已有固定 58 张 train 图增加用途 `local_pair_anchor_ab58_once`，
授权**一次**上述两臂试训：GPU0 单卡合计 10 分钟，每臂最多54步／两臂合计108步。
不生成新数据、不读取开发／封存样本、不查身份、不改默认模型、不部署、不自动重试。

获准后，fresh cache、执行设备等价性和初始两臂一致性仍在同一次预算内先检查。
任何前置检查失败即终止；它不是额外免费的预跑，也不自动改设备或复用这次许可。

入口为 `python -m chemvector.training.curriculum.pair_anchor_ab_trial`。
目前缺少新授权文件，直接执行会拒绝，不会静默开训。

## 产物与摘要

- [准备报告](../../../outputs/audits/chemvector-pair-anchor-ab-preparation-20260924/report.json)
- [默认拒绝的协议草案](../../../outputs/audits/chemvector-pair-anchor-ab-preparation-20260924/protocol-draft.json)
- [历史覆盖依据，不转用历史监督行号](../../../outputs/audits/chemvector-pair-anchor-ab-preparation-20260924/metadata-plan.json)
- [测试收据](../../../outputs/audits/chemvector-pair-anchor-ab-preparation-20260924/synthetic-tests.xml)

准备报告文件 SHA256：`cbed73a26614854e9841bd44d6339af80657b239301ad865d76e05f9e8421de3`。
协议语义摘要（sorted compact JSON）：`1fec951797b777edfd7a855aeacf831fe4b5f9a855f07122360a5ed2854933fe`。
协议文件 SHA256：`37774d262088cdc3f135ee08709c0046eb2c9deee1db4c5ea51b0418c8fd945f`。
