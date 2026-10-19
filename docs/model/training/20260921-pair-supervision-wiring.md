# 保守pair掩码：默认关闭的loss接线与固定58采样验证

## 已完成

新增版本化loss入口及训练端掩码组件，默认保持原行为；只有显式指定
`pair_supervision='matched-only-v1'`才启用保守监督。

**没有修改旧continuation、旧训练launcher、推理路径、模型权重或历史SHA绑定文件。**
这是loss入口的接线完成，不是已启用正式训练。

固定58保存候选上的实际CPU采样结果：

| 本次固定种子重放 | 旧采样路径 | 开启保守掩码 |
| --- | ---: | ---: |
| 实际采样正例 | 1,103 | 1,103 |
| 实际采样负例 | 6,260 | 6,244 |
| 抽中的已标记风险负例 | 26 | 0 |
| 暂不监督的候选pair池 | 0 | 626 |
| 无可监督pair的图片 | 2 | 4 |

上轮只统计了采样池和容量；本轮调用实际运行时采样组件，确认**本次固定种子下**旧路径
确实抽中全部26条诊断风险连接，而掩码后没有抽中任何一条。不能据此推断历史训练的随机采样
或参数梯度，也不能称为补回26条连接或识别准确率提升。

真实样本部分CPU工作进程1.716秒，0模型前向、0checkpoint加载、0优化器步骤、0GPU。
完整head/loss接线另用合成模型和可控候选测试，相关测试总计**129 passed / 1.91秒**。

## 代码入口

- `chemvector/training/curriculum/pair_supervision.py`：只读原匹配结果，生成监督mask；
  先过滤忽略行再调用原采样器，返回原始pair行号。不依赖`tools`、诊断归属或修图代码。
- `chemvector/training/curriculum/continuation_pair_supervision.py`：版本化loss入口。
  默认直接委托旧`continuation.continuation_losses`，开启时改变pair采样并增加覆盖计数。
- `tests/chemvector/test_curriculum_pair_supervision.py`：23项新增接线/采样合成测试。
- `tools/audit_pair_supervision_wiring.py`：固定58的CPU采样与旧结果逐项比对。

调用示例（仅说明loss API，不授权训练或更新参数）：

```python
from chemvector.training.curriculum.continuation_pair_supervision import continuation_losses

# 默认完全委托历史loss路径。
losses, coverage = continuation_losses(model, batch, config)

# 显式选择保守键监督；实际项目调用时仍须传原固定atom/bond阈值。
losses, coverage = continuation_losses(
    model, batch, config,
    atom_threshold=provenance['atom_threshold'],
    bond_threshold=provenance['bond_threshold'],
    pair_supervision='matched-only-v1',
)
```

条件归属版本不在运行时允许值中。无效策略在任何前向之前报错。
启用后原匹配、正例标签、raw/set监督、文字角色和bbox路径均保持不变；
不存在扩大6px半径、注入GT候选或修改推理图的操作。

旧loss源码已被历史实验按SHA绑定，因此没有就地改写或临时猴子补丁。
新版本保留一份有界的loss编排分支，并用逐loss、逐标签及文字覆盖对照测试约束差异；
底层模型和loss计算函数继续复用旧组件。后续改动此分支仍需通过这些对照测试。

## 接线验证覆盖

1. 默认省略策略或显式使用`legacy-all-pairs-v1`时，所有loss、覆盖字段、RNG状态与旧入口一致。
2. 所有端点已匹配时，即使开启新策略，各项loss与旧版逐值一致，原标签逐值一致。
3. 在原6px边界内保持监督；6.25px和7px时，不确定端点涉及的pair不会进入真实键头。
4. 开启策略不改变预测张量、其他五类组件loss、文字覆盖或原有正例。
5. 单图/多图均在采样前过滤，每个样本只使用自己的原子映射；正例和原配额完整保留。
6. 全部pair被忽略或没有pair时，键头不执行；loss保持有限的零，键头不出现伪造梯度。
7. 合成反向传播保留正例logit梯度方向，冻结文字角色头仍冻结，模型参数和buffer哈希不变。
   没有创建优化器，不将合成梯度检查解释为真实checkpoint的训练效果。

启用策略新增coverage字段：`pair_supervision`、`pair_total_rows`、`pair_ignored_rows`、
`pair_supervised_rows`、`pair_supervised_negative_rows`、`pair_sampled_positive_rows`、
`pair_sampled_negative_rows`、`pair_branch_skipped`。默认旧路径返回结构不变。

## 固定58实际采样验证

- 只读取已授权诊断范围内的旧训练回放记录，沿用原manifest顺序。
- 每图种子为`20260921 + manifest行号`；两种策略使用相同种子，并在结束时恢复外部RNG状态。
- 原匹配、正例标签及启用后的监督mask，分别与保存匹配和上轮离线报告逐项一致。
- 旧策略58图的实际采样行号与直接调用原采样器完全一致。
- 58图实际采样的正例数和负例数均符合上轮计算的逐图容量，未用忽略行填充配额。
- 新增空pair分支仍是此前明确的两张基础图；另两张原来就没有pair。

独立复核本轮303项输入SHA、58项逐图SHA，以及上轮245项输入SHA均通过；
逐条检查采样行号、候选对、正例覆盖和26条风险记录，汇总一致。

## 边界与下一步

- 本轮没有使用真实checkpoint运行完整loss前向/反向；真实样本检查的是保存候选上的mask和采样。
- 旧训练launcher没有切换到新入口，旧`train_step`及训练协议也没有被修改或复用授权。
- 旧checkpoint格式只记录原update recipe，不能把它当成记录了新pair策略。
  后续试训必须在新的运行协议/进度记录中明确写入pair策略，避免结果来源不清。
- 下一步可固定极少量已在范围内的样本，用真实initial做有CPU时间/前向次数上限的
  无参数更新检查，核对loss有限性、候选一致性和冻结模块；通过后再决定是否申请定额试训。
- 新掩码只停止不确定pair的负例监督，缺失正例的恢复、剩余归属歧义和真正识别效果仍未解决。

## 报告与复现

- [采样验证报告](../../../outputs/audits/chemvector-pair-supervision-wiring58-20260921/report.json)
- [预算和资源回执](../../../outputs/audits/chemvector-pair-supervision-wiring58-20260921/runtime.json)
- [上轮两种掩码对照](20260921-pair-supervision-mask58.md)

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -m pytest -q \
  tests/chemvector/test_curriculum_pair_supervision.py \
  tests/chemvector/test_pair_supervision_mask.py \
  tests/chemvector/test_selector_missing_graph_trace.py \
  tests/chemvector/test_selector_coordinate_contract.py \
  tests/chemvector/test_curriculum_selected_matching.py \
  tests/chemvector/test_selector_radius_reference.py
```

报告SHA256：`3699d01515776b5e37996ae8f9e2089a18af0bd847aca32f4eb725cf41d0fe83`。
