# G3 P4-E8 Local Multiplicity Components

> 日期：2026-08-08
>
> 状态：已完成，bounded local multiplicity GO
>
> 审计：`chemvector-g3-p4e8-local-multiplicity-components`

## 目标

P4-E6有28个matching-conflict漏点，P4-E7证明固定cell center不能修复。本轮只重放43张已知
失败图，在6px和10px分别构建GT/top1024 candidate二部连通分量，并对每个分量独立计算最大
匹配缺口，确定局部multiplicity decoder需要的容量。

## 固定协议

- 43张身份由P4-E6 failures SHA固定，不读取其余dev图片；
- top1024、预测offset、最大基数Hungarian不变；
- eligibility半径固定6px和诊断用10px；
- 报告每个缺口分量的GT数、candidate数、最大匹配数和deficiency；
- 6px总deficiency必须重放58，10px必须重放23；
- shared-deficient分量若全部GT不超过4、deficiency不超过2，并覆盖至少90% shared deficiency，
  则`bounded_local_multiplicity_go`；否则为unbounded/mixed；
- no-candidate分量不计入multiplicity容量门，它属于峰生成/坐标细化问题；
- 不训练、不读pilot/confirm_v2或外部/封存集。

本轮零训练，不创建tag。

## 实际结果

43张失败图、6px deficiency 58和10px deficiency 23全部严格重放。局部分量比预期更简单：

| 半径 | 缺口分量 | no-candidate | shared-deficient | GT数分布 | candidate数分布 | deficiency |
|---:|---:|---:|---:|---|---|---|
| 6px | 58 | 30 | 28 | 30个×1，28个×2 | 30个×0，28个×1 | 全部1 |
| 10px | 23 | 10 | 13 | 10个×1，13个×2 | 10个×0，13个×1 | 全部1 |

也就是说，所有shared conflict都严格是`2 GT / 1 candidate`，没有3/4 GT分量，也没有需要一次
补两个以上对象的分量。预注册的最多4 GT/deficiency 2门全部通过，实际充分容量更小：二分一个
共享candidate为两个anchor即可覆盖当前全部shared deficiency。

最终决策`bounded_local_multiplicity_go`。这不是让整个Atom模型重新回到固定多slot，而是在
冻结普通单峰路径后，只对检测为共享的少量局部峰运行一个保守binary split head：

- 输入候选局部视觉特征和邻近线/键端点证据；
- 输出`keep one / split two`及两个无序anchor offset；
- split-two用二点Hungarian监督；
- 普通candidate作为大量hard negative，正式门要求paired exact零break；
- 30个no-candidate缺口不由split head处理，另交给峰生成/结构化anchor refinement。

该机制来自一般局部几何，不针对stress、MR-TADF或特定分子。

## 产物

- audit SHA256：
  `7062f586af88ff122b31538890c8bc72a9af7b8dfb54d3f9f5841b889784480b`；
- 运行耗时11.80秒；
- 只读取43张已知失败dev图，pilot、confirm_v2及外部/封存集未读。
