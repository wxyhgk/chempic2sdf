# G3 P1-N Regional Maxima 零训练审计

> 日期：2026-08-03
>
> 状态：机制 GO，需要全新 fresh 确认
>
> 性质：冻结 P1-A 的对象生成机制审计，不是新模型训练

## 问题

P1-I 到 P1-M 把 top-128 三槽 Atom candidate 按固定 `3px` 距离组成连通分量，再训练
component ranker。固定像素半径会随图片尺度和风格改变语义，并可能通过 single-link 链式
合并不同 Atom。P1-N 只回答：

> 不使用任何像素聚类半径，将冻结 P1-A 的三槽输出先压成每个空间 cell 一个连续分数，再从
> 完整 score field 的 regional maxima 直接实例化 Atom object，能否提高对象集合选择？

MolGrapher 只作为实现依据：它训练连续 Atom heatmap，再通过 regional maxima 生成对象；本轮
不采用其 `5x5/15x15` 窗口、`0.1` 阈值、6 近邻或 `1.75x` 键长删边规则。

## 固定算子

1. 复用冻结 P1-A 与绑定 vision，CUDA BF16 forward；权重零更新。
2. 每个 `192x192` cell 取三个 presence logit 的最大值；同分取最小 slot。
3. anchor 取胜出 slot 的原预测 anchor，禁止平均三个 slot 坐标。
4. score 转 FP32，对完整等值 plateau 做 8 邻接 regional-maximum flood-fill。
5. 每个 maximum plateau 只生成一个对象；代表为最小 row-major cell。
6. 对象按 `logit desc, source_id asc` 确定性排序，最多保留 128 个。
7. 不使用 absolute threshold、NMS 窗口或像素聚类半径。
8. 峰数不足请求 count 时返回现有峰并记录 shortfall，禁止用非峰背景补齐。

实现：

- `chemvector/components/vision/regional_maxima.py`
- `chemvector/evaluation/audits/atoms/audit_g3_p1n_regional_maxima.py`

## 数据边界

- train5k：5,000 张，只用于当前机制主判读。
- route100：已经被 P0-M 到 P1-M 多次消费，只能复现已知迁移诊断，不能作为新验证集。
- 未读取 confirm、fresh、sealed、USPTO、MR-TADF，未生成 SDF，未更新前端。
- 6px 只用于结果匹配评分，不参与对象生成。

完整命令：

```bash
source .venv-train/bin/activate
python -m chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima \
  --split both
```

## 结果

### Train5k

| 口径 | Atom recall | 整图 exact |
|---|---:|---:|
| 3px component，零训练 known representable | 98.176% | 64.22% |
| Regional maxima，known representable | **99.490%** | **87.94%** |
| Regional maxima，完整 GT count | 99.401% | 87.32% |
| Regional maxima，top-128 可达性 | 99.889% | 97.48% |
| Regional maxima，冻结 deploy count | 93.932% | 10.82% |

相对同图 3px component 基线：

- 两者都 exact：3,206 张；
- regional maxima 独立救回：1,191 张；
- regional maxima 破坏：5 张；
- 两者都错：598 张。

### 历史 Route100 诊断

| 口径 | Atom recall | 整图 exact |
|---|---:|---:|
| 3px component，零训练 known representable | 97.461% | 53% |
| P1-K 训练后 component ranker（历史参考） | 98.405% | 68% |
| Regional maxima，known representable，零训练 | **98.991%** | **79%** |
| Regional maxima，top-128 可达性 | 99.675% | 95% |

route100 上 regional maxima 相对 3px 基线救回 26 张、破坏 0 张。该结果已经暴露，只能作为
机制诊断，不能据此选择 checkpoint 或声明未知分布泛化。

## 残差

- train5k 平均每图仍有 566.36 个 regional maxima，P1-A 还不是干净的单对象 heatmap；
  当前提升来自更合理的空间峰排序，不代表计数已解决。
- train5k 的 7 张 same-cell collision 图，known-full exact 仅 1/7；单 cell 单对象确有极少数
  表达上限，后续需要更高分辨率或显式 second-object fallback。
- route100 的 stress 布局 known-full exact 为 44%，明显低于 coordgen 100%；必须继续报告
  layout 分桶，不能用总体结果掩盖尺度/风格问题。
- 冻结 deploy count 仍只有 train 591/5,000、route 13/100 count exact。对象场和 count 必须
  分开训练与判读。

## 决策

五项 train 机制门全部通过：top-128 recall/exact、known exact 增益、known recall不回退及
零 peak shortfall 均合格。

```text
decision = mechanism_go_requires_fresh_confirmation
```

停止 P1-M swap verifier 和固定半径继续调参。下一步冻结 P1-A/vision，在 P1-A spatial hidden
上训练轻量单通道 Atom center field 与 sub-cell offset。P1-N 本身没有 checkpoint，不创建
模型 tag；新 center-field 产生正式 checkpoint 后，tag 必须与模型名一致。真正替换主路前仍需
新的、未消费的 molecule/scaffold/style-disjoint fresh commitment。

## 产物

```text
outputs/audits/chemvector-g3-p1n-regional-maxima-zero-train/audit.json
```
