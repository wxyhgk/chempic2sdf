# G3 P3-E14 Style90k Count 数据规模 A/B

> 日期：2026-08-04
>
> 状态：能力门 NO-GO；数据规模因果 GO
>
> 基线：P3-E10 expanded3316 h64
>
> 输出：`outputs/smoke/chemvector-g3-p3e14-deepsets-count-style90932-h64`

## 目标

P3-E13 证明 `coarse Count ±4` 窗口覆盖 confirm 243/256，但冻结 matching curve 的小头仍
不能泛化。与此同时，P3-E9 到 P3-E10 将 Count 训练从 1,760 扩到 3,316 图后，confirm
提高约 10.6pp。P3-E14 直接验证剩余泛化差距是否主要来自训练 molecule 太少。

本轮保持 P3-E10 的 DeepSets h64、输入特征、初始化顺序、AdamW、学习率、总 exposure
99,480 和 optimizer step 1,560 不变。唯一主要变量是训练数据从 829 个 molecule 的 3,316 图
反复 30 epoch，改为 22,733 个 molecule 的 90,932 张唯一图近似单遍。

## 数据隔离

原始 `style_train_100k` 包含旧 dev/confirm/route 样本，不能直接全量训练。本轮先排除：

- P3-E9/P3-E10 dev256 与 confirm256；
- P3-E7 route_select 与 route_confirm；
- historical route100；
- 上述 665 个 molecule 的全部 631 个 scaffold。

最终训练 manifest：

- 90,932 张图；
- 22,733 个 molecule；
- 19,806 个 scaffold；
- 四种布局完整；
- 与全部保护集 molecule/scaffold 重叠均为 0；
- MR-TADF 未参与训练、阈值、checkpoint 或决策选择。

manifest SHA256：

```text
5a234b4984d045db635c0196ce481653574ff39897892db0330ab021a063dda1
```

## 冻结 Evidence Cache

缓存严格冻结：

- P2-D stride-1 top128 candidate checkpoint；
- P2-W base；
- P3-E1 expert；
- Count 输入仍为 P3-E10 的 query/candidate/global 特征。

为适配 100k 规模，新增 20 片可恢复 evidence cache。每片只在候选、base/expert 特征、标签、
样本顺序和文件 SHA 全部成功后原子提交。正式缓存统一使用 candidate batch24；此前 batch4/16
的工程 pilot 位于不带 `-b24` 的废弃目录，不进入训练。

正式缓存：

```text
outputs/cache/chemvector-g3-p3e14-style90932-count-evidence-b24
```

- 90,932 张、20 片、约 2.3 GB；
- index SHA256：`14444a819fedf62130452de0aa85a6d62b061f644deb0dda4f46978a1c68ebb9`；
- 候选 recall：2,841,824 / 2,849,100 = 99.7446%；
- 候选整图可达：86,504 / 90,932 = 95.1304%；
- 全量生成：2,315.5 秒。

## Matched-Exposure 协议

90,932 张唯一图全部随机遍历一次，再从第二个固定随机排列中重复 8,548 张：

```text
unique images = 90,932
repeated exposure = 8,548
total exposure = 99,480
optimizer steps = 1,560
batch schedule = 1,200 × 64 + 360 × 63
```

这与 P3-E10 的 exposure 和 step 完全相同，不允许选择中间 checkpoint。

训练命令：

```bash
.venv-train/bin/python -m \
  chemvector.training.atoms.train_g3_p3e14_style_count_scale
```

## 预注册门

- dev Count exact ≥ 50%；
- confirm Count exact ≥ 50%；
- confirm MAE ≤ 1.0；
- historical route100 Count exact ≥ 40%；
- route100 不参与训练或 checkpoint 选择。

## 结果

| 集合 | P3-E10 exact / MAE | P3-E14 exact / MAE | 变化 |
|---|---:|---:|---:|
| style90932 | 34,012/90,932 / 1.841 | **38,319/90,932 / 1.538** | +4,307 / -0.303 |
| dev256 | 103/256 / 1.406 | **120/256 / 1.234** | +17 / -0.172 |
| confirm256 | 108/256 / 1.270 | **117/256 / 1.078** | +9 / -0.191 |
| historical route100 | 37/100 / 1.570 | **37/100 / 1.280** | +0 / -0.290 |

四项正式能力门仍未全部通过：dev/confirm exact 未到 50%，confirm MAE 未到 1.0，route100
exact 未到 40%。fixed step、exposure、全唯一图覆盖与重复评估全部通过。

## 解释

数据规模假设得到直接支持。相同计算暴露下，dev、confirm 和三个集合的 MAE 都改善；这不是
增加 epoch 或梯度步带来的收益。P3-E14 在新训练集自身只有 42.14% exact，远低于 P3-E10
对 3,316 张反复数据的 84.4%，说明 90,932 张近似单遍仍明显欠拟合。

这也说明此前“训练数据不够”的说法需要更精确：旧 829 个 molecule 确实限制 Count 泛化，
但单纯把唯一数据扩到 22,733 个 molecule、保持原总计算量，还不足以越过能力门。下一变量
应当是 exposure，而不是继续改 Count loss、曲线窗口或网络结构。

## 决策

- 部署能力：`NO-GO`，不接 router、前端或端到端 SDF；
- 数据规模因果：`GO`；
- P3-E14 在固定 dev/confirm 上全面优于 P3-E10，route exact 持平且 MAE 更低，成为新的 Count
  研究 checkpoint；
- 下一轮 P3-E15 固定同一 90,932 图与模型，从 P3-E14 optimizer 连续训练到总计 3 个完整
  unique-data epoch，仍只看 fixed-final，不选择中间 step。

## 产物

- checkpoint：`step-001560.pt`；
- SHA256：`8dd295bc1e1b976c6e851cfa566ce990a1971726c23eebd48880450c4a809687`；
- 训练脚本、manifest、evidence cache 均可确定性重放；
- P3-E14 相关单元测试：6 项通过；
- 文档复核日期：2026-08-04。
