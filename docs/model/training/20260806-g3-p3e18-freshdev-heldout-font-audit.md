# G3 P3-E18 Fresh Dev 跨字体固定审计

> 日期：2026-08-06
>
> 状态：零训练审计 NO-GO，断点定位完成
>
> 模型：P2-D candidate + P2-W base + P3-E1 expert + P3-E16 Count + P3-E7 router
>
> 输出：`outputs/audits/chemvector-g3-p3e18-freshdev8k-heldout-font-audit`

## 目标

P3-E17 在原有 route_confirm 上接入 P3-E16 Count 后明显提高，但这些小集合不能证明能力能跨
字体泛化。本轮冻结全部组件和路由阈值，对 `fresh_eval_12k/fresh_dev` 的 8,000 张 held-out
字体图片做一次无选择审计。`sealed_test` 保持 `000` 权限，不读取、不枚举、不解锁。

## 数据与协议

- fresh_dev：2,000 个全新分子，每个分子四种布局，共 8,000 图；
- variant：coordgen、original、rdkit、stress 各 2,000 图；
- Count 区间：1–20、21–40、41–60、61–80 各 2,000 图；
- manifest 按两个 fresh_dev tar 的物理顺序构建；
- manifest SHA256：`aa6d0254c5b0d8911972b8377a0628c2e0e3fbc72b95af95c9d8807591f4ab0d`；
- 所有 checkpoint、P3-E7 阈值和最终 Gate 在运行前固定；
- 不训练、不调阈值、不选择 checkpoint，不使用 MR-TADF。

预注册门：candidate recall不低于99%，Count exact不低于40%，Count MAE不高于1.5，
known-count Router exact不低于30%，predicted-count Router exact不低于20%。

## 结果

| 指标 | 结果 | 门 | 判定 |
|---|---:|---:|---|
| candidate Atom recall | 99.6877% | ≥99% | 通过 |
| candidate 整图可达 | 7,410/8,000，92.625% | 诊断 | - |
| Count exact | 2,901/8,000，36.2625% | ≥40% | 失败 |
| Count MAE | 1.9808 | ≤1.5 | 失败 |
| known-count Router exact | 2,467/8,000，30.8375% | ≥30% | 通过 |
| predicted-count Router exact | 1,477/8,000，18.4625% | ≥20% | 失败 |

因此总决策为 `freshdev_heldout_font_no_go`。这不是候选点主线整体失效：给真实 Count 时，冻结
Router 已过30%门；predicted 结果下降990张，严格来自 known exact 与 Count correct 的交集损失。

## 分桶

### Count 按布局

| variant | exact | MAE | known Router exact | predicted Router exact |
|---|---:|---:|---:|---:|
| coordgen | 41.55% | 1.627 | 760/2,000 | 468/2,000 |
| original | 40.70% | 1.763 | 720/2,000 | 432/2,000 |
| rdkit | 38.35% | 1.784 | 701/2,000 | 417/2,000 |
| stress | 24.45% | 2.749 | 286/2,000 | 160/2,000 |

### Count 按原子数

| 原子数 | fresh_dev exact / MAE | style90k 训练域 exact / MAE | style90k 图数 |
|---|---:|---:|---:|
| 1–20 | 55.80% / 0.681 | 52.22% / 0.768 | 11,756 |
| 21–40 | 52.55% / 0.725 | 53.17% / 0.715 | 66,020 |
| 41–60 | 27.05% / 1.711 | 26.87% / 1.971 | 9,836 |
| 61–80 | 9.65% / 4.807 | 12.10% / 5.027 | 2,372 |

同一区间的训练域与 held-out 字体域结果接近，说明总准确率下降主要来自 fresh_dev 的四个
Count 区间等量，而 style90k 有72.6%的图片集中在21–40原子。当前模型学到了训练先验，
高 Count 能力没有因六次遍历自动补齐。不能把这次失败简单归因于字体或候选点。

candidate 的 stress 整图可达率为81.85%，61–80原子为81.75%，但对应原子级 recall 仍分别为
99.226%和99.471%。大图通常只漏少数点；这是后续候选专项，不是本轮 Count 暴跌的主要解释。

## Router 残差

known Count 下，base exact为259，expert exact为3,499，冻结 Router 为2,467；Router 仅7次
break，但也没有采用约1,000张可恢复的 expert 结果。Count正确的2,901张中，expert做对2,181，
Router只做对1,477。P3-E7小样本上选择的极保守单阈值跨域仍安全，但利用率不足。Count仍是
第一断点，Router是第二断点，两者必须分开处理。

## 决策

- P3-E18 总 Gate：`NO-GO`；
- candidate 与 known-count Router 两个子门通过，保留现有冻结组件；
- 停止继续追加同分布 Count epoch，也不因 aggregate 失败回头重做候选表示；
- 下一步 P3-E19 使用已有 style90k 冻结 evidence，预注册按 Count 区间平衡的训练协议；
- fresh_dev 不作为训练数据，不按其中间结果选 checkpoint，只允许 fixed-final 复核；
- Router 改进排在 Count 之后，且需要新的独立 route split，不能在 fresh_dev 上重新选阈值。

## 产物校验

- metrics：`outputs/audits/chemvector-g3-p3e18-freshdev8k-heldout-font-audit/metrics.json`；
- metrics SHA256：`ebe02d344fb2ea8fc5fee977239609a3e208143099b2cb8a96196e0eefcc0ab7`；
- candidate cache SHA256：`9d5f8f4ec14e0a43c62f5db7e411b1d9e4716692d22b77d8c07a917231bef09d`；
- 相关测试：25项通过；完整本轮相关回归在提交前复跑；
- 本轮无新 checkpoint，不创建模型 tag；
- `sealed_test_read=false`；
- 文档复核日期：2026-08-06。
