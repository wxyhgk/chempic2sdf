# G3 P3-E21 五路线完整 SDF 验证周期

> 日期：2026-08-07
>
> 状态：Oracle Gate NO-GO，周期停止
>
> 周期：`chemvector-g3-p3e21-five-route-sdf-cycle`
>
> 基线：P3-E16 Count + P3-E7 Router

## 目标

在同一几何缓存、数据划分、训练预算和冻结下游上，比较五种 Atom/Count 表示，并将每个候选
都接入 v0.10.10 与 v0.10.21 两套 ChemScene/SDF 链。只有一个固定 winner 可以运行一次
sealed_test；若无人通过完整门，不从失败模型中强选。

## 五条路线

1. `hard_latent_occupancy`：动态一一匹配 presence + Poisson-binomial Count；
2. `soft_coverage_occupancy`：兼容集合 soft coverage + 分量质量约束；
3. `query_slot_occupancy`：query presence + pointer-compatible 唯一匹配；
4. `bond_conditioned_occupancy`：在 hard latent 上只增加冻结入射 Bond evidence；
5. `hybrid_structured_count`：结构 occupancy posterior 与冻结 P3-E16 posterior 的可学习凸融合。

五路统一输出 `AtomSelectionResult`。推理时只能读取图像衍生 evidence，不得读取 GT、split、字体、
scaffold 或渲染参数。等价候选的正例每步动态求解，不固定 candidate ID。

## 固定数据

从 style90k 按 Count 分桶，并做 molecule、InChIKey、scaffold 跨 split 隔离：

| split | 分子 | 图片 | 每个Count区间分子 |
|---|---:|---:|---:|
| train | 800 | 3,200 | 160 |
| dev | 160 | 640 | 32 |
| confirm | 160 | 640 | 32 |

Count区间为1–20、21–40、41–60、61–80、81–100；每个分子的四种variant保持完整。

固定 manifest：

- train：`a410bdfbaff8ca2a0c6c5d3b43cd56bb4e9fe1e86dc18ed04bd2f81118b0b2a0`；
- dev：`a33a3a364013e365777f1538af17d75b59371666c6aa36659be0c1701201f79c`；
- confirm：`5fa44672546e830176c1ee1498062a69d17b9ebd041e83d212552a7eb206acfd`；
- 三组 scaffold overlap：0；
- `fresh_dev`、USPTO、MR-TADF 和 `sealed_test` 均未用于上述选择。

## 共同下游

ChemScene 新增 external junction 输入，只替换 junction 坐标。OCR附着、BondObject、图修复、
价态校验和SDF编译保持冻结。五路都运行v0.10.10与v0.10.21；相同junction输入必须逐位复现。

完整指标包括Count、Atom 6px、Atom set exact、SDF valid、Skeleton exact、Atom/Bond typed exact、
non-stereo exact、attributed-SDF Chemical exact，以及paired rescues/breaks。

## 阶段门

### Oracle / Adapter

- train/dev/confirm candidate Atom recall均不低于99%；
- external adapter在相同junction输入下逐位一致；
- GT junction能够通过两套冻结下游生成完整诊断结果。

### Micro Capacity

- fully-representable 160图上Count exact和Atom set exact均不低于95%；
- presence正负概率间隔不低于0.8；
- 少于两路通过则周期停止。

### Pilot Dev640

- 相对P3-E16：Count exact和Atom set exact各提高至少5pp；
- Atom recall不下降，61–100 macro Count提高至少3pp；
- 两套SDF valid各回退不超过1pp，Skeleton均不下降；
- Chemical exact rescues不少于breaks；
- 固定排序后只取前两路扩量。

### Top-2 Confirm640

- Count/Atom set exact相对基线各提高至少8pp；
- 任一Count区间回退不超过3pp；
- 两套SDF valid均不下降；
- 两套合计Skeleton提高至少2pp、Chemical提高至少1pp，且每套rescues不少于breaks。

winner按最差profile Chemical增量、Chemical总增量、最差Skeleton增量、Atom set exact、Count
MAE、CPU延迟依次排序。

## 外部与 sealed

winner固定后才运行fresh_dev8k、USPTO clean1k和MR-TADF诊断。fresh需Count exact不低于40%、
MAE不高于1.5、predicted Atom set不低于20%，且双profile SDF不回退。全部通过后先提交代码、
checkpoint SHA和同名tag，再临时解封sealed_test4k单次运行；失败不测试第二名，并恢复000权限。

## 当前进度

- 周期接口与scaffold隔离manifest：完成；
- external junction adapter及协议测试：完成；
- 共享geometry/query/count缓存：完成，约477MB；
- Candidate Oracle Gate：失败；
- 五路训练和完整SDF：按预注册停止规则未执行。

## Oracle Gate 结果

| split | Atom recall | 整图可达 | 99%门 |
|---|---:|---:|---:|
| train | 98.8009% | 79.13% | 失败 |
| dev | 98.9290% | 79.84% | 失败 |
| confirm | 99.1065% | 83.75% | 通过 |

失败集中在大分子而非普通区间。train/dev/confirm的81–100原子 recall分别为97.47%、97.87%、
98.11%；1–60原子各桶则约为99.5%–99.9%。stress在train/dev分别为98.13%、98.25%。top128
候选池在高Count和stress图中已被重复峰占满，后续selector无法恢复池中不存在的GT。

因此本周期判定 `candidate_oracle_no_go`。不修改99%门，不训练五个route，不运行SDF来掩盖
上游不可达。下一周期必须先验证top192/256或新的高Count候选表示，在train/dev/confirm三组
同时达到99%后，才可原样恢复本周期的五路线部分。
