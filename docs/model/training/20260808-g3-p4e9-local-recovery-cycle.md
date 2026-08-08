# G3 P4-E9/P4-E11 局部候选恢复周期

> 日期：2026-08-08
>
> 状态：执行中，P4-E9-0 Oracle 已完成；仅 ADD_REFINE 进入证据门

## 目标

P4-E8证明，P4-E3 top128 在6px下的残余表示缺口只有两类：`2 GT / 1 candidate`共享峰，
以及`1 GT / 0 candidate`近距离漏峰。本周期不再扩大候选预算，也不修改或删除原top128，
而是验证两个小型、保守、只追加候选的恢复头：

- `ADD_SPLIT`：从严格`2 GT / 1 candidate`局部分量追加第二个anchor；
- `ADD_REFINE`：从6px不可达、但在12px内有唯一规范来源的top128 candidate追加anchor。

该设计只学习一般局部图像证据，不使用MR-TADF特征，不使用像素聚类、键长删除或固定几何修复。

## P4-E9-0 数据与 Oracle

固定P4-E3 checkpoint和top128。pilot20k按scaffold整体分为70% fit、15% calibration、
15% holdout；scaffold先按`SHA256(seed:scaffold)`排序，再按归一化剩余缺口确定归属。

Oracle仅用于生成训练标签：

1. 在6px图中找严格`2 GT / 1 candidate`缺口分量；原candidate保持不动，追加anchor监督为
   最大基数匹配后唯一未匹配GT，标签为`ADD_SPLIT`。
2. 在6px图中找严格`1 GT / 0 candidate`缺口分量；若一个top128 candidate与该GT距离
   位于`(6,12]px`，按`(距离, source_id)`选择规范来源，标签为`ADD_REFINE`。
3. 每个base source最多一个动作；冲突时split优先，refine GT与剩余source按最大基数、最小距离
   做整图一一匹配。每路至少需要256个动作、100个分子并覆盖四种variant，才允许训练。
4. 冻结Bond endpoint证据只有在训练域三折probe中AUC至少提高0.02，且动作目标endpoint
   coverage至少90%时才能加入。否则恢复头保持纯图像输入。
5. 不满足两条严格Oracle定义、但处于缺口分量或竞争refine来源中的candidate显式标为
   `UNSUPPORTED`，训练loss必须屏蔽；只有明确的普通candidate可以作为`NONE`负例。

Oracle不会进入推理组件，也不会读取`confirm_v2`、外部集或封存集。

## P4-E9-0 实际结果

pilot20k严格得到3,500/750/750个fit/calibration/holdout分子，对应14,000/3,000/3,000
张图；三个split的scaffold重叠均为0。506个动作全部逐个重放为6px matched `+1`，同source
互斥且`UNSUPPORTED`已显式写入。

| 路线 | 总动作 | 分子 | fit动作 | calibration | holdout | 容量门 |
|---|---:|---:|---:|---:|---:|---|
| ADD_SPLIT | 144 | 72 | 112 | 7 | 25 | NO-GO |
| ADD_REFINE | 362 | 294 | 264 | 31 | 67 | GO |

`ADD_SPLIT`同时未达到256动作和100分子，fit正例也不足固定capacity所需的128，因此本周期不
训练P4-E9A，不创建伪checkpoint或tag。`ADD_REFINE`三项容量门全部通过，进入训练域三折Bond
endpoint evidence probe；只有该probe完成后才决定P4-E9B的最终输入通道。

Oracle audit SHA256：
`5dc3189f7901a42edcc0d97b27cd43c8e8bb47b33573db4605d560cce0179698`；运行655.35秒。

P4-E8的58个缺口来自top1024，用来确定残差分量的容量上界；P4-E9部署预算仍固定top128。
其中P4-E8的28个shared source最高rank为117，全部确实位于top128，因此split专项门仍可重放，
但训练数据容量门已经先失败。

另一个边界必须明确：P4-E3曾用完整pilot20k训练，所以本轮calibration/holdout只衡量新增恢复头，
不作为冻结骨干的无偏泛化证据。旧dev仍是未见分布硬门，最终结论只由未读的confirm_v2确认。

## 公开推理协议

- 原top128 candidate绝不删除、移动或改ID；
- split ID：`589824 + base_source_id`；
- refine ID：`1179648 + base_source_id`；
- split offset：`base + 8px * tanh(offset)`；
- refine offset：`base + 12px * tanh(offset)`；
- 每路最多追加16个，合计最多32个；
- 不做距离去重或像素聚类。

## 模型与训练门

两个pilot模型分别命名为：

- `chemvector-g3-p4e9a-local-binary-split-pilot20k`；
- `chemvector-g3-p4e9b-structural-anchor-refine-pilot20k`。

恢复头输入32x32灰度crop、32x32中心分数crop、32维candidate feature、归一化score/rank，
参数少于10万。action bias初始化为-6，offset末层零初始化。capacity固定128正例、384 hard
negative、2000步；pilot固定5000步、batch128、AdamW、学习率`3e-4`、weight decay
`1e-4`、clip 1，正负采样比1:3。

阈值只在calibration选择，顺序固定为：零无用追加、恢复GT最多、动作数最少、阈值最高。
pilot必须在calibration、holdout和旧dev全部零无用追加，恢复至少50%本路Oracle；split还必须
恢复P4-E8的28个共享缺口中至少14个，并使top128 matched/exact净增。重复dev必须完全确定。

通过路线才允许从冻结P4-E3重新训练full80k，模型名固定为
`chemvector-g3-p4e10-local-recovery-full80k`。full80k按scaffold做90/10 fit/calibration，训练
10000步，不从pilot热启动。

## Confirm 纪律

最终winner、组件SHA和阈值全部冻结后，`confirm_v2`只允许读取一次。运行前验证commitment和
manifest，运行后恢复`000`权限。要求零无用追加、至少恢复40% confirm Oracle增益、exact净增、
各variant均无错误追加，并通过确定性重复。失败也不重新选择阈值。

每个实际训练出的模型，无论GO或NO-GO，都必须提交记录并创建与模型名完全一致的annotated
Git tag。本周期不接ChemScene、SDF或前端。
