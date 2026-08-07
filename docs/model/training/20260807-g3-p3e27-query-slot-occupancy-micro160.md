# G3 P3-E27 Query-slot Occupancy Micro160

> 日期：2026-08-07
>
> 状态：已完成，Micro Capacity NO-GO
>
> 模型：`chemvector-g3-p3e27-query-slot-occupancy-micro160`

## 目标

P3-E25/E26在冻结base/expert evidence上新增candidate occupancy头，known-count exact仅47/49，
新增可达均0/27。本轮实现第三条`query_slot_occupancy`：训练128个query直接读取true-global192
candidate memory，让query existence、pointer、坐标与Count联合适配tail。

## 固定输入与初始化

- 完全复用P3-E25的Micro160 sample digest
  `06f44d30a88013643be6041fe55741fd75f6d58bcf1d875a22224972a7944501`；
- true-global192 tensor digest
  `274222ec409a6f90999976cbb7898af6ca4d8dca9c5597fcea28c515353d8a9a`；
- 从P3-E7 expert checkpoint
  `7880bc31c906c4afbbfd7265188c68921a115329aa795903ccbbac07c3529a3b`
  初始化，原state dict严格加载，只把config candidate数128改为192；
- 128 query、192 candidate、129 Count classes不变；全部query-set参数可训练。

不使用P3-E25/E26 occupancy checkpoint，不读取dev/旧confirm/fresh/sealed，不运行SDF。

## 损失与解码

每步先按每个query在GT兼容候选集合上的pointer log-probability质量，加existence成本，做
query↔GT Hungarian。损失固定为已有`g3_atom_query_set_loss`：

- matched query existence BCE，unmatched query权重0.1；
- matched query到其GT兼容候选集合的pointer log-sum-exp；
- pointer期望坐标L1，权重5；
- 129类Count CE；
- 最弱positive query与最强negative query的softplus boundary，权重1、margin1。

不固定candidate identity，不对兼容替代峰逐个设负，不启用unique-pointer训练标签。解码使用
existing joint fixed-cardinality Hungarian：Count head给K，同时选择K个query和K个唯一candidate；
另报known-K。

## 训练预算

- batch4、100个完整epoch、4,000 steps、16,000 exposures；
- AdamW lr=3e-4、weight decay=0.01、grad clip=1；
- BF16、seed=20260827、fixed-final-only；
- 不保存/选择中间checkpoint，不追加同协议epoch。

## Micro门

- Count exact≥152/160且MAE≤0.10；
- predicted/known-count Atom set exact各≥152/160；
- predicted Atom recall≥99.9%；
- existence boundary exact≥152/160；
- 新增可达图片predicted Atom exact≥50%；
- 五Count桶、四variant的Count/Atom exact各≥90%；
- Count≤128、pointer candidate唯一、两次评估逐项一致、固定步数/exposure通过。

全部通过为`query_slot_occupancy_micro160_go`，否则NO-GO。无论结果如何都提交checkpoint和结果，
并创建与模型名完全一致的annotated tag。至少两条路线过Micro前，不解除P3-E21完整SDF门。

## 实际结果

固定4,000步完整运行。Count容量完全通过，但Atom集合显著失败：

| 指标 | 结果 | 门 |
|---|---:|---:|
| Count exact / MAE | 160/160 / 0 | ≥152 / ≤0.10 |
| predicted Atom exact | 11/160 | ≥152 |
| known-count Atom exact | 11/160 | ≥152 |
| predicted/known recall | 92.21% | ≥99.9% |
| existence boundary exact | 0/160 | ≥152 |
| 新增可达 exact | 0/27 | ≥50% |

Count loss降到累计均值0.417并最终全对，证明模型容量、数据读取和optimizer能记住Micro全局数量；
pointer loss和coordinate loss也持续下降，但动态pointer-compatible query身份没有形成全图稳定存在性，
最终最弱正query未在任何图片上超过最强负query。相对两个candidate occupancy头，query-slot甚至将
known-count F1从约95.8%降到92.2%。

正式判定`query_slot_occupancy_micro160_no_go`，不扩量、不运行dev/SDF。下一步只允许评估
`bond_conditioned_occupancy`是否有冻结、纯图像衍生的incident Bond evidence可合法接入；不得用
GT bond/render layer冒充forward特征。`hybrid_structured_count`不能修复known-count Atom exact，
因此在selector不变时受本轮结果支配。

## 产物

- checkpoint SHA256：`3aa13908f5682b7fc3a21c6eb677c24a32c4c5012166a2aa7c2de5e6462ef6bd`；
- audit SHA256：`156aeecb69fa1df6c48983dc6709804941b9b508a2060fe56a5495c810916f18`；
- final metrics SHA256：`760414e2dbf4711bccf5cf818ecad63de1633ef4d6a0045076d72f4e6199c623`；
- 训练耗时：246.56秒；
- dev、旧confirm、fresh和sealed均未读取。

结果提交后创建同名annotated tag：
`chemvector-g3-p3e27-query-slot-occupancy-micro160`。
