# G3 P3-E27 Query-slot Occupancy Micro160

> 日期：2026-08-07
>
> 状态：预注册，待训练
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

