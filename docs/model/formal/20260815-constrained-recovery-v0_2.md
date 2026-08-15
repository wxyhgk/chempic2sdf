# 受约束恢复 v0.2：红队修订规格

日期：2026-08-15

状态：`PASSED_FINITE_SPECIFICATION_ONLY`
真实图片验证：`not_run`
训练授权：`false`

## 1. 为什么 v0.1 是 NO-GO

v0.1 证明了若干正确的条件式命题，但接口仍允许以下空洞实现：

1. `supported(a,b)` 只是一个 Prop，未要求具名、非空的 pixel/segment 证据，也未冻结最小 support score；
2. `BondBundle.endpoint` 忽略 lane 参数，因此“所有 lane 共享端点”是由定义直接写死，不是 lane 自身证据；
3. 输出 graph 使用旧匿名边格式，丢失 primitive、carrier、lane geometry 和 owner provenance；
4. 候选集合只有一个图类时，不需要完整性证书即可返回 Graph；
5. 任意手工填写的分数区间只要分开，就能授权 Graph，不需要 epsilon 的校准记录；
6. 没有直接锁死 `A-B-C` 的零墨迹 shortcut、fan 外围 chord 和一根线冒充双/三键。

所以 v0.1 文件原样保留，作为红队 NO-GO 证据，不再修改或用于授权训练。

## 2. v0.2 的类型边界

### 2.1 CarrierEvidence

每个 carrier 必须包含：

- 非空 `carrierId` 和 `primitiveId`；
- 非空、具名且无重复 ID 的 support pixels；
- 非空、具名且无重复 ID 的 support segments；
- `supportScore`、`minimumSupportScore` 及 `minimum <= score` 的证明；
- 唯一 primitive owner：`bondLane / glyphStroke / marker`。

Python 接口允许解析无效外部输入，但 `BondLane` 构造器会拒绝零 support、低 support、owner 不匹配、
primitive 不匹配或 geometry 未被 support segment 点名的 carrier。

### 2.2 BondLane 与 BondBundle

每条 lane 自己保存：

```text
primitiveId
geometry
CarrierEvidence
bondId / owner
beginNode / endNode
```

`endpoint(lane, side)` 读取该 lane 的端点，不再忽略 lane。一个 order-2/order-3 bundle 必须恰好有 2/3 条
lane；所有 lane 共享语义端点和 bond owner，但 primitive ID 与 support token 必须互斥。

这不是按 3px、3.5px、4px 猜键级。线宽只可能成为视觉 scorer 的输入；语义键级由一个有完整 lane
provenance 的 bundle 候选表示。

### 2.3 Graph provenance

每条输出边保留：

```text
begin / end / order / bond_id
primitive_ids[]
carrier_ids[]
lane_provenance[]
```

每个 lane provenance 继续包含 geometry、完整 carrier support、owner 和 primitive。三组数组的长度必须
等于 order。整图还检查 primitive、carrier 和 support token 不得被不同边重复消费。

旧格式 `{begin,end,order}` 即使化学图看起来正确，也不能作为 v0.2 的 Graph 输出。

## 3. Graph 输出的双证书

### 3.1 Candidate coverage

没有 `CandidateCoverageCertificate` 时：

```text
候选非空 -> Ambiguous
候选为空 -> Reject
```

coverage 证书绑定 evidence SHA256、enumerator、coverage audit、完整 candidate ID 集、支持解释 ID 集和
candidate-set bytes SHA256。Lean 的 `DecodeDecision.graph` 只能携带有 coverage authority 的
`CertifiedGraphOutput`。

这仍是规格接口，不代表当前测试证书已经证明真实图片候选完备。生产证书必须由独立 enumerator/auditor
生成，不能由模型进程自签。

### 3.2 Calibration 与 interval winner

多图类排序必须同时提供：

- 非空 calibration ID 和固定 SHA256；
- 正样本数；
- `maximum_observed_error <= epsilon`；
- 每个 candidate 的 observed score；
- 从该 score 和 epsilon 唯一计算出的 `[max(0,s-epsilon), s+epsilon]`。

调用者不能自由填写 lower/upper。只有一个图类的上界严格小于其他类的下界时才输出 Graph；区间接触
或重叠均为 Ambiguous。Lean 继续证明干净间隔严格大于 `2 * epsilon` 时，校准误差内赢家保持。

## 4. 红队反例结果

新增 16 项测试，全部通过：

- `A-B-C` 的 `A-C` zero-support shortcut 无法构造 BondLane；
- fan 外围 chord 低于 support threshold，无法构造 BondLane；
- 一根 physical stroke 不能构造 order-2/order-3 bundle；
- order-2/order-3 只接受精确 2/3 条、不同 primitive 的 lane；
- lane endpoint 确实随 lane 实例变化；
- 不同 lane 或不同边不能重复消费 support/primitive；
- Graph 边完整保留 primitive/carrier/lane provenance；
- 旧式无 provenance 边不能输出 Graph；
- singleton 没有 coverage 证书仍为 Ambiguous；
- 被篡改的 candidate-set hash 不能授权 Graph；
- 任意区间没有 calibration 时不能授权 Graph；
- 伪造为非 epsilon 派生的区间不能授权 Graph；
- 覆盖证书与校准间隔均有效、且严格分离时，有限规格允许 Graph；
- `2 * epsilon` 边界仍为 Ambiguous。

机器结果：

`formal/chemgraph/audit/results/constrained_recovery_v0_2/validation.json`

Lean 模块：

`formal/chemgraph/ChemGraph/ConstrainedRecoveryV02.lean`

## 5. 结论与下一门

v0.2 修复了 v0.1 的规格漏洞，但没有让真实分子识别突然可用。下一步只能做冻结图片上的候选覆盖审计：

1. proposer 只读最终图片，输出具名 CarrierEvidence；
2. GT 只在独立 verifier 中检查每条真实 Bond 是否至少有一个完整 bundle 候选；
3. 分别报告 zero/low support 假边、漏 lane、重复 ownership 和 provenance 丢失；
4. 在未参与调参的 style 与 MR-TADF 诊断集上估计 calibration epsilon；
5. 未达到预注册 coverage 门或没有独立 calibration 时，不开始 scorer 训练。

当前结论：`v0.2 finite specification GO`，`real-image recovery NO-GO/not run`，训练仍暂停。
