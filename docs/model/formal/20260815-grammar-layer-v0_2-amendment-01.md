# Grammar Layer v0.2 预注册修正案 01

日期：2026-08-15

状态：`PREREGISTERED_AMENDMENT`。本修正案在 v0.2 实现结果产生前冻结；原预注册文件保留，
不得覆盖。

## 发现的问题

原协议同时要求：

1. prior 是 observation-class 级函数，不依赖 Scene ID 或 GT graph；
2. prior 在 crossing/junction 或 double/parallel 的同 observation 类中只保留一种解释；
3. prior 保持 observation coverage。

如果 prior 被表示为 `P : Scene -> Prop`，并要求它对 render 相同的 Scene 取值相同，那么三项
不能同时成立。对碰撞 Scene `s,t`：

```text
render(s) = render(t)
meaning(s) != meaning(t)
```

coverage 要求这个 observation 至少保留一个 Scene。render-invariance 随即强制另一个 Scene 也
被保留，因此碰撞仍在，无法恢复 identifiability。若两个都删除，则违反 coverage。

所以“纯 observation-invariant 的 Scene predicate prior 可以消歧”是错误命题，必须废止。

## 修正后的对象

将 Oracle Field 拆成两个有来源标记的对象：

```text
VisibleEvidence
  = extract(productionObservation)

ProtocolInterpretation
  = choose(VisibleEvidence, enumerateCandidates(VisibleEvidence), priorId)
```

类型接口：

```text
extract             : Observation -> Evidence
enumerateCandidates : Evidence -> CandidateSet Graph
choose              : Evidence -> CandidateSet Graph -> PriorId -> DecodeResult Graph
```

`choose` 的生产 forward 禁止读取 Scene、meaning、graph signature、collision class、SDF 或
benchmark truth。审计时可以定义：

```text
agreesWithChoice(s) :=
  choose(extract(render(s)), enumerateCandidates(extract(render(s))), priorId)
    = Graph(meaning(s))
```

这个 Scene 级谓词只用于测量某个公开约定选择了哪些真值，不能反过来作为 chooser 的实现。

策略 B 的准确表述也随之改变：它不是“从相同像素恢复了隐藏真值”，而是“将绘图语言规范化为
每个 Evidence 只接受一个公开解释”。与该约定冲突的合法原图会被判为协议外，而非识别成功。

## Factorization 门

任何声称可由图片预测的 Oracle evidence 必须先证明：

```text
exists extract : Observation -> Evidence,
  forall s in supportedDomain,
    extract(render(s)) = oracleEvidence(s)
```

如果两个同 observation Scene 的 `oracleEvidence` 不同，则由碰撞反定理可知不存在这样的
`extract`。因此以下字段不能放进 `VisibleEvidence`：

- GT lane ownership；
- GT Bond bundle ID；
- GT Atom/Bond endpoint identity；
- crossing/junction 的 GT semantic kind；
- canonical graph signature。

它们只能出现在 `ProtocolInterpretation`、候选解释或审计 sidecar 中。可见像素中的 line carrier、
几何 terminal hypothesis、intersection 和 marker observation 可以进入 `VisibleEvidence`，前提是
其 canonical 定义对同 production observation 给出相同结果。

## 修正后的 Lean 门

除原门外必须证明：

1. render-invariant Scene predicate + coverage 不能解决一个已有碰撞类。
2. `oracleEvidence(s) != oracleEvidence(t)` 且 `render(s)=render(t)` 时，不存在
   `Observation -> Evidence` factorization。
3. conservative decoder 对 candidate graph 数量大于 1 时不输出 Graph。
4. convention chooser 是 Observation/Evidence 的函数，因此同 observation 输出相同结果；它只
   能改变任务协议，不能同时匹配碰撞类中的两个 truth。

## 修正后的可执行门

- `VisibleEvidence` canonical bytes 对同 production observation 必须 100% 相同。
- Evidence factorization mismatch 为 0，否则 Oracle 表示 `NO-GO`。
- 候选生成 completeness 与 precision 分开报告；不得只保留 chosen candidate 后冒充完整。
- 策略 A 对 candidate graph 数大于 1 的 observation 全部输出 `Ambiguous`。
- 策略 B 必须报告 truth agreement、truth exclusion 和每类风险，不得只报告 survivor collision=0。
- hash 查表可以验证有限域定理，但必须标为 `audit_oracle_only=true`、
  `deployable_from_opaque_observation=false`。
- 只有 image-computable `extract/enumerateCandidates` 建立后，才允许训练 evidence predictor；
  训练仍不等于授权 graph decoder。

## 对下一步的影响

v0.2 当前最高优先级从“Oracle field 到图的左逆”改为：

```text
Observation
  -> VisibleEvidence factorization
  -> complete CandidateSet
  -> Ambiguous 或公开 convention choice
```

只有前两步不读取 GT 且通过，后续模型训练才具有正确的可观测目标。
