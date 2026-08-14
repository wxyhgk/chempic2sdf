# Raster Evidence v0.3 Hidden Challenge 红队协议

日期：2026-08-15

状态：`REDTEAM_PREREGISTERED`。本文只冻结挑战协议；尚未选择或提交 secret seed，未生成 challenge、
reveal、GT、预测或结果。训练未授权。

## 1. 审计目标与声明边界

本挑战只在 future extractor、candidate generator 和 conservative decoder 全部冻结后运行，回答：

```text
opaque PNG/tensor bytes
  -> neutral visible Evidence
  -> complete CandidateSet
  -> Graph(unique) | Ambiguous(kind)
```

它不允许用 lane IoU、重建图相似度或少量示例替代候选集合正确性。挑战通过最多说明：冻结流水线在
本文定义的有限 hidden grammar 和有限 nuisance shell 上通过；不说明现实化学图片空间已经完备，
也不自动授权训练或生产部署。

以下对象必须分开：

- `InkProgram`：只描述最终可见墨迹，不含 Atom、Bond owner 或图签名；
- `Interpretation`：同一 InkProgram 的一个合法语义解释；
- `Raster`：canonical PNG bytes 及 production tensor bytes；
- `Evidence`：未解释 lane、terminal event、intersection、marker observation 和 stroke-role 候选；
- `CandidateSet`：从 Evidence 枚举的 canonical chemical graph 集合；
- `DecodeResult`：`Graph | Ambiguous | Reject`。

GT `Interpretation`、Scene ID、family、transform、seed、candidate graph、SDF 和 ambiguity kind 都不是
extractor 输入。

## 2. 威胁模型

挑战主动防御以下捷径：

1. 按 observation SHA、PNG bytes 或文件名查训练/审计表；
2. import Scene generator、SDF、Render JSON、grammar layer、collision certificate 或 canonical GT；
3. 从 PNG metadata、目录顺序、sample ID、进程状态或相邻重复样本推断 family；
4. 先输出单个 prior 答案，再用 RDKit sanitize 或 GT candidate membership 选择；
5. 把整张图片、hash 或任意 opaque payload 塞进 Evidence，延后查表；
6. 在 challenge seed、图片或标签暴露后修改代码、权重、阈值、schema 或依赖；
7. 只在 strict 样本输出全部候选，或在全部样本 Reject，以真空方式通过 soundness；
8. 利用不同 semantic witness 的渲染顺序、抗锯齿舍入或 metadata 破坏 exact collision。

至少一名 seed custodian、generator reviewer 和 evaluator reviewer 不得参与 extractor 实现。任一角色
重合必须记录并使本轮结果降级为非独立诊断。

## 3. 冻结顺序与 commitment/reveal

### 3.1 协议与 generator 冻结

在 extractor 冻结前提交：

- 本协议的 Git commit 和 SHA256；
- reference generator、固定点 rasterizer、canonical PNG serializer、production preprocessing 的
  source-tree SHA256；
- generator container digest、字体/字形模板 hash、PRNG 算法和输出 schema；
- challenge family 配额、拒绝采样谓词和最大尝试次数；
- 独立 oracle enumerator 和 verifier 的 source-tree SHA256。

generator 与 verifier 可以公开；seed 必须未知。公开 generator 不能削弱挑战，因为坐标空间足够大，
且最终 seed 在 extractor 冻结后才可导出。

### 3.2 两方 seed commitment

两个互不隶属 extractor 实现者的 custodian 分别从 CSPRNG 取得 256-bit `r1`、`r2`，只公开：

```text
C1 = SHA256("chemgraph-v0.3-share-1\0" || r1)
C2 = SHA256("chemgraph-v0.3-share-2\0" || r2)
```

commit 文件只含 `C1/C2`、协议 commit、generator commit、创建时间和签名，不含 seed 或 reveal 数据。
至少一个 share 在 extractor 冻结前保持未知，即可阻止实现者预计算 observation 表。

### 3.3 Extractor 冻结

冻结记录 `E` 必须覆盖：

- extractor、candidate generator、decoder 的完整源码 hash；
- 模型权重、tokenizer、配置、数值阈值和输出 canonicalizer hash；
- lockfile、wheel/container、系统库和 production preprocessing hash；
- 允许的文件读取白名单、网络关闭声明和命令行接口；
- `Evidence`、`CandidateSet`、`DecodeResult` 的严格 schema；
- `sorry/admit/axiom`、测试和构建状态。

冻结后任何一字节变化都产生新版本，旧 hidden challenge 结果作废。

### 3.4 Seed 导出与私有运行

在 `E` 冻结后，两方只向隔离 evaluator 揭示 share。最终 seed 使用冻结的 HKDF-SHA256：

```text
salt = SHA256(protocol_commit || generator_commit || E)
S    = HKDF-SHA256(r1 || r2, salt, "chemgraph-raster-evidence-v0.3")
SA   = HKDF-SHA256(S, salt, "hidden-pack-a")
SB   = HKDF-SHA256(S, salt, "hidden-pack-b")
```

Pack A/B 使用独立 domain separation，分布相同但样本不同。evaluator 先生成 image pack 和 sealed GT，
公开二者 commitment 及样本计数，但不公开 seed、GT、family、transform 或对应关系。

### 3.5 预测 commitment 与 reveal

extractor 对 A/B 完成后，先提交：

- 输入 pack commitment；
- canonical Evidence、CandidateSet、DecodeResult 的 Merkle root；
- stderr、退出码、资源和系统调用审计 hash；
- extractor freeze record `E`；
- 相同 pack 的第二次 replay commitment。

只有上述 commitment 固定后，才公开 `r1/r2`、完整生成 manifest、GT 和拒绝采样日志。独立 reviewer
从 seed 重建 A/B；任一 image、GT、顺序或 commitment 不一致即整轮 `REJECT`。

challenge reveal 后该 pack 永久退役，只能用于解释失败，不得用于调参后再次报告 hidden 分数。新版本
必须使用新的两方 share 和新 pack。

## 4. 隔离执行协议

每个 observation 在全新、无持久状态的容器中执行一次；重放样本也必须使用全新容器。extractor：

- 只从 stdin 接收 canonical PNG bytes；不接收路径、sample ID、family、尺寸 sidecar 或 batch index；
- 只向 stdout 写 canonical Evidence JSON；未知字段、NaN、无限值和 opaque byte/blob 字段均拒绝；
- 网络 namespace 为空，时钟、主机名、用户目录、仓库、`Data/` 和其他样本不可见；
- 文件系统只读，仅开放冻结 executable、权重、白名单依赖和当前 PNG；
- 禁止读取 manifest、SDF、Scene、seed、GT、digest table、collision certificate 和 evaluator 代码；
- 由 syscall trace/文件访问日志证明没有白名单外读取；
- 设定固定 CPU/GPU、线程数和随机状态；同 bytes 的 canonical 输出必须逐字节一致。

evaluator 在容器外将 Evidence 交给冻结 candidate generator 和 decoder。candidate generator 同样不得
接触 Scene/GT；GT 只由独立 verifier 在预测 commitment 后比较。

PNG 必须删除所有 ancillary chunk，只允许 `IHDR/IDAT/IEND`。challenge handle 由 evaluator 私下维护，
不会进入 PNG、stdin、环境变量或输出 schema。

## 5. Hidden grammar 与配额

每个 Pack 含 5,632 个唯一 observation class：4,096 个 strict，1,536 个 exact ambiguity。exact
ambiguity observation 各以完全相同 PNG bytes 隔离调用两次；另从 strict 中抽取 512 个做独立 replay。
因此每个 Pack 共 7,680 次 extractor invocation。

### 5.1 Strict families

下列每类恰好 512 个，且必须由独立 oracle enumerator 确认在冻结 challenge grammar 中
`|GraphSet(raster)| = 1`，不能只信 family 名称：

| ID | strict motif |
|---|---|
| S1 | 一根任意方向 single carrier，长度跨 short/medium/long 桶 |
| S2 | 非共线、非 180 度的 degree-2 angled chain |
| S3 | 三臂 fan junction，任意两臂不重合 |
| S4 | 四臂 junction，但不存在两组严格对向 perfect matching |
| S5 | triangle 或 quadrilateral cycle，允许长短边混合 |
| S6 | 两个互不相交、非共线、投影不完全对齐的 disconnected single bonds |
| S7 | 2--4 条边的 long/short mixed path 或 fan，禁止 through-atom 和 T-touch 歧义 |
| S8 | near-crossing 或 near-parallel chemical motif，所有 carrier 保持严格正间隙 |

strict acceptance 同时要求：无零长 lane、无重复坐标、无正长度共线重叠、无未登记 through-atom、
无同像 bundle 或 crossing 替代解释。若拒绝采样，attempt counter 和原因必须进入 reveal 日志；不得在
看到 extractor 输出后丢弃样本。

### 5.2 Exact raster collision families

每个 collision class 先生成一个共享 `InkProgram`，只 rasterize 一次，再附加两个或多个 hidden
Interpretation。禁止分别渲染 semantic Scene 后声称“看起来相同”。raw PNG bytes 和 production
tensor bytes 必须逐字节相同。

| ID | 每 Pack 数量 | 同像解释 |
|---|---:|---|
| C1 | 512 | 两根内部 proper-cross carrier；disconnected crossing vs 中心 degree-4 junction |
| C2 | 384 | 两条同投影平行 lane；一个 order-2 Bond vs 两个 disconnected single Bonds |
| C3 | 256 | 三条同投影平行 lane；一个 order-3 Bond vs 三个 disconnected single Bonds |
| C4 | 384 | `H/N/A/M/W/X/K/V` 或 `Cl/Br/Ph/CH3/NH2` stroke ownership vs bond-lane ownership |

C1--C3 进入核心 chemical CandidateSet exact 门。C4 是 stroke-role 可辨识性门：neutral Evidence 必须
保留两种 role hypothesis，并输出 `Ambiguous(text_or_lane)`；不得把 source text mask 当作唯一可见
真值。C4 不用于增加或减少 C1--C3 的 chemical graph denominator。

## 6. 坐标与外观生成协议

所有 family 按相同分层表平衡采样，防止 line width、AA mode 或方向泄露 family。若 family 数量
不能整除某个桶数，各桶计数只能相差 0 或 1；具体余数分配由 seed shuffle 决定，禁止手工指定。

### 6.1 固定点坐标

- logical canvas 为 `768 x 768`，所有几何使用 Q16.16 有符号固定点；
- base endpoint 从内部 `[64,704]^2` 的 Q16 网格抽样，不复用 v0.1 的固定 `{12,32,52}` 点阵；
- minimum carrier length 为 48 logical pixel，maximum 为 600；
- short/medium/long 桶分别为 `[48,128)`、`[128,320)`、`[320,600]`；
- 边界 margin 至少为 `max(24, 4 * line_width)`；不满足者按冻结 rejection rule 重抽；
- 除协议声明的 collision/replay invocation 外，不同 unique observation class 的 raw PNG SHA256 不得
  重复；其 production tensor digest 不得与 v0.1 commitment 中已知 observation hash 重合。

### 6.2 旋转、反射、缩放和平移

变换顺序固定为：base motif -> reflection -> rotation -> uniform scale -> translation。

rotation 不调用平台浮点三角函数，使用有理 Pythagorean 矩阵及其符号/坐标交换闭包：

```text
(cos, sin) in {(1,0), (3/5,4/5), (5/13,12/13), (8/17,15/17)}
```

reflection 从 `{none, x, y}` 分层选择；scale 从
`{1/2, 5/8, 3/4, 1, 5/4, 3/2, 2}` 选择。translation 从仍满足 margin 的 Q16 区间均匀抽样。
同一 collision class 的全部 Interpretation 必须共享变换后的同一个 InkProgram。

每个 family 的 observation 组成 metamorphic quartet：同一个 base InkProgram 生成 identity 和 3 个
不同的 rotation/reflection/scale/translation 版本。因所有 family 数量均可被 4 整除，S1--S8 各有
128 个 quartet，C1/C2/C3/C4 分别有 128/96/64/96 个。quartet 内 line width、cap/join、AA mode 和
glyph template 固定，只改变几何变换；quartet 之间再按 style bucket 平衡。这样 inverse-transform
Evidence 等变性有一一对应的审计对象，不允许拿不同随机 motif 做近似比较。

### 6.3 Line style

- final line width 从 `{3/4, 1, 3/2, 2, 5/2, 3, 4, 5, 6}` pixel 分层选择；
- cap 从 `{butt, round, square}` 选择；join 从 `{miter, round, bevel}` 选择；
- geometry scale 与 line width 独立采样，禁止用 width 推断 motif；
- 前景固定纯黑，背景固定纯白；v0.3 不测试颜色、透明背景或模糊；
- bundle lane gap 从 `{3/2, 2, 3, 4, 6, 8} * line_width` 选择，同时满足画布边界。

### 6.4 Anti-alias

AA mode 在下列三种模式中按“各桶计数差至多 1”平衡：

```text
A0: 1x binary center sample
A1: 4x4 integer supersample + exact box coverage
A2: 8x8 integer supersample + exact box coverage
```

coverage 到 uint8 使用冻结的整数 numerator/denominator 和 round-half-to-even；不得调用版本漂移的
隐式绘图库 AA。reference rasterizer 和 PNG compressor 的 container/source hash 在协议阶段提交。
exact collision 只渲染共享 ink 一次，因此在全部 AA mode 下必须保持 byte identity。

### 6.5 Text-like interference

glyph 使用冻结的 stroke template，不依赖系统字体发现。大小相对当前 motif median bond length 从
`{2/5, 3/5, 1, 3/2, 5/2}` 选择，rotation/reflection 与 chemical motif 独立。

C4 的共享 InkProgram 同时含一个 chemical motif 和 text-like strokes，并分层覆盖
`clear/near/touch/overlap/occlude`。即使 clear/near 的最终 AA 支撑不相交，黑色 `H/N/A` 仍可能被
解释为化学线图，因此所有 text-like ownership 样本都进入显式 role ambiguity，不伪装成 strict。
`near` 的最小几何 clearance 覆盖 `{1/2, 1, 2, 4} * line_width`；最终支撑是否相交由 frozen family
规则记录。OCR 未实现不能成为唯一选择 lane ownership 的理由。

## 7. PRNG 与确定性

未来 generator 必须使用冻结的 ChaCha20 counter stream 或等价明确规定的 CSPRNG，不得使用
Python `random`、平台浮点 RNG 或遍历顺序相关 hash。所有抽样使用 family/domain-separated counter：

```text
stream = HMAC-SHA256(pack_seed, family_id || sample_ordinal || attempt_ordinal)
```

整数区间使用 rejection sampling，禁止 modulo bias。每个 family 有冻结最大尝试次数；耗尽即整 Pack
失败，不能缩小配额。A/B 从 reveal seed 重建的 PNG、GT、顺序和 rejection log 必须逐字节一致。

## 8. 输出与独立评分

Evidence JSON 采用固定字段顺序、整数或约分有理数字符串和稳定 ID；禁止 confidence 驱动的后验
阈值 sweep。candidate generator 输出去重、排序的 canonical graph signature `List`，不能用
`Graph -> Prop` 的不可枚举接口冒充生产结果。

核心集合指标对每个 observation 计算：

```text
candidate_precision    = |PredSet intersect OracleSet| / |PredSet|
candidate_completeness = |PredSet intersect OracleSet| / |OracleSet|
candidate_exact        = (PredSet == OracleSet)
```

`OracleSet` 由独立 challenge grammar enumerator 得到，只相对该有限 grammar 声称完整。独立 verifier
不得 import extractor、candidate generator 或其 graph canonicalizer；图同构按 Node chemical label 和
Edge order 重新实现。

几何 Evidence 另报告 lane support IoU、endpoint Hausdorff、intersection precision/recall、text-like
false-lane 数和变换等变性，但这些都不能替代 candidate exact。

同一 InkProgram 的旋转、反射和 scale 变体在 evaluator 用已知逆变换拉回后比较 Evidence 等变性；
CandidateSet 和 DecodeResult 应保持图同构不变。变换参数只在 reveal 后提供给 verifier，不传给
extractor。

## 9. GO / NO-GO

所有数字在看结果前冻结，不做置信区间替代。Pack A 和 Pack B 必须各自满足全部硬门：

### 9.1 完整性门

- commitment/reveal、seed、source tree、container、image pack、GT 和 prediction hash 全部匹配；
- family 配额、分层外观计数和 rejection log 完整；
- 白名单外文件读取、网络访问、跨样本状态和 schema 外输出均为 0；
- challenge image 与公开旧 observation hash 重合为 0；
- 同一冻结流水线两次 replay 的 canonical 输出逐字节一致。

### 9.2 Exact collision 门

- C1--C4 每类 raw PNG 和 production tensor pair mismatch 为 0；
- 同 bytes 的 canonical Evidence mismatch 为 0；
- C1--C3 `PredSet == OracleSet` 为 100%，不得遗漏任一异图解释；
- C1--C3 全部输出正确 `Ambiguous(kind)`，Graph/Reject 输出为 0；
- C4 全部输出 `Ambiguous(text_or_lane)`，唯一 text mask 或唯一 lane owner 输出为 0；
- C4 的 `clear/near/touch/overlap/occlude` 每个 relation bucket 单独为 100%，不得合并平均。

### 9.3 Strict 门

- S1--S8 `PredSet == OracleSet` 为 100%；
- strict Graph 输出与唯一图同构为 100%；
- strict 的 Ambiguous/Reject、空候选和额外候选均为 0；
- 每个坐标、rotation/reflection、scale、width、cap/join、AA 和 near-clearance 桶单独报告，任一桶
  candidate exact 小于 100% 即失败，不能用总体平均掩盖。

### 9.4 Factorization 与变换门

- exact same raster 的 Evidence canonical bytes 一致率为 100%；
- inverse-transform 后 Evidence 等变性通过冻结 matcher 的比例为 100%；
- CandidateSet/DecodeResult 的变换不变量不一致为 0；
- candidate soundness 与 completeness 均为 100%，不能只验证 chosen graph。

### 9.5 稳定性门

只在移除已登记 exact ambiguity class 后的 strict finite shell 上计算。metric 冻结为 production
preprocessing 后 `768 x 768 x 3` uint8 RGB 数组上的整数 L1：

```text
d(X,Y) = sum[c,y,x] abs(int(X[c,y,x]) - int(Y[c,y,x]))
```

它满足反身、对称和三角不等式。必须完整枚举：

```text
epsilon = 同一 strict Scene 的允许 nuisance shell 到其 canonical raster 的最大距离
Delta   = 两个非同构图 nuisance shell 之间的最小距离
```

只有实际存在 strict Scene、至少存在一对非同构 strict 图，且 `2 * epsilon < Delta` 时才通过。空域、
单图域或只声明未计算的 Delta 都是 `NO-GO`，防止 Lean 条件命题真空实例化。

## 10. 失败处理

任一硬门失败：

```text
overall_gate_status = no_go
training_authorized = false
```

结果必须保留按 family/style bucket 的完整 certificate，不得只公布平均数。可以在 reveal 后诊断并修改
下一版本，但修改后的系统不得重跑本 Pack 冒充 hidden 结果。特别地：

- hash/table lookup 即使分数为 100% 也直接 `REJECT`；
- Scene/GT/seed 泄漏直接 `REJECT`；
- exact collision 上猜中一个 truth 仍是失败；
- Strategy B 或 RDKit 后验选择不能替代 Strategy A；
- 语义 field 左逆不能替代 raster extractor；
- MR-TADF、USPTO 或公开 benchmark 改善不能覆盖 hidden hard gate 失败。

只有 A/B 两个未知 Pack、独立 replay、独立 verifier 和稳定性门全部通过，才允许进入“是否训练
Evidence predictor”的下一次预注册讨论；本协议本身永远不授权训练。
