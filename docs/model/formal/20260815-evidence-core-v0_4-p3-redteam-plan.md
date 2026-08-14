# EvidenceCore v0.4 P3 红队预案

日期：2026-08-15

状态：`redteam_preregistered_before_p3_freeze_and_diagnostic_seed`

本文只冻结 P3 实现冻结后的单次诊断协议和红队验收方法。本文没有读取未来 extractor/candidate
源码，没有生成 diagnostic seed 或 pack，没有请求两名 P2 custodian 的 share，也不授权训练。

P3 只审计下列有限链路：

```text
opaque 768x768 Gray8 PNG
  -> bounded neutral EvidenceCore
  -> anonymous CandidateGraphSet
  -> Graph | Ambiguous | Reject
```

即使全部通过，也只说明冻结实现通过 v0.4 的 48-observation diagnostic grammar；不说明真实化学图片、
OCR、USPTO、MR-TADF 或任意分子已经解决。

## 1. 冻结顺序

顺序必须是：

```text
P3 preregistration/red-team plan commit
  -> implementation + dependency/container freeze E
  -> independent freeze audit
  -> one-time diagnostic seed commitment
  -> pack generation
  -> two isolated prediction runs
  -> prediction commitment
  -> diagnostic seed/GT reveal
  -> independent rebuild and verdict
```

`E` 至少绑定：

- extractor、Candidate、decoder、canonical JSON serializer 的完整 source-tree hash；
- EvidenceCore/CandidateGraphSet/DecodeResult schema bundle hash；
- geometry interpretation rules 文档 hash；
- Python/NumPy/OpenCV 版本、wheel/二进制 manifest、OpenCV build information；
- 容器或等价只读运行环境 digest；
- 所有阈值、Q4 rounding、排序规则、数组/字节上限；
- extractor 与 Candidate 的命令行、stdin/stdout 和文件访问白名单；
- 单元测试、公开 zero-seed 结果及其逐样本 artifact commitment。

`E` 后任一源码、依赖、schema、阈值、排序规则、环境或权重字节变化都产生新版本；旧 diagnostic seed
不能重用。

## 2. 单次 Diagnostic Seed Commitment / Reveal

P3 diagnostic seed 与 P2 custodian share 完全分离。它不能用于 P4 正式 hidden A/B，也不能提前请求或
组合 `r1/r2`。

### 2.1 Commitment

在 `E` 冻结并通过独立闭包审计后，由一名未参与 extractor/Candidate 实现的 diagnostic reviewer 使用
操作系统 CSPRNG 产生 32-byte `d`。raw `d` 只存在于 reviewer/evaluator 的隔离内存或批准的密钥设施。
公开：

```text
D = SHA256("chemgraph-v0.4-p3-diagnostic-seed\0" || d)
```

commitment record 必须同时绑定：

- P0、P1、P2、P3 preregistration commit；
- generator、rasterizer、oracle/verifier、schema bundle hash；
- freeze record `E` 与容器 digest；
- `D`、32-byte 长度、`raw_seed_included=false`；
- `generated_pack=false`、`training_authorized=false`；
- reviewer 身份、UTC 时间和签名。

commitment 发布前不得生成 diagnostic pack，不得试 seed，不得查看配额能否“更容易”。

### 2.2 Seed derivation

commitment 发布后，evaluator 冻结使用：

```text
salt = SHA256(
  "chemgraph-v0.4-p3-diagnostic-salt\0" ||
  P0 || P1 || P2 || P3 || E || generator_hash || oracle_hash || schema_bundle_hash
)
diagnostic_seed = HKDF-SHA256(
  ikm  = d,
  salt = salt,
  info = "chemgraph-v0.4-p3-one-time-diagnostic-pack"
)
```

所有 hash 以固定字节编码和 length-prefix framing 连接，不允许字符串拼接歧义。generator 只能收到派生
后的 32 bytes；不得收到 reviewer 身份、文件名或 family sidecar。

### 2.3 Prediction commitment

在 reveal `d`、GT、family、transform、style、InkProgram 或 rejection log 前，必须完成两次隔离运行并
提交：

- input image-pack manifest hash 与 invocation-order hash；
- 每个 invocation 的 canonical EvidenceCore、CandidateSet、DecodeResult hash；
- 两次 run 的 Merkle root、stderr、exit code、资源与 syscall audit hash；
- `E`、容器和 evaluator runner hash；
- 每张 PNG 的 replay 对应关系由 evaluator 私下保存，不传入模型。

### 2.4 Reveal 与退役

prediction commitment 固定后才 reveal：raw `d`、seed derivation record、完整 GT、transform/style、
InkProgram、GraphSet、support certificate 和 rejection log。独立 verifier 必须从 `d` 重建全部 artifact，
验证 `D`、HKDF、图片、顺序、GT 和日志逐字节一致。

该 seed reveal 后永久退役。失败后不能修改实现再重跑同 seed；只能新开 P3 版本、重新冻结并由另一
个从未使用的 seed 重新预注册。diagnostic PASS 不 reveal P2 custodian share。

## 3. 固定配额与调用

一个 diagnostic pack 恰好包含：

| family | unique observation |
|---|---:|
| S1--S8 | 每类 4，共 32 |
| C1--C4 | 每类 4，共 16 |
| 合计 | 48 |

invocation 固定为：

```text
48 primary
+ 16 collision replay（全部 C1--C4）
+ 8 strict replay
= 72 isolated invocations
```

不得删失败 observation、替换 family、缩小分母或在看到输出后重采样。generator 自己的预注册拒绝
采样可以执行，但 attempt/reason 必须完整 reveal，并由独立 oracle 重建。

diagnostic PNG SHA256 与公开 zero-seed 48 张图片重合必须为 0；不同 unique observation 除协议 replay
外也不得同 hash。

## 4. 未知输入条件与隔离执行

Extractor 每次只收到当前 PNG stdin；Candidate 每次只收到当前 canonical EvidenceCore stdin。
两者都不能收到：

- path、delivery/sample/invocation ID、pack 顺序；
- family、strict/collision 标记、quartet/variant；
- rotation/reflection/scale/translation、line width、cap/join、AA profile；
- InkProgram、template ID、GT、GraphSet、support certificate、seed 或 rejection log。

每个 invocation 使用全新容器或经证明无状态的等价隔离。网络关闭，时钟/主机名/随机环境固定，repo、
`Data/`、其他图片和 evaluator manifest 不可见。72 次调用另以第二种随机顺序执行一次；同一 PNG 的
canonical 输出必须不依赖前后样本。

style/transform 只在 reveal 后用于分桶。任何 bucket 的失败不能被总体平均掩盖。

## 5. EvidenceCore 容量与编码攻击

### 5.1 Closed schema

冻结门至少为：

- `stroke_primitive <= 64`，`event_region <= 64`；
- 坐标/宽度只用有范围的 Q4 整数；score 只用 `0..10000`；
- ID 由 canonical geometry 排序连续产生；
- string 使用枚举或严格长度/pattern，禁止自由文本；
- stdout canonical JSON `<= 65,536 bytes`；
- unknown field、重复 key、NaN、Infinity、超范围整数和尾随 bytes 全部拒绝。

只验证 JSON schema 不够。64 个 primitive 的端点、score 和 residual 数仍可能成为 covert channel，必须
再做 grounding 与差分测试。

### 5.2 明确禁止

任何下列内容出现即 P3 `NO-GO`：

- foreground row-runs、RLE、bitmap、PNG bytes、压缩 raster、base64/hex blob；
- 完整 pixel/skeleton/edge-point list；
- PNG/tensor hash、per-image lookup key、逐像素 embedding；
- 通过大量短 primitive、坐标低位、score/residual 数字编码像素；
- Scene/Atom/Bond/family/transform/Graph signature 或 chemical truth。

### 5.3 检测 row-runs / pixel 藏入 primitive

同时执行以下四类审计：

1. **静态数据流审计**：检查 serializer 的所有输出字段赋值路径。禁止把 row iterator、connected-component
   pixel list、skeleton list、PNG bytes、`tobytes()`、压缩/编码/hash 结果流入 JSON。搜索并人工复核
   `run/rle/row/pixel/skeleton/base64/hex/zlib/hash/digest/packbits`、bit shift、任意大整数拼装和动态 key。
2. **独立几何 grounding**：reveal 后 reviewer 从 PNG 独立检测可见 support。每个 primitive 必须有真实
   直线/曲线支撑；Q4 endpoint、width、orientation、support score 和 residual 必须由独立算法在冻结容差内
   重算。没有可见支撑、坐标低位与最佳拟合不一致、score/residual 无法复算，全部拒绝。
3. **碎片容量攻击**：输入 checkerboard、每行不同 barcode、密集短 run、孤立单像素和随机噪声。输出仍须
   满足 64/64/65,536 上限；不能把每个 run 伪装成 primitive。共线且中间无 event 的相邻 fragments 必须
   canonical merge；fragment endpoint 只有落在 terminal/event region 才合法。
4. **差分敏感性**：对空白 margin 做单像素/小块扰动，并改变与已检测 carrier 无关的噪声。primitive/event
   geometry 与 ID 应保持不变；只允许冻结的全局 residual aggregate 改变。若无关像素能系统性改变 endpoint
   Q4 低位、orientation、score 数字序列，视为 covert payload。

另由红队实现一个“最大信息解码器”，尝试从 Evidence 重建二值 raster。若能无损或近无损恢复 row-runs，
即使 schema 字段名称合法也直接失败。该攻击用于发现问题，不能反过来成为 Candidate 的依赖。

## 6. Line Fragmentation / Merge 审计

### 6.1 Fragmentation

下列行为禁止：

- 一根无内部 event 的直 carrier 被切成多个 primitive；
- 以 1--2 pixel 短 fragment 编码 AA 边缘或 row-run；
- 在同一直线上重复、重叠 primitive；
- fragment 边界没有 terminal/junction/intersection/glyph-like hypothesis 支撑。

独立 verifier 将 oracle neutral carrier 与 Evidence primitive 做一对一匹配。只有 fragment endpoint 落在
冻结 event bbox 内时才允许受控 one-to-many，并必须在 diagnostic 中单独报告；P3 的 48 张有限 grammar
目标是非法 fragmentation count 为 0。

### 6.2 Merge

禁止把以下对象合并为一根 primitive：

- C1 的两根 proper-cross carrier；
- C2/C3 的两/三条平行 lane；
- angled chain/fan/cycle 在共享端点两侧的不同方向 carrier；
- C4 的 chemical carrier 与单 stroke glyph；
- S8 仍有正 support clearance 的 near-cross/near-parallel carrier。

共享 round join 只能产生 event region，不能消除 incident primitive 身份。merge/fragment 错误既按
primitive-level 计数，也必须通过 Candidate exact 端到端暴露，不能只报较高的像素 IoU。

## 7. C1--C4 专项门

每类 4 张单独报告，不合并平均：

| family | Evidence 必须保留 | Oracle CandidateSet / DecodeResult |
|---|---|---|
| C1 | 两 carrier、中心 event 的 crossing/junction 中性 hypotheses | disconnected crossing 与 degree-4 junction，`Ambiguous(crossing_junction)` |
| C2 | 两条分离 lane，不提前赋 order/owner | order-2 bond 与 two disconnected singles，`Ambiguous(parallel_multiplicity)` |
| C3 | 三条分离 lane，不提前赋 order/owner | order-3 bond 与 three disconnected singles，`Ambiguous(parallel_multiplicity)` |
| C4 | chemical carrier 与单 stroke glyph 分离；role 未决 | `TEXT_STROKE` 与 glyph-as-one-bond-lane 两解释，`Ambiguous(text_or_lane)` |

C4 只测试 stroke role，不测试 `I/l/1/slash` 字符识别。template ID 不能出现在 Evidence/Candidate，且四种
template 的 text candidate graph signature 必须相同。

exact collision 的 primary/replay 输入 PNG bytes 相同，Evidence bytes 也必须相同。只猜其中一个 prior
答案、输出 Reject 或用 sanitize 后验选一个图都失败。

## 8. Metamorphic Quartet 与 Replay

每个 family 的 4 个 observation 是同一 base motif 的 transform quartet。transform 对模型未知；reveal
后 verifier 使用已知逆变换检查：

- primitive 数量、incident pattern、neutral event hypothesis 集合不变；
- Q4 endpoint/centroid/bbox 在冻结 quantization tolerance 内等变；
- support score/residual 的 style-dependent 变化只在预注册容差内；
- CandidateGraphSet 按图同构完全不变；
- DecodeResult kind 和 ambiguity taxonomy 不变。

Replay 门：

- 16 个 collision replay 的 Evidence/Candidate/DecodeResult canonical bytes 全相同；
- 8 个 strict replay同样逐字节相同；
- 第二种 invocation 顺序下按 PNG hash 对齐后仍逐字节相同；
- 任一 stateful counter、前样本缓存、batch position 或文件名依赖均为失败。

## 9. Candidate 闭包与间接 Import 攻击

Candidate 只能读取一条 EvidenceCore stdin、冻结 schema 和手写 geometry rules。不能看 PNG，不能 import
或读取 generator、rasterizer、Scene、GT、GraphSet、support certificate、legacy canonicalizer 或 evaluator。

“源码没有直接 import”不够；红队必须检查间接依赖：

1. 递归生成 Python module/import graph，记录所有 `sys.modules[*].__file__`、wheel `RECORD`、native library
   和资源文件 hash；任何路径落入 repo generator/Scene/audit/GT/Data 即失败。
2. 在只挂载 Candidate executable、stdlib、schema、geometry rules 的最小只读容器运行。generator repo、
   PNG decoder、NumPy/OpenCV、GT 和网络均不存在；仍必须通过手写 Evidence fixture。
3. 用 syscall trace/audit hook 记录 `open/openat/stat/readlink/execve/socket`。白名单外读取、symlink 逃逸、
   subprocess、`ctypes`/动态库加载、网络或环境 lookup 全部失败。
4. 静态搜索和人工复核 `importlib/pkgutil/eval/exec/marshal/pickle/subprocess/ctypes`、绝对路径、observation
   hash、generator family coordinate table和大常量表。复制 generator 逻辑进 Candidate 与 import 同罪。
5. Candidate fixture 必须手写，不能从 GT/InkProgram 自动转换。对相同几何但重排 primitive ID/JSON key 的
   Evidence，canonical CandidateSet 应保持图同构不变，防止 ID 查表。

最终 freeze artifact 要提交“允许依赖闭包”和“实际运行依赖闭包”两份 manifest；二者差集必须为空。

## 10. 独立 Candidate Exact 验收

独立 verifier 不 import Candidate 的 graph canonicalizer，也不信 runner summary。prediction commitment
后，它从 reveal artifact 重建 Oracle GraphSet，并自行实现小图同构：

- node 比较 `implicit/text_stroke` role；
- edge 比较无向 endpoints 与 order `1/2/3`；
- 忽略 node/edge 数组顺序和匿名 ID；
- CandidateSet 先去同构，再与 OracleSet 双向比较。

逐 observation 计算：

```text
sound       = PredSet subset of OracleSet
complete    = OracleSet subset of PredSet
exact       = sound and complete
```

同时独立验证：

- `|PredSet|=0 -> Reject`；
- `|PredSet|=1 -> Graph` 且图同构；
- 多个非同构候选 -> 正确 taxonomy 的 Ambiguous；
- strict 不得 Reject/Ambiguous；collision 不得 Graph/Reject。

必须输出 48 行逐样本 certificate，以及 S1--S8、C1--C4、style、transform、replay 分桶计数。总体 100%
不能掩盖任一 bucket 失败。

## 11. GO / NO-GO

P3 `GO` 要求以下全部成立：

1. 本预案和 P3 preregistration 早于 implementation freeze `E`；
2. diagnostic seed commitment 早于 pack generation，commit/reveal/HKDF/rebuild 全部匹配；
3. 48 unique observation、72 invocation 配额精确，zero-seed hash overlap 为 0；
4. Extractor/Candidate 进程与文件闭包隔离，无 GT/generator/PNG 间接依赖；
5. EvidenceCore closed schema、64/64/65,536 容量和 adversarial serializer 测试全部通过；
6. row-run/pixel/blob/hash payload、无支撑 primitive、非法 fragmentation/merge 均为 0；
7. 48/48 CandidateSet exact，soundness 与 completeness 都为 100%；
8. 32/32 strict 输出正确 Graph，错误 Graph/Ambiguous/Reject 均为 0；
9. 16/16 C1--C4 输出完整多个候选和正确 Ambiguous taxonomy；
10. 24/24 replay 的 Evidence/Candidate/DecodeResult bytes 精确一致，换序运行也一致；
11. 12 个 metamorphic quartet 的 CandidateSet/DecodeResult 图同构不变量为 100%；
12. independent verifier 重建通过，两名未参与实现的 reviewer 批准；
13. `training_authorized=false`，P2 custodian share 未 reveal，未生成 P4 hidden A/B。

任一项失败即 `NO-GO`。以下做法禁止：

- 看结果后改 seed、阈值、schema、candidate rules 或删样本；
- 把 Reject、部分 candidate overlap、chosen graph accuracy 或 geometry IoU 当作 exact；
- 同一个已 reveal diagnostic pack 调参后重报 hidden 分数；
- 因公开 zero-seed 通过而跳过 freeze 后一次性 diagnostic；
- 因 Lean 条件定理成立而声称 OpenCV/Python implementation 已验证。

## 12. P3 结束边界

P3 GO 只允许提交一份有限 diagnostic 结果和是否进入下一阶段的评审材料。它不自动允许训练，也不自动
允许 P4。正式 P4 必须另按 P0/P2 协议，在 `E` 不变且 custodian 角色仍隔离时派生全量 hidden A/B；P3
diagnostic seed、图片和 reveal 数据不得混入训练、阈值选择或正式测试。
