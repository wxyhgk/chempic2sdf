# Hidden Generator v0.4 预注册

日期：2026-08-15

状态：`generator_first_preregistered_before_v0_4_implementation`

v0.3 已被三名 reviewer 限定为公开 `64x64 / 3px` 整图穷举基线。v0.4 重新开始，不继承 v0.3
的 extractor、candidate、runner 或 hidden 资格。本文件只冻结 generator-first 顺序与验收协议；在
generator、oracle、container、schema 和两方 seed commitment 完成前，禁止实现 v0.4 extractor。

## 1. 不可更改的顺序

```text
P0  本预注册 commit
P1  generator + fixed rasterizer + schemas + oracle contract commit
P2  两名 custodian 的 C1/C2 commit，不 reveal share
P3  v0.4 extractor + EvidenceCore + candidate + decoder freeze E
P4  由 C1/C2 和 E 派生 hidden A/B seed，私有生成 pack
P5  提交 prediction/replay commitment
P6  reveal share/GT/rejection log，独立重建和验收
```

如果 P1/P2 晚于 P3，整个版本无 hidden 资格，不能补做。P3 后任何源码、依赖、阈值、schema、权重或
容器字节变化都必须新开版本和新 seed。

## 2. Generator 闭包

公开 generator 必须位于 `formal/chemgraph/challenge_v0_4/`，只依赖 Python 标准库，不得 import：

- v0.1/v0.2/v0.3 Scene、fixture、grammar classifier 或 canonicalizer；
- chemvector、RDKit、Pillow、OpenCV、Torch、NumPy、NetworkX；
- SDF、Render JSON、训练 manifest 或 benchmark。

它必须从公开 specification 和显式 seed 独立生成：

```text
seed -> InkProgram -> one Raster -> InterpretationSet -> sealed GraphSet
```

exact collision 必须先生成一个共享 InkProgram，只 rasterize 一次，再附加多个 Interpretation；禁止分别
渲染两个 semantic witness 后比较“看起来相同”。

## 3. Canonical raster profile

- 画布：`768x768`；
- 几何：有符号 Q16 fixed-point；
- 变换：Pythagorean 有理旋转、水平/垂直反射、有理缩放、整数平移；
- 线宽：1--9 px 的冻结离散桶；
- 抗锯齿：1x、2x、4x 确定性 supersampling；
- 颜色：第一门只用白底黑墨，颜色不作为语义旁路；
- PNG：8-bit grayscale，non-interlaced，只含 `IHDR/IDAT/IEND`；
- PNG encoder 和 zlib level 固定，chunk 顺序与行 filter 固定；
- 无 metadata、文件名语义、样本序号水印或不同 interpretation 的编码差异。

fixed rasterizer 必须用整数运算定义 coverage、cap/join、rounding 和 downsample；不允许依赖平台浮点或
系统字体。文字式干扰只用公开的 line/arc glyph templates，不声称 OCR 字符覆盖。

## 4. Hidden grammar

每个 A/B pack 的核心配额：

| 类别 | observation class |
|---|---:|
| strict S1--S8 | 4,096 |
| C1 crossing / degree-4 junction | 512 |
| C2 double / two parallel singles | 384 |
| C3 triple / three parallel singles | 256 |
| C4 glyph-stroke / bond-lane ownership | 384 |
| 合计 | 5,632 |

另从 strict 中取 512 个 replay；每个 exact collision 以相同 PNG bytes 分别调用两次，所以每 pack 总计
7,680 次隔离 extractor invocation。A/B 分布相同、seed domain 不同、样本不重叠。

Strict families：single carrier、angled chain、degree-3 fan、非十字 degree-4 junction、triangle/
quadrilateral、两个非对齐 disconnected bonds、long/short mixed path/fan、near-crossing/near-parallel
但保持正间隙。名称不构成 strict 真值；独立 oracle 必须证明该 grammar 下 `|GraphSet|=1`，否则样本
转入 ambiguity 并记录，不能删掉后继续算原分母。

## 5. 预测前可见/不可见

预测前可见：

- generator、rasterizer、schemas、配额、PRNG/HKDF 算法；
- image pack bytes 与 image pack commitment；
- extractor/candidate freeze record E。

预测前不可见：

- seed shares、最终 seed、family、transform、InkProgram、Interpretation、GraphSet；
- sample key 与图片的对应 GT；
- rejection log、epsilon/Delta bucket identity。

图片 transport handle 只能由 evaluator 保存，不进入 PNG、stdin、argv、env 或 Evidence。

## 6. EvidenceCore 硬边界

v0.4 EvidenceCore 允许：

- 有界数量的 normalized line/curve primitives；
- terminal、junction-region、intersection-region、glyph-region hypotheses；
- 每个 primitive 的局部 support score、uncertainty 与 residual count；
- 公开固定精度的有理/整数坐标。

禁止：

- 完整 foreground row-runs、完整 pixel/skeleton 列表或可无损还原 raster 的 payload；
- PNG bytes、PNG/tensor digest、任意整图压缩字符串或逐像素 embedding；
- Scene/Atom/Bond ID、bundle ownership、graph signature、family、transform、seed；
- OCR 真值、crossing/junction 真值或键级真值。

Host 可以保存完整 raster coverage certificate，但该证书不得进入 candidate 进程。schema 必须冻结数组上限、
数值范围和总输出字节上限，防止把整图藏入超长 primitive 列表或高精度小数。

## 7. 四个独立闭包

```text
Host/generator: Raster + sealed GT
Extractor:      PNG stdin -> EvidenceCore stdout
Candidate:      EvidenceCore stdin -> anonymous explicit CandidateGraph[]
Verifier:       prediction commitment + revealed GT -> verdict
```

Extractor 不得读取 generator/GT；Candidate 不得 import 或读取 Scene、generator、rasterizer、GT、现有
canonicalizer，也不得看到 PNG；Verifier 不信任 runner summary，必须从逐样本 Evidence、CandidateGraph、
DecodeResult 和 reveal manifest 重算。

CandidateGraph 使用本协议自带的匿名节点/边 JSON schema。独立 verifier 自己实现小图同构或 canonical
比较；不能调用 candidate 的 canonicalizer。

## 8. Seed commitment

两名独立 custodian 各自产生 256-bit share，只公开：

```text
C1 = SHA256("chemgraph-v0.4-share-1\0" || r1)
C2 = SHA256("chemgraph-v0.4-share-2\0" || r2)
```

P2 文件记录 P0/P1 commit、C1/C2、custodian role 和签名，不含 share。P3 冻结后由 evaluator 使用
HKDF-SHA256 派生 A/B seed。至少一个 share 对 extractor 作者未知；A/B prediction commitment 固定后才
reveal。

## 9. Stability metric

generator 同时生成同图 nuisance orbit 与异图 comparison domain。主距离冻结为 8-bit grayscale raster
的整数 L1；另报二值 foreground Hamming，不混用门槛。

逐 renderer-profile / motif bucket 完整计算：

```text
epsilon = 同图 nuisance orbit 到 canonical render 的最大 L1
Delta   = 完整 comparison domain 中异图 render 的最小 L1
```

只有 finite-complete、非空且至少含一对异图的 bucket 才能申请 stable。严格要求
`2 * epsilon < Delta`，等号失败。采样最小值只能叫 sampled diagnostic，不能叫 Delta 下界。

## 10. Generator-first 验收门

P1 只有以下全部通过才可提交：

1. generator/rasterizer/schema/contract 在 extractor 之前单独 commit；
2. stdlib-only，直接和传递依赖扫描通过；
3. canonical PNG parser 复核只有允许 chunks；
4. 同 InkProgram 重复生成 PNG bytes 相同；
5. C1--C4 每个 interpretation 的 PNG bytes 来自同一份 Raster；
6. GraphSet 对 atom/edge 排列不敏感，由独立 oracle 复核；
7. smoke pack 的配额、拒绝采样、replay 和 A/B domain separation 通过；
8. 相同 seed 的 pack/GT/rejection log commitment 逐字节一致；
9. 不生成或提交真实 hidden seed、share、full pack 或可调参 preview；
10. 两个未参与 generator 实现的 reviewer 均批准。

P1 通过也不授权训练，只允许进入 P2。任何字段缺失、平均通过、删失败样本或动态改配额均为 NO-GO。

## 11. 本周期明确不声称

- 不声称 generator distribution 等于 USPTO、MR-TADF 或真实论文图；
- 不声称 v0.4 extractor 已存在；
- 不声称 EvidenceCore 已足够；
- 不声称 candidate generator 可恢复任意图；
- 不声称固定点合成数据替代真实 benchmark；
- 不声称 hidden PASS 自动允许大规模训练。
