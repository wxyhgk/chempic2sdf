# EvidenceCore v0.4 P3 结果

日期：2026-08-15

最终状态：`NO-GO`

训练授权：`false`

P3 验证的链路是：

```text
canonical PNG
  -> bounded neutral EvidenceCore
  -> Evidence-only anonymous CandidateGraphSet
  -> Graph | Ambiguous | Reject
```

本轮没有请求 custodian share，没有生成正式 hidden A/B，也没有开始训练。由于公开门已经失败，按预注册
协议不再生成 freeze 后的一次性 diagnostic pack。

## 1. 数学边界

新增的 Lean 接口证明了以下条件式结论：

- 相同 Evidence 的 raster 必须得到相同 Evidence-only CandidateSet；
- soundness 与 completeness 合取才是按图等价类 exact 的候选集合；
- strict observation 的 exact candidates 只有一个图等价类；
- ambiguous observation 的 exact candidates 至少有两个非等价图类；
- correct decoder 的 `Graph/Ambiguous/Reject` 分支分别对应 singleton/multiple/empty CandidateSet；
- 两个有效场景产生相同 Evidence、但图不等价时，sound Evidence-only decoder 不得输出单一图类。

Lean 没有证明 Python/OpenCV extractor 正确，也没有读取 JSON、PNG 或序列化字节。
`NoRasterPayload`、primitive/event 数和 stdout bytes 都是必须由外部审计提供的见证。

## 2. Extractor 结果

公开 zero-seed smoke 含 48 张图、116 条 oracle primitive。冻结的一对一端点审计门限为每端不超过
12 pixel，先最大化匹配数，再最小化无向端点误差。

初始实现：

```text
geometry exact  30 / 48
precision       78.10%
recall          92.24%
```

问题包括粗斜线的双 centerline、无向角 seam、junction 端点截短、terminal 截短，以及孤立像素改变
全局 Hough 输出。修复使用 Zhang-Suen centerline、相对方向对齐、support-valley duplicate merge、
有墨迹见证的 junction snap、terminal extension 和 tiny-component 隔离。最终：

```text
geometry exact       48 / 48
primitive precision  100%
primitive recall     100%
C1/C2/C3/C4          全部 4 / 4
```

独立红队另外通过：

- 负对角和垂直无向角 seam probe；
- 216 组 2/3 lane、1--9px、不同 clearance 与方向的误吞检查，失败 0；
- 48 张图在固定空白角落增加一个黑 pixel，primitive/event 改变 0；
- 48/48 fresh-process byte replay exact；
- 最大 4 primitives、6 events、1,652 stdout bytes，均低于 64/64/65,536 上限。

红队同时发现初版 `status=reject` 不被成功-only schema 接受。修复后，主审计脚本和 Candidate 测试确认
reject 已进入独立 closed schema 分支，合法 reject Evidence 会输出 Candidate `Reject`。

机器结果：

- `formal/chemgraph/evidence_core_v0_4/public_geometry_validation.json`
- `formal/chemgraph/evidence_core_v0_4/public_stability_validation.json`

这些 PASS 只覆盖冻结 synthetic 微语法，不外推 USPTO、MR-TADF、文字、多笔画缩写、楔键或虚线。

## 3. Candidate 结果

Candidate 是 stdlib-only 独立进程，只读一个 canonical EvidenceCore stdin，不读取 PNG、文件名、pack、
Scene、generator、GT 或旧 canonicalizer。独立 verifier 没有导入 Candidate 的图同构代码，而是重新
按 node role、无向边和 edge order 计算图同构签名。

最终公开结果：

```text
CandidateSet exact  44 / 48 = 91.67%
decision exact      44 / 48 = 91.67%
S1--S8              全部通过
C1--C3              全部通过
C4                   0 / 4
```

机器结果：

- `formal/chemgraph/evidence_core_v0_4/public_candidate_validation.json`

没有使用绝对 width、bbox、family、quartet 或 style 相关阈值把 C4 凑成 PASS。

## 4. C4 为什么是协议反例

C4 要求“一根孤立短线是文字 stroke”与“它是普通 lane”两个候选；S8 则把同样由一长一短两根分离
直线构成的观测标为唯一的两条普通 lane。EvidenceCore 有意删除 `glyph_template_id` 来源，只保留：

```text
open/straight primitive geometry + neutral glyph_like event
```

所以 Candidate 不能从来源无关 Evidence 恢复已经删除的 `line/glyph_template` 身份。

更强的可执行反例位于允许参数空间内。取：

```text
S8: half=62, gap=15, scale=3/4, rotation=0, translation=(384,384)
C4-I: half=93, scale=1/2, rotation=90deg, translation=(355,384)
```

对应几何最多相差 0.25 pixel。在 `binary_1x`、偶数 width 2/4/6/8、butt 或 square cap 下，冻结
rasterizer 产生的 Gray8 bytes 和 canonical PNG 都逐字节相同，且两侧 support gate 均通过；但 S8
要求 `Graph`，C4 要求 `Ambiguous`。

当前 generator 的 `seen_png` 遇到重复 PNG 会拒绝后出现的 sample 并重试，而不是把所有有效解释并入
同一个 GraphSet。这样只能让一个 pack 内的文件 hash 唯一，不能证明全局逆映射是单射。

机器反例：

- `formal/chemgraph/evidence_core_v0_4/s8_c4_collision_validation.json`
- `formal/chemgraph/evidence_core_v0_4/audit/audit_s8_c4_collision_v0_4.py`

因此 P3 的失败与参数量、训练图片数量或优化器无关。它是标签协议没有对 observation class 闭包。

## 5. 下一版必须修改什么

v0.5 不能直接调 Candidate。应先重新预注册真值构造：

1. 枚举允许 scene 域并按 canonical raster 等价类分组；
2. 同一 raster class 的所有非同构图全部并入 RasterGraphSet，不能丢弃 duplicate observation；
3. Extractor freeze 后，再按相同 EvidenceCore 合并 RasterGraphSet，得到 EvidenceGraphSet；
4. Candidate exactness 与 EvidenceGraphSet 比较，而不是与某个 family 的单条标签比较；
5. C4/S8 这类冲突必须输出 `Ambiguous`，除非增加了来源无关且可从最终像素稳定恢复的新证据；
6. 重新做公开门和 freeze 后一次性 diagnostic，全部通过后才讨论训练。

v0.4 到此封口为诚实 NO-GO。禁止用 style 旁路继续调到 48/48，也禁止把 synthetic geometry PASS
解释成真实化学识别已经解决。
