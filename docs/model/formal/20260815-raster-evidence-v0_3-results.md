# Raster-neutral Evidence v0.3 正式结果

日期：2026-08-15

预注册提交：`caa8ad1f87c52243c4058ba8884510784db3c2c5`

冻结实现提交：`35a32ef6dc758bfc697795b5242c373c1649697a`

总判定：

```text
Lean conditional interface theorems:         PASS
Public 64x64/3px foreground exhaustive run:  PASS
Structured Evidence -> Candidate causality:  NO-GO
Candidate implementation independence:       NO-GO
Strong extractor isolation:                  NO-GO
Hidden challenge eligibility:                NO-GO
Stability 2 epsilon < Delta:                  NOT RUN / NO-GO
Training authorized:                         false
```

本轮最重要的结果不是 100% 数字，而是红队证明这些数字来自“完整前景 RLE + 公开有限 Scene
全枚举”，不是 lanes/terminal/intersection 已经足以恢复图。v0.3 因此只保留为公开有限域的
analysis-by-synthesis 上界基线，不能作为训练或 hidden challenge 的放行证据。

## Lean 结果

新增 `RasterEvidence.lean`，SHA256：

```text
4c3c1a4f422ec4a3b07804ae5524c4ef6b776c51e7bd3c88061282d9c59e9fff
```

Lean 4.33.0 build 为 14/14，无 `sorry`、`admit` 或新增 `axiom`。本轮只证明条件式接口命题：

- 同 raster 的确定性 extractor 必然输出相同 Evidence；
- Oracle evidence 若在同 raster 碰撞对上不同，则不可能 factorize；
- candidate soundness 与 completeness 合起来才是 exact；
- exact singleton 必须输出 Graph，multiple 异图候选必须输出 Ambiguous；
- `2 epsilon < Delta` 加全域异图距离下界可推出噪声球内图唯一；
- stability GO 还必须提供非空域和至少一对异图 witness，不能真空通过。

Lean 没有证明 Python extractor、candidate enumerator、taxonomy、全域 Delta 或 decoder 满足这些假设。

## 公开有限域 A/B

extractor 只从 stdin 接收一个 PNG，并用 Python 标准库解 PNG、阈值化、细化和输出中性几何。它还
输出完整二值 foreground row-runs 和全部 skeleton pixels。候选器随后在运行时前向渲染公开的
v0.1 Scene grammar，并用完整 row-runs 作精确键匹配。

两个完整运行覆盖：

| 域 | Scene | observation | Graph | Ambiguous | Reject |
|---|---:|---:|---:|---:|---:|
| simple | 19,824 | 19,823 | 19,822 | 1 | 0 |
| crossing | 36,657 | 36,649 | 36,641 | 8 | 0 |

在这一个公开固定 `64x64 / 3px / solid black` renderer profile 上：

- submitted candidate set 与冻结 manifest graph set 在 56,472 个累计域 observation 上相等；
- wrong Graph 为 0；
- crossing/junction fixture 的 PNG 与 Evidence bytes 相同；
- double/parallel fixture 的 PNG 与 Evidence bytes 相同；
- A/B 产物逐字节一致。

A/B hash：

```text
summary.json
414afe9447af888882dd97e50fa9aabd7183ba0a62658cbc10966306d1e92ba9

validation.json
6e93e2a915207489de446a31524c68f175ac109c02be1c9e34ccb5a9490c9b6e

raster_evidence_manifest.jsonl
bec6e25df412f3a87670032e3080542689564e77cbc44b0903df5e43950d3221

independent_review.json
b8a8f2a1c762994bdea9fa294c879fd09d640ed2126a1370415c9e50351da0a0
```

独立 stdlib verifier 从冻结 v0.1 manifest 重组 Scene observation class，并拒绝了四类篡改：删
候选、改决策、删 observation、改冻结 manifest。它证明 artifact 内的 Scene coverage、GraphSet 数组
和决策一致，但不读取或重算 Evidence，所以不能称为独立验证了 `Evidence -> CandidateSet` 因果链。

## Reviewer REJECT

### 1. 完整前景旁路

`foreground.row_runs` 是阈值后二值图的无损 RLE。它不是隐藏 GT，也不是加密 blob，但当前候选器用
它直接执行：

```text
完整前景 tuple -> 公开有限 grammar 的 canonical graph set
```

因此候选正确性完全可以绕过本轮输出的 lanes、terminal、intersection。当前结果不能证明这些结构化
字段足够，也不能证明路线已从整图查表泛化到未知几何。

### 2. Candidate 边界未隔离

候选器直接 import Scene、`iter_core_scenes`、fixture、grammar classifier 和 graph canonicalizer，并和
持有 Scene ID/GT index 的 runner 同进程。这与工程审查要求的 extractor、candidate、verifier、Host GT
四个依赖闭包不符。

### 3. Runtime isolation 声明过强

本轮实际执行了 `python -I -S`、空临时 cwd、清理环境和单样本 `%file,%network` strace；源码没有打开
仓库数据或网络。但没有实现最小只读 rootfs、namespace、seccomp、资源/FD 全量审计或全 A 轮 trace。
A/B 内部的 `extractor_runtime_isolation=true` 只能解释为“该单次 trace 未见仓库/网络访问”，不能解释
为强隔离已完成。

### 4. Factorization 布尔值过强

A/B 内部的 `raster_factorization_finite_domain=true` 实际由“公开 candidate gate 通过 + 两组 fixture
Evidence 相等”组成，没有逐全部合法 Scene 验证 Lean 定义的
`extract(render(scene)) = oracleEvidence(scene)`。正式元 validation 将该字段降级为 unsupported。

### 5. Hidden 冻结顺序不成立

hidden generator、container、schema、metric、oracle enumerator 和两方 seed commitment 没有在 extractor
之前独立冻结。当前 extractor 上限为 256，正式数据只有 64x64/3px，也不支持红队协议冻结的 768、Q16、
9 档线宽、抗锯齿和文字干扰。因此不能追溯性补跑 hidden pack。

## 有效结论

本轮可以且只能声称：

1. 完整 PNG bytes 可以确定性转换成完整二值前景 RLE 和若干中性派生几何；
2. 在公开固定有限 grammar 上，以完整前景做穷举 analysis-by-synthesis 可以得到 exact candidate set；
3. exact raster collision 必须保留为多个候选，不能靠更多训练恢复被像素丢掉的语义；
4. 该穷举法不扩展到真实分子空间，不能回答未知坐标、风格、文字和抗锯齿；
5. `training_authorized=false`。

## 下一周期硬顺序

下一版本不能在 v0.3 上追加 hidden 分数，必须按以下顺序重新开始：

1. 先单独提交和审批 hidden generator、固定点 rasterizer、PNG encoder、family 配额、metric、独立 oracle
   和 container schema；
2. 两名独立 custodian 提交 `C1/C2`，不 reveal share；
3. EvidenceCore 删除完整 foreground row-runs 和完整 skeleton pixel list，只保留有界结构化几何；完整
   raster 只能进入 Host-only coverage certificate；
4. candidate generator 只读 EvidenceCore，不导入 Scene、generator 或现有 canonicalizer，输出匿名显式
   图候选；
5. 建立真实最小 rootfs、namespace、无网络、资源/FD 限制的一图一进程 sandbox；
6. 再冻结 extractor/enumerator/decoder；
7. 冻结后才派生 hidden A/B，先提交 prediction commitment，再 reveal 和独立重建；
8. 逐 bucket 完整计算 epsilon/Delta；只有严格满足 `2 epsilon < Delta` 的 strict bucket 才可 Graph。

在以上步骤完成以前，继续暂停模型训练。
