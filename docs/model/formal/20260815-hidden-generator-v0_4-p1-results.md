# Hidden Generator v0.4 P1 正式结果

日期：2026-08-15

P0 预注册提交：`55a459499a5f72775e2796c7bd0c0ff012eaf244`

P1 冻结提交：`b015d5d61f9dcc8f4a6c8581d3ff1e2392827557`

总判定：

```text
Generator-first Git order:       PASS
Lean conditional protocol:       PASS
Public smoke deterministic A/A:  PASS
Public smoke A/B separation:     PASS
Schema and artifact audit:       PASS
Finite candidate certificates:   PASS on named smoke grammar only
Full 5,632 hidden pack:           NOT RUN
Formal seed commitments C1/C2:   PENDING at this record
Extractor / Candidate model:     NOT IMPLEMENTED
Training authorized:             false
```

P1 只批准进入 seed commitment 阶段，不批准生成正式 hidden pack，不批准训练。

## 冻结闭包

P1 在 extractor 实现之前冻结：

- stdlib-only Q16 fixed-point generator；
- `768x768` gray8 canonical PNG rasterizer；
- 1--9 px width、butt/round/square cap、真实整数 round join；
- 1x/2x/4x deterministic supersampling；
- S1--S8 strict 与 C1--C4 collision grammar；
- 19 个 closed JSON Schema；
- stdlib schema validator 与独立 pack auditor；
- Lean `HiddenChallenge.lean` 条件式协议。

Lean SHA256：

```text
d5a092cf4b412f3a86eb9ebabbef8098a9260f7f6eb452e5f3acca0c1e26b481
```

Lean 15/15，无 `sorry`、`admit` 或新增 `axiom`。Lean 已从 finite-complete scene list 和列表内全部
异图 pair 的距离下界推出全域 separation；但没有计算本轮 epsilon/Delta，也不证明 Git 时间、seed
秘密、SHA256 抗碰撞或 Python 实现正确。

## 第一轮 NO-GO

最初 generator 自测为 10/10，A/A 也逐字节一致，但三名 reviewer 一致拒绝 P1：

1. C3 在 width=9px 和缩小 scale 下发生 lane 融合；zero-seed smoke 中两个样本分别有 720/456 个
   per-lane foreground overlap pixel，另两个仅约 1px gap 并形成 8-neighbor 接触；
2. C4 的多 stroke `M` 只登记 text 与 disconnected lanes 两图，但共享端点至少产生 5 类 lane-only
   topology，加 text 至少 6 类，GraphSet 漏候选；
3. join 只写入 metadata，rasterizer 没有实现 join coverage；
4. generator source commitment 未覆盖 README、schemas 和 oracle code；
5. independent auditor 没有重算 protocol bundle，schema 修改后的旧 pack 仍会错误 PASS；
6. I/l/1/slash label 可由相对 carrier offset 猜出，污染了只测 stroke-role 的目标。

因此第一轮公开 smoke 被永久登记为 `NO-GO`，其 hash 不作为 P1 正式证据。

## 修复

- C2/C3/C4 在 accept 前逐 primitive 使用正式采样规则生成 support；任一 pair 有 pixel intersection 或
  8-neighbor adjacency 时，以 `invalid_exact_collision` 拒绝该 attempt 并继续，目标 quota 不变；
- C2/C3 base spacing 同时按 width 和最小 scale 预留，最终 support certificate 仍是权威门；
- C4 收窄为 single-stroke `I/l/1/slash` 模板；candidate graph 中 label 恒为 `TEXT_STROKE`，模板名只在
  sealed 诊断字段，公开 pack 不暴露；
- round join 真实 rasterize 共享 Q16 endpoint 的整数 disk，并有测试证明 butt cap 下像素确实改变；
- `protocol_bundle_sha256` 递归覆盖 Python、README、root schemas 与 `schemas/*.json`；
- independent auditor 自己重算 source/protocol commitment、GraphSet、gray raster、support certificate、
  A replay 和 A/B overlap。

## 第二轮正式证据

两名未参与对方实现的 reviewer 分别从空目录生成 A/A/B：

```text
Python tests                         28 / 28
Lean targets                         15 / 15
Schema bundle                       19 / 19
Support-vs-gray combinations        81 / 81
C2/C3/C4 support certificates       12 / 12
A1/A2 artifact bytes                identical
A/B PNG overlap                     0
Parallel lane failures              0
C4 unenumerated junction witnesses  0
GraphSet reconstruction failures    0
Semantic blockers                   0
```

最终 hash：

```text
schema bundle
0ec53107573e482bd7c9b3942d2ce89974648c903fc93ef01b3fb8b9773bea49

generator source tree
362afbf9c19d8b9d4c87cb38666868cd2141969920a1055996cfcbc2195b421c

protocol bundle
7d90c89617bcae70bacd6d6013df3316122b6b2563f45a0ff4f3eb811fcd129f

public smoke Pack A commitment file
5eb7882e9096a933695fc8cbfd5fe2184213685c1b66b37c94895b532992df99

public smoke Pack B commitment file
2303ae97b29f853628d9a559c3ca13746ec62e1555f5d62bb519da630891912b
```

## 批准边界

P1 只证明公开 smoke profile 与冻结有限 grammar 的 generator/oracle 协议自洽。特别是：

- C4 只覆盖 `single_stroke_template_offset_v0_4`，不是 ChemicalOCR；
- 不覆盖多 stroke 字符、共享端点 glyph、T-touch、glyph crossing 或任意缩写；
- 没有运行 5,632 observation 的 full pack；
- 没有正式 seed、custodian reveal 或 sealed GT 加密；
- 没有 EvidenceCore extractor 或独立 Candidate model；
- 没有 epsilon/Delta 全域计算；
- 没有 USPTO、MR-TADF 或真实论文图片 benchmark。

因此准确结论是：`P1 PASS -> allow P2 commitment only`。`training_authorized=false`。
