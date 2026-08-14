# ChemGraph Hidden Challenge Generator v0.4

本目录实现一个 **generator-first、只依赖 Python 标准库** 的有限隐藏挑战生成器。它的目的不是
训练模型，也不是证明真实化学图片已经可辨识，而是在训练前先构造一组可重复、可审计的点线观测，
验证“同一可见墨迹可能对应多个化学解释”以及候选集合协议是否自洽。

本轮只实现生成器、固定点 rasterizer 和 smoke 自测。没有选择正式 seed，没有生成或保留正式 hidden
pack，也没有读取未来 extractor。

## 文件职责

- `rasterizer_v0_4.py`：Q16.16 整数几何、线段覆盖、整数 supersampling、round-half-to-even 和
  canonical grayscale PNG 编码。
- `generate_pack_v0_4.py`：HMAC-SHA256 counter PRNG、S1--S8/C1--C4 family、几何变换、pack/GT
  隔离、拒绝采样日志和 hash commitment。
- `sample.schema.json`：未来 extractor 调用清单中单条公开记录的最小 schema。
- `pack_reveal.schema.json`：reveal 后 observation 与隐藏 GT 的连接记录 schema。
- `schemas/`：独立 oracle agent 维护的完整协议 schema。本生成器不修改该目录。
- `test_challenge_v0_4.py`：stdlib `unittest`，覆盖 PNG、重复生成、泄漏边界和 exact collision。

## 数学与数值约束

- 画布固定为 `768 x 768`，坐标固定为 Q16.16，`Q = 65536`。
- 变换顺序固定为 `reflection -> rational rotation -> uniform scale -> translation`。
- 旋转只使用 Pythagorean 有理矩阵，不调用浮点三角函数。
- scale 取自 `1/2, 5/8, 3/4, 1, 5/4, 3/2, 2`；reflection 取自 `none/x/y`。
- 线宽为整数 `1..9 px`；cap 为 `butt/round/square`。P1 只实现 `round join`：两个以上
  primitive 共享完全相同的 Q16 端点时，rasterizer 显式 union 一个整数 round disk，即使 cap 是
  butt/square 也会生效。miter/bevel 留到后续版本，不能作为本版可见 style 声称。
- AA 支持 `binary_1x`、`box_2x2`、`box_4x4`。coverage 合并使用 supersample bitmask，重叠线段
  不会重复累加；coverage 到 uint8 使用整数 ties-to-even。
- PNG 为 8-bit grayscale（color type 0），每行 filter 固定为 0，只含 `IHDR/IDAT/IEND`，禁止
  metadata 和尾随字节。

每个 family 由 metamorphic quartet 组成。quartet 内基础 motif 和 style 相同，只改变几何变换。style
按 quartet 轮转平衡；因此正确的约束是“每个 style 桶的 quartet 数差不超过 1”，换算为 observation
后差最多为 4，而不是 observation 计数差必须不超过 1。

## Family

完整 pack 有 5,632 个唯一 observation：S1--S8 各 512 个，C1/C2/C3/C4 分别为
512/384/256/384 个。C family 定义如下：

- C1：相同十字墨迹对应 disconnected crossing 或中心 degree-4 junction。
- C2：相同两条平行 lane 对应 order-2 bond 或两个 disconnected singles。
- C3：相同三条平行 lane 对应 order-3 bond 或三个 disconnected singles。
- C4：相同单 stroke 文字形（`I/l/1/slash`）对应 atom glyph 或一条 chemical bond lane。

P1 刻意不覆盖 `H/N/A/M/W/X/K/V`、`Cl/Br/Ph/CH3/NH2` 等复合 stroke 字符，也不声称完成 OCR。
这些字符含共享端点、T-touch 或 crossing，如果只手写“文字/全部独立线”两个解释，CandidateSet 会
不完整；必须等独立 oracle 能枚举接触事件后再加入。

四种单 stroke 模板只用于隐藏 GT 的渲染诊断；text interpretation 的图标签统一为
`TEXT_STROKE`。`I/l/1/slash` 不进入 candidate graph signature，也不出现在公开 manifest。这样模型
不能通过相对主键的 offset 或方向旁路，把模板 ID 当作 OCR 结果输出。

C2/C3 的每个最终变体，以及 C4 的 chemical/glyph primitive，在 accept 前都会使用与正式 rasterizer
相同的采样规则分别生成 foreground support。任意两者共享前景像素或发生 8-neighbor adjacency 时，
该 attempt 以 `invalid_exact_collision` 拒绝，不能缩减配额。通过的详细 pair certificate 只写入
`sealed_gt/observations.jsonl`。

**实现上的硬约束是：每个 unique observation 只调用一次 rasterizer。** C1--C4 的多个
interpretation 共享同一个 `InkProgram`、同一个 PNG hash 和同一个 production grayscale hash。所谓重复调用
只是在 `samples.jsonl` 中重复引用同一 PNG，不会重新按不同语义渲染。

full pack 另含全部 1,536 个 collision replay 和 512 个 strict replay，共 7,680 次 invocation。
`--smoke` 每个 family 生成 4 个 observation，并选择 8 个 strict replay，共 48 个 observation、72 次
invocation。

## 输出隔离

```text
OUTPUT/
  config.json
  commitment.json
  sealed_image_pack/
    images/img-XXXXXXXX.png
    manifest.json
    samples.jsonl
    pack.json
  sealed_gt/
    ink_programs.jsonl
    transforms.jsonl
    interpretations.jsonl
    graph_sets.jsonl
    observations.jsonl
    rejection_log.jsonl
    gt.json
```

`sealed_image_pack/manifest.json` 和 `samples.jsonl` 不含 family、transform、InkProgram、GT、ambiguity
kind 或 seed。未来 extractor 只能接收单张 PNG 的 stdin bytes；manifest 本身不能进入 extractor
sandbox。

`sealed_gt/` 是逻辑隔离目录，不是密码学加密。本地 smoke 可以直接检查它；正式挑战发布前必须由
独立 custodian 使用冻结的加密协议加密，明文目录不能交给 extractor 作者。

`commitment.json` 保存 generator source-tree、protocol bundle、配置、公开 manifest、GT index、拒绝日志和 image tree
的 SHA256 commitment。protocol bundle 递归覆盖顶层 Python、README、顶层 schema 和 `schemas/*.json`。
它只保存 domain-separated seed commitment，不保存 raw seed。生成采用临时目录
和原子 rename，并拒绝覆盖非空输出目录。

## Smoke

下面的全零 seed 只允许作为公开测试向量，绝不能作为正式 challenge seed：

```bash
cd /home/zhenghe/GK/Project/chempic2sdf/formal/chemgraph/challenge_v0_4
python3 generate_pack_v0_4.py \
  --smoke \
  --pack-id pack-a \
  --seed-hex 0000000000000000000000000000000000000000000000000000000000000000 \
  --output /tmp/chemgraph-v0.4-smoke
python3 -m unittest -v test_challenge_v0_4.py
```

测试结束后应删除 `/tmp` 输出。正式运行必须由两个 custodian 在 extractor 冻结后导出新的 256-bit
seed，并分别对 Pack A/B 使用域分离 seed；本代码不会替操作者生成、选择或保存正式 seed。

## 声明边界

生成器能保证的是有限 grammar 下的结构和字节协议。它不能证明 seed 在现实中未泄漏、SHA256 绝对
无碰撞、发布时间顺序真实、oracle 候选集合完整，或模型已经理解化学。以上事实仍需独立 custodian、
oracle/verifier、隔离执行日志和 reveal 后重建共同审计。在这些门全部通过之前，不应据此授权训练。
