# EvidenceCore v0.4 Candidate

本目录实现 P3 的第二个独立进程：

```text
canonical EvidenceCore JSON -> anonymous CandidateGraphSet JSON
```

它不读取图片，也不判断元素字符。输入接口逐字节镜像公开 extractor schema，冻结 SHA256 为：

```text
f4624ef3971373cad5ceb78562ab99fd549bacc7920472c93177da0a3ecd12ed
```

## 文件

- `candidate.py`：只依赖 Python 标准库的几何解释、候选展开和匿名图同构去重。
- `evidence_core.schema.json`：公开 extractor EvidenceCore schema 的逐字节副本。
- `candidate_set.schema.json`：Candidate 输出的 closed schema。
- `fixtures/*.evidence.json`：手写整数 Q4 Evidence，不来自 PNG、pack 或 GT。
- `test_candidate.py`：几何、容量、canonical JSON、进程边界和依赖闭包测试。

## 输入边界

Candidate 运行时只从 stdin 读取一个不超过 65,536 bytes 的 canonical JSON。根协议必须为：

```text
schema_version = chemgraph-evidence-core-v0.4
status = ok
image = 768 x 768, Q4-pixel
```

primitive/event ID 必须各自唯一并覆盖 `0..N-1`，数组顺序不影响结果。Candidate 会在内存中按 ID
重排，并把每根 primitive 的两端规范为字典序方向；incident ID 数组也会规范排序。输入拒绝未知字段、
浮点数、NaN、非 canonical 编码和超容量数组。

## 几何规则

全部规则只读取 EvidenceCore 中的整数几何和中性事件：

1. 端点以 Q4 Chebyshev 距离不超过 `16` 的单链接聚类形成 anonymous implicit node。
   `junction` event 只会合并 event 引用 primitive 中距质心最近的端点，且该端点必须落在
   `max(16 Q4, primitive width)` 容差内。
2. 两条线段发生严格内部 proper crossing 时保留两个候选：
   - 两根线互不连接；
   - 交点成为 degree-4 junction，两根线分别在此拆分。
3. 两条或三条 straight/open primitive 只有同时满足以下条件才构成 multiplicity ambiguity：
   - 环形 `orientation_bucket` 差不超过 `3/720` 圆周；
   - 向量同向，且夹角正弦不超过 `1/64`；
   - 两侧端点的轴向投影误差均不超过 `24 Q4`；
   - 对应端点没有被聚成同一 node。

   输出 bundle order 2/3 与 disconnected singles 两个候选。非对齐平行线只保留 singles。
4. `glyph_like` event 必须只引用一根不与其他 node 共端点的 primitive。输出：
   - 该 primitive 是普通 order-1 bond lane；
   - 删除该 lane 的两个 implicit 端点，加入一个孤立 `text_stroke` sentinel node。

   不输出 `I/l/1/slash` 或任何 OCR 字符。
5. chain、fan、ring 没有触发上述歧义时只输出一个图。
6. 多重 crossing 共享 primitive、closed/非直线 primitive、互相连接的多个 bundle group、重复/self-loop
   anonymous edge、超过 6 个二元歧义轴、节点/边/候选超限时返回空候选和 `Reject`，不猜测、不 repair。

## 匿名图去重

`candidate.py` 自实现 color refinement 加回溯图同构。比较内容只有：

- node role：`implicit/text_stroke`；
- 无向邻接关系；
- edge order：`1/2/3`。

node 编号、edge 顺序和 primitive ID 不参与图同构。Candidate 不调用仓库中的任何 canonicalizer。

## 输出

```json
{
  "candidates": [],
  "decode_kind": "Reject",
  "schema_version": "anonymous-candidate-set-v0.4"
}
```

- 0 个候选：`Reject`
- 1 个候选：`Graph`
- 2--64 个非同构候选：`Ambiguous`

每个图最多 128 nodes、128 edges，node ID 从 0 连续编号，输出同样使用 canonical JSON。

## 运行

```bash
python3 formal/chemgraph/evidence_core_v0_4/candidate/candidate.py \
  < formal/chemgraph/evidence_core_v0_4/candidate/fixtures/crossing.evidence.json

python3 -m unittest -v \
  formal.chemgraph.evidence_core_v0_4.candidate.test_candidate
```

## 已知协议阻塞

EvidenceCore 对 `glyph_like` 只提供 event kind、score、bbox 和一根 open/straight primitive。如果一根普通短键与一个
单笔画文字模板在几何、style 和 event score 上相同，Candidate 看到的 EvidenceCore 完全相同，却被要求输出
`Graph` 与 `Ambiguous` 两种不同答案。线宽、bbox 膨胀、残差像素或特定角度都只是 style/family 相关特征，
不能作为 unknown-style 语义证明。因此本实现只对 `score >= 9000` 的高置信 event 展开文字/键线候选；
对同分但真值相反的公开 family，必须修订 Evidence 协议或 challenge 真值，不能在 Candidate 中写入样式旁路。

## 公开评测

`/tmp/v04-second-a` 公开 48 张 challenge 的最终结果为 `44/48 candidate-set exact`，extractor 和
Candidate replay 均为 byte exact：

- `S1--S8`：各 `4/4`。
- `C1--C3`：各 `4/4`。
- `C4`：`0/4`，因为它与 `S8` 输出同类 `glyph_like/7500` Evidence，但真值要求相反。

因此公开评测总状态仍是 `failed`，P3 不应被标记为通过。`C4` 是 Evidence/challenge 协议 NO-GO，
而不是需要继续放宽 Candidate 几何阈值的失败。

## 当前边界

本实现的规则由手写 finite geometry fixtures 固定。尚未覆盖曲线、楔形键、虚线、
芳香圆、颜色、多笔画文字、缩写 OCR、一个 primitive 上的多个交叉或真实图片噪声。通过本目录测试不等于
P3 通过，不授权训练，`training_authorized=false`。
