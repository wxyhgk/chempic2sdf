# Fragment Root Constructor v0.4 微场景预注册

日期：2026-08-15

## 目标

先验证一个比“识别化学图”更窄的命题：给定 canonical base stroke 有限集合，能否不依赖文字/键分类，确定性重建
全部几何交点、split site 和 atomic fragment。

表示采用精确有理数。Q4 只约束输入端点；交点和沿 parent 的参数 `t` 保存为约分后的
`[numerator, denominator]`，禁止浮点求交和先舍入后去重。

## 预注册 fixture

| 场景 | 预期 |
|---|---|
| X | 1 个交点、2 个 split site、4 个 fragment |
| T | 1 个交点、1 个 split site、3 个 fragment，无零长 child |
| Double-X | 2 个交点、4 个 split site、7 个 fragment |
| Three-way | 1 个去重交点、3 个 split site、6 个 fragment |
| Collinear positive overlap | closed failure，不输出部分 root |

## 数学边界

对一条 parent stroke，若内部 exact cut 参数为严格有序的：

```text
0 < t1 < ... < tn < 1
```

则 delimiter 列表 `[0,t1,...,tn,1]` 的相邻对恰好产生 `n+1` 个非零 atomic fragment。这个定理只相对
base stroke 与 exact intersection constructor，不证明 PNG 到 base stroke 的召回率。

## 明确不做

- 不判断 crossing/junction/touch 的化学语义；
- 不判断 glyph/bond 或 primitive kind；
- 不构造 support RLE、width、ownership、port、lane、Bond；
- 不授权 public proposer、UniqueGraph、SDF 或训练。

当前 Python 实现只是独立研究探针；必须补 Lean 区间分割定理、小域暴力参考和 payload constructor 后，才能评审
是否把这一层写入 v0.4 closed schema。
