# E1 构造性同 tuple 歧义证明

E1 已通过。它给出一个完整父域中的具体合法 tuple，并证明该 tuple 允许两个图不等价的
interpretation。因为渲染只读取 tuple，不读取 interpretation，所以不存在对该场景域处处正确的
确定性单图解码器。

这是一条存在性定理，不是碰撞率估计，也不表示现实化学图片都存在歧义。

## 冻结绑定

```text
v0.6 P0 contract SHA256
e326fc744903b998793746f299232c4225ce3b8648e1abdda7b4e065f06a9402

v0.5 parent contract SHA256
55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65

parent tuple cardinality
36,088,416
```

正式 witness 是父域中的 `tuple_id=18037323`：

```text
family                  c1
half_length             85
motif_ordinal           25
width_px                5
cap                     round
antialias_profile       box_2x2
reflection              none
rotation_index          0
scale_index             3
interpretation_indices  [0, 1]
```

两个解释分别为：

| interpretation | 原子数 | 键数 | 图结构 |
| --- | ---: | ---: | --- |
| 0 | 4 | 2 | 两条内部交叉但不连接的单键 |
| 1 | 5 | 4 | 四条键终止于中心隐式 C |

完整 Gray8 raster 为 `768 x 768`，共 `589824` 字节。两种 interpretation 使用同一个 tuple
render，因此完整字节相同；两个规范图签名不同。

## 文件

```text
produce_constructive_witness_v0_6.py  witness producer
verify_constructive_witness_v0_6.py   canonical bytes 重放验证器
test_constructive_witness_v0_6.py     篡改与非规范 JSON 负向测试
witness.json                          正式构造性 witness
validation.json                       正式重放结果
../schemas/e1_constructive_witness.schema.json
                                      closed Draft 2020-12 schema
```

正式 artifact 哈希：

```text
witness.json
e9f5a423e1031f756be3e6441a18ff0676f5d5f24f84f8160f60e223ce21221d

validation.json
b7ded6a4ee193aaca7f6043c7a3d289bb2af38a6625432aae792aa7505ac8012
```

## 重放

在仓库根目录运行：

```bash
python formal/chemgraph/challenge_v0_6/e1/produce_constructive_witness_v0_6.py \
  --v0-6-contract-sha256 e326fc744903b998793746f299232c4225ce3b8648e1abdda7b4e065f06a9402 \
  --output /tmp/e1-witness.json

python formal/chemgraph/challenge_v0_6/e1/verify_constructive_witness_v0_6.py \
  --input /tmp/e1-witness.json \
  --v0-6-contract-sha256 e326fc744903b998793746f299232c4225ce3b8648e1abdda7b4e065f06a9402

python -m unittest discover \
  -s formal/chemgraph/challenge_v0_6/e1 -p 'test_*.py' -v

cd formal/chemgraph
lake build
```

Lean 入口是 `ChemGraph/StructuredRecovery.lean`。关键结论包括：

- raster-only 结构化提取不能解除 exact-raster collision；
- 同一个合法 `q` 的两个异图 interpretation 否定 universal single decoder；
- crossing/junction 的具体 dependent-domain witness；
- sound + complete CandidateSet 必须保留歧义类；
- sampled ambiguity 可以提升到 full ambiguity；
- sampled singleton 不能推出 full singleton。

所有公开关键定理均执行 `#print axioms`，当前构建输出为空公理集。

## 边界

E1 没有证明 Python rasterizer、PNG encoder、OCR 或神经网络正确，也没有证明有限语法覆盖现实
OCSR 分布。生产字节级跨 tuple 绑定属于 E2；分层发生率诊断属于 S；完整商闭包属于可选 Q。

`training_authorized=false`。E1 通过不授权训练。
