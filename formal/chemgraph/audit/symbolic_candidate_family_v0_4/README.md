# Symbolic Candidate Family v0.4 容量探针

这是 v0.4 的有限域 CSP 表示实验，不是图片 proposer、生产 schema 或训练代码。

## 要验证的问题

v0.3 最多显式保存 128 个 interpretation。若一张图存在 8 个相互独立的二元歧义，则可行解释数为：

```text
2^8 = 256
```

显式列表必然溢出。符号表示只保存 8 个二元变量和约束，仍可精确定义全部 256 个满足赋值。

`symbolic_family.py` 实现一个 closed 有限域 table-CSP：

- 变量 ID 连续，每个 domain 大小为 1..16；
- constraint scope 升序、唯一；
- allowed tuple 完整、规范排序、不能越域；
- assignment 是 solution 当且仅当满足所有 table constraint；
- `materialize(capacity=128)` 用于证明显式展开会溢出，不是实际解码接口。

## 已证明与未证明

该探针能证明表示容量差异：8 个二元变量的符号文档很小，但解集恰有 256 项。

它没有证明：

- 真实图片存在 8 个独立歧义；
- v0.4 的变量和 constraint 已能从 PNG/root universe 独立重建；
- table-CSP 是最终最优格式；
- 两个满足赋值编译出的化学图是否同构；
- SAT/CSP 求解器的 proof certificate 已被独立验证。

因此下一步是先冻结 root constructor 和约束语言，再决定使用 table-CSP、CNF、BDD/ZDD 或带 proof trace 的
SAT，而不是直接把本探针接入模型。

## 运行

```bash
cd /home/zhenghe/GK/Project/chempic2sdf
.venv-train/bin/python -m unittest \
  formal.chemgraph.audit.symbolic_candidate_family_v0_4.test_symbolic_family -v
```

当前仍保持：

```text
proposer_public_execution_authorized=false
unique_graph_authorized=false
training_authorized=false
```
