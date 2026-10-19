# r16 芳香专项 split 与 CPU 训练顺序

## 先确认当前问题

`outputs/audits/chemvector-atom-aromatic-head-cpu-r16/report.json` 的
fresh holdout 有 95 张图、884 个 atom rows，但芳香正例为 0；因此 `.770`
accuracy 不能作为芳香泛化结果。对当前 r6 rows 的只读复核见
`outputs/audits/chemvector-aromatic-identity-split-r16-audit/report.json`：
fresh 全部落在 `TRIPLE_CONTROL`，不是模型学坏，而是 molecule-level
stratum 分配造成的覆盖缺口。

## 数据构造

沿用 `Data/sdf/chemvector_curriculum_v0_2.catalog.entries()`，先按
canonical SMILES 计算 `connection_identity`，排除既有 r5 与 6k/r6
身份，再做以下 molecule-level strata：

- `AROMATIC_CARBON`：苯环/取代苯环；
- `HETEROAROMATIC_NSP`：含 N/S/P 的芳香环；
- `FUSED_AROMATIC`：两个或以上共享环原子的芳香环；
- `AROMATIC_MIXED_MULTIPLE`：芳香环外同时含 double/triple 控制；
- `DOUBLE_CONTROL`、`TRIPLE_CONTROL`：非芳香双/三键对照。

在 profile 扩展之前，以 `sha256(connection_identity)` 稳定排序，并在每个
stratum 内按 molecule 分成 train/dev/fresh holdout（建议 20/8/8 个
molecules，四个渲染 profile 各生成一张）。这样四种 profile 不会把同一
molecule 泄漏到不同 split；审计同时检查 molecule_id、connection_identity
和 canonical_smiles 三层交集为零。

当前 catalog 排除后没有真正 fused aromatic（现有唯一双环候选是非稠合的
biphenyl-like 结构）；不能声称已有数据足够。应先补充至少 12 个独立
fused identities（例如 naphthalene/indole/benzofuran/quinoline 等，经
RDKit sanitize 与 identity 排除），否则 fused cell 标为 unavailable，
不得用普通芳香环或非稠合双环替代。

## 最小 CPU 顺序

1. 生成 metadata-only manifest；审核身份交集和每个 stratum 的配额。
2. 渲染四 profile；检查 render/hash/graph-SDF identity，失败样本剔除并
   保持 split 不重排。
3. 只读 deterministic coverage：报告 aromatic C/N/S、fused、mixed 和
   DOUBLE/TRIPLE controls 的 atom/bond 正例数；任一 fresh cell 为 0 即停。
4. 冻结 production initial 与视觉 trunk，CPU 训练 257/385 参数 aromatic
   head，最多 300 steps；阈值只从 train 选，fresh 仅一次评估。
5. 若 fresh 每个非空 cell 的 positive recall 与 precision 达到预注册门槛，
   再训练 direct pair residual，最多 500 CPU steps；否则停止，不启动 GPU。

GPU 条件是两个 fresh gate（atom 与 pair/graph）均通过，并且 graph-level
   dangling、edge precision/recall 也通过；本方案本身不授权 GPU 或生产合并。
