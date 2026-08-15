# OCSR 数学与结构建模文献地图

日期：2026-08-15

## 结论

现有文献支持我们采用“逆问题门禁 + line object + set/graph prediction”的组合路线，但没有任何一篇论文能替代
本项目对 raster fiber、候选完整性和化学图唯一性的逐层证明。

## 已核对方向

| 方向 | 文献 | 对本项目的作用 | 不能证明 |
|---|---|---|---|
| 数据驱动逆问题 | Arridge et al., *Solving inverse problems using data-driven models*, DOI `10.1017/S0962492919000059` | 区分存在性、唯一性、稳定性与学习先验 | 具体化学 raster 是否可辨识 |
| 线段检测 | LSD, DOI `10.1109/TPAMI.2008.300` | 提供线段候选的经典确定性/统计基础 | 线段如何组成化学 Bond |
| Wireframe parsing | *Learning to Parse Wireframes*, DOI `10.1109/CVPR.2018.00072` | 支持 line/junction 作为结构对象 | 文字与键的化学歧义 |
| Transformer line set | LETR, DOI `10.1109/CVPR46437.2021.00424` | 支持直接预测线段集合 | 候选集合 sound/complete |
| Image-to-graph | Relationformer, DOI `10.1007/978-3-031-19836-6_24` | 支持集合级节点/关系联合建模 | OCSR 的 raster collision 与价态唯一性 |
| OCSR graph | MolGrapher, DOI `10.1109/ICCV51070.2023.01791` | 提供化学图预测基线与数据规模参照 | 长键/交叉候选被启发式提前删除的问题 |
| OCSR sequence | MolScribe, DOI `10.1021/acs.jcim.2c01480` | 提供图到字符串/结构序列基线 | SDF/图解释在像素层是否唯一 |

## 对路线的约束

文献可以帮助选择模型归纳偏置，但不能授权以下跳跃：

```text
line detector recall high
!= root candidate universe complete

per-edge accuracy high
!= whole graph uniquely identified

one valid RDKit molecule
!= raster only支持这一种分子
```

因此继续使用当前门禁：先证明每层 finite root constructor 的 soundness、completeness、canonicality 和容量，再把
该层作为后续 CSP 的权威输入。模型训练只能优化已冻结协议中的可学习部件，不能替代协议证明。
