# G3 P2-H Atom Query Set Capacity64

> 日期：2026-08-04
>
> 状态：协议通过 / Count机制GO / Pointer集合容量NO-GO

## 受控问题

P2-G确认可恢复route上的top128中心候选已经全覆盖，但P2-E逐候选boundary排序不能泛化。P2-H
不再给128个候选分别贴固定正负标签，而是引入128个Atom query：

1. query与GT Atom通过整图Hungarian匹配；
2. query预测存在性和全图Atom Count；
3. pointer目标是“该GT 6px内任一兼容候选”的集合NLL，不固定某个peak身份；
4. 推理时用全局一对一assignment把保留query映射到不同候选。

这里的6px只与既有评测匹配半径保持一致，用于定义等价监督集合；没有按该半径聚类候选，也没有
在推理中删除近邻。模型参数291,394，输入是冻结P2-A stride-1 top128 cache。

## Capacity64数据

只从train5k选64张完全可达图片，不使用route或MR-TADF。两个域各32个不同分子，且四种布局各8张：

| 域 | 选择规则 | 原始score-prefix matched | exact |
|---|---|---:|---:|
| easy32 | 原始排序已整组正确 | 967/967 | 32/32 |
| hard32 | 原始排序recall最低 | 695/807 | 0/32 |

这是同图容量实验，不代表泛化。训练4,800 steps、batch 4、19,200 exposures、300轮，AdamW
`3e-4`，BF16，两个域交替。

## 结果

| 域 | Count exact | known-count matched/exact | predicted-count matched/exact |
|---|---:|---:|---:|
| easy32 | **32/32** | 938/967，9/32 | 938/967，9/32 |
| hard32 | **32/32** | 764/807，8/32 | 764/807，8/32 |

Count在64/64图全部正确，证明全局query集合能学习整图基数；但pointer没有通过容量门。hard32相对
原始排序提高69个Atom和8张exact，同时easy32从32/32退到9/32，不能部署也不能扩到5k。

训练末期Count loss约0.0049、coordinate loss约0.0050、pointer loss约0.274，低loss与低exact
并存，说明目标与离散解码之间存在协议缺口，而不是简单欠训练。

## Pointer离散消融

固定final checkpoint，对已知Count的保留query比较三种输出：

| 输出 | easy32 | hard32 |
|---|---:|---:|
| pointer概率加权连续坐标 | 556/967，0/32 | 510/807，3/32 |
| 每query独立argmax pointer | 815/967，0/32 | 692/807，4/32 |
| 全局唯一pointer assignment | **938/967，9/32** | **764/807，8/32** |

独立argmax在easy/hard分别发生148/102次重复候选。训练时每个query只要求把概率放到自己的任意
兼容峰，多个query可以偏好同一候选；推理时才强制一对一，于是未受训的次优pointer被选中。

## 决策

- 正式结论为`atom_query_set_capacity_no_go`。
- Count机制保留；现有pointer loss停止，不追加epoch或直接扩大数据。
- P2-I保持相同模型、数据、初始化和预算，只把pointer监督改成训练期全局唯一
  `query -> compatible candidate` matching，使训练约束与推理解码一致。
- 本实验产生正式checkpoint，因此创建与模型名相同的annotated tag；NO-GO权重只用于复现。

```text
model = chemvector-g3-p2h-atom-query-set-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2h-atom-query-set-capacity64/step-004800.pt
SHA256 = 168c43a8dadf5f21a8c1f9eae91ae52e3e6f3a6d892eb467014a2e14d542efe5
training images = 64
samples seen = 19,200
decision = NO-GO
```

复现：

```bash
.venv-train/bin/python -m \
  chemvector.training.atoms.train_g3_p2h_atom_query_set_capacity64
```
