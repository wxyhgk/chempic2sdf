# G3 P2-I Unique Pointer Set Capacity64

> 日期：2026-08-04
>
> 状态：协议通过 / Pointer机制接近容量 / Query存在性NO-GO

## 受控改动

P2-I与P2-H保持完全相同的291,394参数架构、64张图、初始化SHA、4,800 steps、19,200
exposures和优化器。唯一改动是pointer监督：训练期先把query匹配到GT，再在整张图上运行一次
Hungarian，为每个匹配query分配互不重复的兼容候选。这样训练和推理都要求一对一pointer。

```text
shared initialization SHA256 = 6566dd14c8d4054c1e919281955ab938c26c5e09ef96433da6a60946ffa85d79
```

没有候选聚类、固定peak正样本身份、键长裁剪或MR-TADF数据。

## 结果

| 域 | P2-H matched/exact | P2-I matched/exact | Count exact |
|---|---:|---:|---:|
| easy32 | 938/967，9/32 | **943/967，12/32** | 32/32 |
| hard32 | 764/807，8/32 | **768/807，5/32** | 32/32 |

唯一pointer目标提高两个域的Atom matched，easy增加3张exact，但hard exact减少3张，仍未通过
容量门。不能追加相同epoch或扩到5k。

## GT-linked Query只读审计

固定final checkpoint，用GT Hungarian只选择本轮与真实Atom匹配的query，再运行完全相同的唯一
pointer解码。该审计只用于定位，不是部署指标：

| 模型 | 域 | top-N existence与GT-linked query重合 | GT-linked pointer matched/exact |
|---|---|---:|---:|
| P2-H | easy32 | 881/967，91.11% | 966/967，31/32 |
| P2-H | hard32 | 744/807，92.19% | 800/807，27/32 |
| P2-I | easy32 | 893/967，92.35% | **967/967，32/32** |
| P2-I | hard32 | 753/807，93.31% | **802/807，28/32** |

这证明唯一pointer在正确query集合上已经接近解决；主要损失来自existence top-N选错query。
当前query-to-GT Hungarian使用pointer概率加权平均坐标，而P2-H审计显示该连续坐标只有约
57%--63% matched。用一个本身不可靠的平均坐标定义query身份，使存在性标签在训练中漂移。

## 决策

- 正式结论：`unique_pointer_set_capacity_no_go`。
- 保留Count和全局唯一pointer；停止当前坐标Hungarian目标。
- P2-J保持相同架构、数据、初始化和预算，将query-to-GT匹配代价改为GT兼容候选上的pointer
  log-probability mass，使query身份、pointer训练和推理解码使用同一证据。

```text
model = chemvector-g3-p2i-unique-pointer-set-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2i-unique-pointer-set-capacity64/step-004800.pt
SHA256 = ad6a976aff6fb52c453e5bd636bf07d540d67d439d70228b9c8cd1a12d222c8b
training images = 64
samples seen = 19,200
decision = NO-GO
```

复现：

```bash
.venv-train/bin/python -m \
  chemvector.training.atoms.train_g3_p2i_unique_pointer_set_capacity64
```
