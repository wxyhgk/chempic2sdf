# G3 P2-R P2-J Identity Existence Capacity64

> 日期：2026-08-04
>
> 状态：固定身份机制确认 / 65参数weighted BCE容量NO-GO

## 实验

P2-R加载P2-J final，用P2-J原始`existence_cost_weight=1.0` Hungarian只生成一次固定
query-positive Action Cache。缓存覆盖64张图、每图128位query mask，共1,774个正query；重复生成
完全一致。随后冻结encoder、decoder、pointer与Count，从P2-J原65参数existence head继续训练，
1,600步中不再调用Hungarian，唯一目标为固定标签weighted BCE。

| 域 | P2-J final | P2-R final | Count exact |
|---|---:|---:|---:|
| easy32 | 963/967，28/32 | **964/967，29/32** | 32/32 |
| hard32 | 798/807，26/32 | **799/807，27/32** | 32/32 |

结果各提高1张，但未通过严格32/32容量门。

## 残差归因

final checkpoint上额外比较existence top-N与固定Action Cache：

| 域 | top-N identity exact | identity overlap | 强制cache identity解码 |
|---|---:|---:|---:|
| easy32 | 22/32 | 955/967 | **967/967，32/32** |
| hard32 | 25/32 | 795/807 | **807/807，32/32** |

固定identity一旦被选中，pointer与唯一候选解码可以完整恢复两域全部原子。剩余容量断点只在65参数
existence head的整组排序：weighted BCE降低平均分类损失，但不保证每个正query都排在全部负query之前。

## 决策

- Action Cache机制成立，P2-M至P2-Q的identity协议干扰已经排除。
- `p2j_identity_existence_capacity_no_go`：相同head与weighted BCE不再追加epoch。
- P2-S复用同一Action Cache和P2-J初始化，增加覆盖全部正负query对的平滑pairwise排序损失；禁止回到
  动态min-positive/max-negative极值目标。

```text
model = chemvector-g3-p2r-p2j-identity-existence-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2r-p2j-identity-existence-capacity64/step-001600.pt
checkpoint SHA256 = 2e679e9264b00c6c15bc8f4d1adfe3ac6bd7adaea20e607e3580f0e1fd2120f5
Action Cache SHA256 = d2134087c5c1d79cb698a3aefa90fcfeb6e3defeba3e602e1e1317a5362e8aca
training images = 64
calibration samples seen = 6,400
trainable parameters = 65
decision = NO-GO
```
