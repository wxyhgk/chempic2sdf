# G3 P1-L 最小 Component 替换审计

> 日期：2026-08-03
>
> 状态：已完成，选择一跳action verifier
>
> 性质：P1-K final零训练结构化可达性审计

## 唯一问题

P1-K known-count train仍有1,097张失败图。P1-L对每图固定top-M selected component和全部
target-free component，建立component representative到representable GT的6px兼容图。若全部
component存在完整匹配，则给selected component成本0、unselected成本1，求完整GT匹配的最小
成本；该成本就是达到整图exact至少需要替换的selected component数。

## 固定协议与判读

- 固定P1-K final `69243466...3d4ed`、P1-A、vision、3px component与base-softmax anchor；
- 只读train5k与route100，不训练、不读取外部/封存数据；
- 报告0/1/2/3/4/5+最小替换分布、component oracle不可达图、Atom recall和失败GT兼容多重度；
- 只在component oracle可达的失败图上判读：
  - 单替换占比≥70%：下一步一跳action verifier；
  - 单/双替换累计≥90%：下一步bounded two-move set corrector；
  - 否则：需要全局component set目标，不做局部repair。

route100只作迁移诊断，不参与路线选择。P1-L不保存checkpoint、不生成SDF、不接前端。

## 正式结果

P1-K final在train5k与route100完成零训练重放，保存的known-count exact/recall逐项复现。

| 最小替换数 | Train图数 | Route图数 |
|---:|---:|---:|
| 0（当前已exact） | 3,903 | 68 |
| 1 | 783 | 22 |
| 2 | 183 | 7 |
| 3 | 64 | 2 |
| 4 | 25 | 0 |
| 5+ | 26 | 1 |
| Component oracle不可达 | 16 | 0 |

train共有1,081张oracle可达失败图，单替换占`783/1,081 = 72.43%`，单/双累计89.36%；
route 32张失败图中单替换68.75%，单/双累计90.63%。train当前漏匹配1,634个GT，其中按直接
selected兼容性统计的1,614个失败GT里，591个只有一个兼容component，1,018个仍有多个兼容
component；剩余是共享component引起的全局匹配冲突。

## 决策

train单替换比例超过预注册70%门：

```text
decision = one_move_action_verifier
```

不能直接执行oracle替换，推理时没有GT。下一步先审计正swap是否落在最低分selected与最高分
unselected的有限边界带；若top-K边界覆盖足够，再训练候选级安全verifier，并把3903张当前
exact图上的零有害替换作为硬门。若边界覆盖不足，才考虑全pair或two-move。

## 产物

```text
outputs/audits/chemvector-g3-p1l-minimum-component-replacement/audit.json
SHA256 52b09f6128710f9145b1e5f9da7df1f41a491e82addcbac1dd2e66a37d8cca74

审计源码
SHA256 48972b64f7e8fbf4eed468b91029bb97bfa417b932048e48eac77e6d246a0a88
```
