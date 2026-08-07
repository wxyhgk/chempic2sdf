# G3 P4-E0 Source Union Cycle Manifests

> 日期：2026-08-07
>
> 状态：已完成，manifest身份门通过
>
> 审计：`chemvector-g3-p4e0-source-union-cycle-manifests`

## 周期目标

P3-E33确认train5k/train25k source-id并集在256容量可使Micro160 candidate reachability达到
160/160，但此前所有selector都只在Micro160反复训练，不能证明泛化。本周期冻结256 union输入，
将训练与所有旧周期门集彻底隔离后，比较两条路线：

- source-aware 256-candidate set selector；
- 继续训练center field，使互补真实峰回到更小candidate预算。

## 数据边界

源为`style_train_100k`。先排除P3-E21 train/dev/confirm共1120个分子，以及与它们任一相同的
scaffold，再分配：

- `pilot_train`：5,000分子、20,000图，是full_train的固定子集；
- `full_train`：20,000分子、80,000图，pilot通过后才能使用；
- `confirm_v2`：175分子、700图，Count桶分配50/50/50/20/5，每个分子保留四variant。

confirm_v2优先分配且内部scaffold唯一；full_train与confirm_v2 molecule/scaffold均为0重叠。
三个manifest与旧P3-E21全部split molecule/scaffold均为0重叠。manifest必须按tar物理顺序书写，防止
替换样本的CSV顺序与tar顺序不一致。

## 门禁

- pilot/full/confirm图数严格为20k/80k/700；
- pilot molecule集合严格是full子集；
- train与confirm_v2 molecule/scaffold 0重叠；
- 三者与旧周期全部split molecule/scaffold 0重叠；
- 每个分子四variant完整，source commitment与CSV SHA固定；
- 不读取fresh_dev、USPTO、MR-TADF或sealed_test。

本轮只建立manifest，不训练、不创建tag。通过后才允许构建pilot20k双源cache。

## 实际结果

全部门禁通过：

| split | 分子 | 图片 | scaffold | stress图片 |
|---|---:|---:|---:|---:|
| pilot_train | 5,000 | 20,000 | 4,737 | 5,000 |
| full_train | 20,000 | 80,000 | 17,472 | 20,000 |
| confirm_v2 | 175 | 700 | 175 | 175 |

pilot严格为full molecule子集；full与confirm_v2 molecule/scaffold均0重叠；三个split与旧P3-E21
train/dev/confirm的1120个molecule、1101个scaffold均0重叠。confirm_v2 Count桶严格为
50/50/50/20/5。pilot的四种stress operation为1237/1231/1254/1278，没有单一增强主导。

产物SHA256：

- audit：`50d46024cd07d8313590de85903c5795e9919a7f7dfa5cbbdc7beeb5e4f565e3`；
- pilot_train：`22a03e2b2e37b41b7b298a1fec324fbe383e0f09c49e173a39c8d90347636daa`；
- full_train：`c1c787a646b5827ae72accd62dbe791ba2beb0a866d88741be3d79d26d114795`；
- confirm_v2：`8a832a483b7e52f224ecc21eef82cbb4b7b4a62907cb471e5de9083764f90600`。

正式状态`passed`，允许进入P4-E1 pilot20k双源cache。未读取fresh、外部或sealed_test；零训练，
不创建tag。
