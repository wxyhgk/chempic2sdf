# G3 P4-E0 Source Union Cycle Manifests

> 日期：2026-08-07
>
> 状态：预注册，待执行manifest审计
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
