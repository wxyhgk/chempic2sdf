# G3 P1-J Component Representative 审计

> 日期：2026-08-03
>
> 状态：已完成，base-softmax representative GO
>
> 性质：零训练、3px component输出坐标审计

## 唯一问题

P1-I证明3px target-free component内部保留99.992%的train GT覆盖，但“任一member接近GT”不是
可直接输出的单一Atom坐标。P1-J比较四种完全target-free representative：

1. `base_max`：component内P1-A base logit最高的真实candidate anchor；
2. `medoid`：component内到其他member总距离最小的真实candidate anchor；
3. `base_softmax`：按component内base logit softmax加权的anchor；
4. `centroid`：member anchor算术平均。

## 固定统计与选择

- 固定P1-A、vision、top-128和P1-I选择的3px连通分量；
- 只读train5k与route100，禁止外部和封存数据；
- 对每种representative报告所有component对representable GT的6px一一覆盖；
- 同时以component内最大base logit作为component score，选择top-M，报告零训练known-count几何
  recall与exact；
- representative选择只看train5k reachability：先最大化Atom recall，再最大化exact，完全相同时
  按`base_max > medoid > base_softmax > centroid`固定优先级；route100只做迁移诊断。

选中方案必须在train与route同时达到reachability Atom recall≥99.9%、exact≥99%，否则单坐标
component NO-GO，必须保留member pointer。P1-J不训练、不保存checkpoint、不生成SDF。

## 正式结果

train5k与route100只读审计完成，`updates=0`，没有读取禁止数据。

| Representative | Train reach recall/exact | Train base top-M recall/exact | Route reach recall/exact | Route base top-M recall/exact |
|---|---:|---:|---:|---:|
| base_max | 99.987% / 99.62% | 98.173% / 64.16% | 100% / 100% | 97.461% / 53% |
| medoid | 99.988% / 99.62% | 98.167% / 64.08% | 99.967% / 99% | 97.428% / 52% |
| base_softmax | 99.990% / 99.68% | 98.176% / 64.22% | 100% / 100% | 97.461% / 53% |
| centroid | 99.989% / 99.66% | 98.177% / 64.26% | 99.967% / 99% | 97.428% / 52% |

四种单坐标方案均通过能力门。按预注册只看train reachability并先最大化Atom recall，选择
`base_softmax`；它在154,911个train representable GT中仅16个未覆盖，4,984/5,000张整图
reachability exact，route100零损失。

最关键的发现不是四种坐标之间不到0.1pp的差异，而是零训练component base top-M：train
几何exact达到`3,211/5,000 = 64.22%`，route达到`53/100`。相比P1-G final的24.18%/22%，
单纯折叠重复candidate就提高40.04pp/31pp；train Atom recall也从95.35%升到98.18%。

## 决策

```text
selected_scheme = base_softmax
decision = component_representative_go
```

P1-D到P1-H长期停滞的主因现在有了直接证据：ranker一直在128个重复candidate之间竞争，而
不是在约67个Atom-like对象之间选择。下一步建立3px component-level set ranker；step 0必须
严格复现64.22% train / 53% route基线，训练只改善component presence排序，不再回到原始
candidate identity。

## 产物

```text
outputs/audits/chemvector-g3-p1j-component-representative/audit.json
SHA256 096e658b5e3193621b21c6b59f77082580af544a1786345d72feead75fb77af2

审计源码
SHA256 f7cc9b2700f4681d71a6c97b92d4cc80a049974005ac7b21244292f4d0f3c7c7
```
