# G3 P1-M 一跳替换边界可达性审计

> 日期：2026-08-03
>
> 状态：已完成，选择16×16动作边界
>
> 性质：P1-K final零训练动作候选审计

## 唯一问题

P1-L确认783张train失败图只需一次component替换。P1-M按P1-K final score排序，对每张一跳
可达图只允许从最低K个selected中drop、最高K个unselected中add，扫描`K=1/2/4/8/16/all`。
每个swap执行后重新做完整GT一一匹配，只有整图exact才算覆盖。

## 固定判读

- 固定P1-K final、3px component、base-softmax anchor和known-count；
- 只读train5k与route100；不训练、不读取外部/封存数据；
- 主判读：若top-8覆盖≥90%的train一跳图且route≥80%，下一步只生成8×8动作；否则若
  top-16覆盖train≥95%、route≥90%，使用16×16；否则必须全pair或换结构化动作头。

P1-M不保存checkpoint、不生成SDF、不接前端。

## 正式结果

| 边界K | Train覆盖 | Route覆盖 |
|---:|---:|---:|
| 1 | 331/783 = 42.27% | 8/22 = 36.36% |
| 2 | 524/783 = 66.92% | 12/22 = 54.55% |
| 4 | 672/783 = 85.82% | 15/22 = 68.18% |
| 8 | 746/783 = 95.27% | 17/22 = 77.27% |
| 16 | 775/783 = 98.98% | 20/22 = 90.91% |
| all | 783/783 = 100% | 22/22 = 100% |

top-8在train充足，但route少1张才达到80%门，不能用train结果覆盖迁移门。top-16同时通过
train≥95%和route≥90%，仅8张train、2张route的一跳图需要更深动作。

## 决策

```text
decision = top16x16_action_candidates
```

下一步每图只生成最低16个selected×最高16个unselected的最多256个swap。训练标签必须是执行
swap后的整图exact；当前exact图上的所有动作均监督为禁止执行，部署门要求零有害替换。不能将
局部补回一个GT当正例，也不能因top-8只差一张route图而修改预注册门。

## 产物

```text
outputs/audits/chemvector-g3-p1m-one-move-boundary-reachability/audit.json
SHA256 427e5e6c760a6e65e9528e9d5659e097816d1eaec8d39827ba76af0df693b18b

审计源码
SHA256 62b280c458ea711b075a816f5e5a5afb09e0e7753671e825054cae74e13bbed9
```
