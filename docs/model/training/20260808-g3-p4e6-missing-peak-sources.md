# G3 P4-E6 Missing Peak Sources

> 日期：2026-08-08
>
> 状态：预注册，待运行
>
> 审计：`chemvector-g3-p4e6-missing-peak-sources`

## 目标

P4-E5在dev top1024仍缺58个Atom，768到1024完全无增长。本轮固定同一checkpoint和dev640，
逐个定位58个最大基数匹配下的未匹配GT，确定下一次修改应该针对局部峰生成、深尾排序还是
一对一匹配冲突。

## 固定协议

- 使用与生产指标完全相同的最大基数优先、距离次优Hungarian匹配；
- top1024必须严格重放32,342 matched、58 missing、43张失败图；
- 每张图同时提取全部完整plateau regional maxima，上限200,000，实际不得触顶；
- 每个漏点记录top1024与全部峰的最近距离/排名、GT cell logit与regional-max状态；
- 记录最近GT距离、是否相邻GT cell、3×3局部竞争者是否为另一GT；
- 独占分类顺序为matching conflict、below top1024、GT cell不是峰、峰anchor偏移超6px；
- 为所有失败图生成原图叠图和HTML索引，红圈为漏GT、橙圈为最近完整峰；
- 不训练、不修改checkpoint、不读pilot、confirm_v2、fresh或外部/封存集。

若某一类别占58个漏点至少一半，则报告为primary；否则报告mixed。该结论只决定下一实验方向，
不构成模型GO，不创建tag。

## 预期产物

- `outputs/audits/chemvector-g3-p4e6-missing-peak-sources/audit.json`
- `outputs/audits/chemvector-g3-p4e6-missing-peak-sources/failures.jsonl`
- `outputs/audits/chemvector-g3-p4e6-missing-peak-sources/index.html`
- `outputs/audits/chemvector-g3-p4e6-missing-peak-sources/overlays/`
