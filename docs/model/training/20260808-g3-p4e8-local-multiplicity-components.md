# G3 P4-E8 Local Multiplicity Components

> 日期：2026-08-08
>
> 状态：预注册，待运行
>
> 审计：`chemvector-g3-p4e8-local-multiplicity-components`

## 目标

P4-E6有28个matching-conflict漏点，P4-E7证明固定cell center不能修复。本轮只重放43张已知
失败图，在6px和10px分别构建GT/top1024 candidate二部连通分量，并对每个分量独立计算最大
匹配缺口，确定局部multiplicity decoder需要的容量。

## 固定协议

- 43张身份由P4-E6 failures SHA固定，不读取其余dev图片；
- top1024、预测offset、最大基数Hungarian不变；
- eligibility半径固定6px和诊断用10px；
- 报告每个缺口分量的GT数、candidate数、最大匹配数和deficiency；
- 6px总deficiency必须重放58，10px必须重放23；
- shared-deficient分量若全部GT不超过4、deficiency不超过2，并覆盖至少90% shared deficiency，
  则`bounded_local_multiplicity_go`；否则为unbounded/mixed；
- no-candidate分量不计入multiplicity容量门，它属于峰生成/坐标细化问题；
- 不训练、不读pilot/confirm_v2或外部/封存集。

本轮零训练，不创建tag。
