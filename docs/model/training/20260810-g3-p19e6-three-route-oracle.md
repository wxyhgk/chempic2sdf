# G3 P19-E6 Three-route Oracle

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
>
> 类型：零训练 Oracle 上限，不创建模型 tag

## 目标

P19-E5 的全局融合只差 crossed 一项 Pareto 门。训练 router 前，先用已有 eval 行级结果计算
旧模型、新模型、融合三路线的逐图 exact union，判断路由是否存在足够上限。

## 固定协议

- 输入固定为 P19-E4 新模型 eval rows、P19-E5 融合 rows 与旧模型 paired rows；
- 只按 sample key 对齐，不重新推理、不调参数；
- Oracle exact 表示三路线中至少一条整图 skeleton exact；
- 该结果读取 eval 标签，只是理论上限，不得称为部署结果；
- 不读取 fresh640、confirm、sealed test、USPTO 或 MR-TADF。

## GO/NO-GO

- overall Oracle 至少 553/640，并比融合至少多 10 张；
- 普通三布局至少 456、stress 至少 82；
- crossed 至少 7、near-nonbonded 至少 9；
- Oracle 只能选择现有三条路线。

通过才允许进入输出级 router capacity；失败则直接停止路由路线。
