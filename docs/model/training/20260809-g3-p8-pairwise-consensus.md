# G3 P8 Pairwise Consensus

> 日期：2026-08-09
>
> 状态：已预注册，尚未训练

P7 patch-only在calibration达到110 rescue/0 harmful，但固定holdout为151 rescue/1 harmful。之后
只使用原calibration做五折交叉证书：四折选零harmful阈值、一折验证，其中1/5折再次出现1个
harmful。这说明单模型阈值的安全边界不稳定，不是继续追加同一模型step可以解决的问题。

P8不读取已暴露的P7 holdout。复用P7 winner作为成员A，用不同模型初始化与采样顺序训练两个新的
patch-only成员B/C，各8,000步。一个source只有在三个成员各自选择同一个alternative时才可执行，
证书分数固定为该alternative在三个成员中的最低corrected margin。

## 固定门控

- train pair和损失保持P7不变，不加入holdout派生特征；
- calibration按scaffold哈希做五折交叉证书；
- 每折仅用其余四折选择零harmful最大rescue阈值；
- 五个certificate折合计必须0 harmful且至少60 rescue；
- 全calibration固定阈值必须0 harmful且至少80 rescue；
- 任一门失败即NO-GO，不构造新数据、不读取历史holdout/dev/confirm/外部集；
- 通过后才允许从未进入P6/P7的scaffold构造新的P8安全确认集。

两个新成员无论GO/NO-GO都保存checkpoint、提交结果并创建与模型名相同的annotated tag。

第一次正式执行已完成两个8,000步成员并保存checkpoint，但calibration中存在只有prior、没有任何
alternative的source；空索引默认成为float tensor，评估在产生任何证书结果前报错。修复固定将
该边界样本视为“不执行”，并按checkpoint/audit SHA复用已训练成员，禁止重新训练或改变seed。
