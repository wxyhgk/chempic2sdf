# G3 P12 Explicit Score KEEP/MOVE

> 日期：2026-08-09
>
> 状态：已完成，NO-GO

P11-E3将真实MOVE扩大5.28倍后仍无安全动作，否定单纯数据不足。calibration审计显示center-field
精确点差`alternative-source`对rescue/harmful的pair AUC为0.8113；当前CNN的多层卷积与全局池化
没有可靠保留该标量。

P12保持P11-E3的1,464:1,464+2,048数据、patch-only编码器、损失、8,000步和随机隔离，只在每个
candidate特征中显式追加candidate center-field值及其相对source差值。source、alternative和差值
均由已有score crop双线性采样，不重扫数据。

calibration至少40 rescue/0 harmful才GO；否则停止。未过门不读取P11-E2 holdout、confirm或外部集。
正式checkpoint无论结果均提交并创建同名tag。

## 结果

- 模型29,293参数，显式score evidence已启用；
- checkpoint SHA256：`044d79f4740a689805943622af536a4dadaaf111e2cd322b6deb354f3842ad90`；
- 与P11-E3相同数据训练8,000步；
- calibration零harmful阈值仍只能0执行，0 rescue/0 harmful，正式NO-GO；
- P11-E2 holdout保持000且未读。

显式点值未把相关性转化为安全判别，否定“CNN只丢了center-field标量”这一充分解释。局部source与
candidate pair本身不足以判断当前source是否已经被整组Atom集合正确占用。停止增加单pair特征、
局部patch尺度或相同训练step；下一机制必须读取真实top128的集合邻域、source score/rank和候选
竞争关系，在集合级预测KEEP/MOVE。
