# G3 P7 Pairwise Structural Counterfactual

> 日期：2026-08-09
>
> 状态：已预注册，P7-E0 数据门通过，尚未训练

P6-E9 的三个视觉 residual 臂可以恢复约一半 hard target，但都会伤害 easy；P6-E10 三臂共识和
P6-E11 26维安全 MLP 虽能达到零 harmful，却分别只有46和69次 rescue。下一阶段不再增加汇总
特征或搜索规则，而是直接比较同一 source 的 prior 与 alternative 局部视觉结构。

## P7-E0 数据

P6-E8 已有缓存直接提供96x96原图、score field、候选坐标、GT及scaffold。本阶段只生成索引，
不复制约1GB raster，也不读取holdout。

- train：7,376 rescue、136,562 harmful、6,142 equivalent；
- calibration：921 rescue、17,416 harmful、804 equivalent；
- train/calibration scaffold重叠为0；
- 两个holdout cache继续保持`000`；
- 两者都错误的pair只审计，不提供方向监督。

## 固定模型与损失

模型接收source、prior、alternative三点及原图/score crop。共享候选编码器输出两个有条件效用，
最终只返回：

```text
utility_delta = U(alternative | prior) - U(prior | alternative)
```

因此交换prior/alternative时输出由结构保证严格变号。peak logit不进入learned encoder，只在最终
计算`corrected_margin = normalized_prior_margin + utility_delta`时使用。

只允许两个正式臂：

1. `patch-only`：候选中心局部视觉patch；
2. `patch-band`：patch加source到candidate的定向二维ink band。

损失固定为pairwise logistic ranking、hard-negative top-k、equivalent零delta和residual L2。模型
参数必须小于300k。

## 固定训练与门控

1. P7-E1先在256个唯一source directed pair上训练2,000步，训练准确率至少99%；失败即停止。
2. 容量通过后，每个臂按scaffold做5-fold OOF，每fold 3,000步，再全train 8,000步。
3. train使用全部cap可达rescue、等量确定性harmful和2,048 equivalent；batch固定28:28:8。
4. calibration每个source最多执行一个最高corrected-margin alternative。
5. 合法winner必须`harmful=0`且`rescues>=100`；再按rescues最大、参数最少选择。
6. 两臂都失败则P7正式NO-GO，不读取holdout，也不增加第三臂或新规则。
7. 无论GO/NO-GO，两个正式模型均保存checkpoint、提交结果并创建同名annotated tag。

只有calibration GO才允许实现并单次运行holdout阶段；MR-TADF、USPTO、confirm_v2和整图SDF当前
全部禁止读取。

## 入口

```bash
.venv-train/bin/python -m \
  chemvector.evaluation.audits.atoms.build_g3_p7e0_pairwise_counterfactual_data

.venv-train/bin/python -m \
  chemvector.training.atoms.train_g3_p7_pairwise_counterfactual
```
