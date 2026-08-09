# G3 P7 Pairwise Structural Counterfactual

> 日期：2026-08-09
>
> 状态：已完成，calibration GO / holdout NO-GO

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

## Calibration结果与holdout预注册

- 256-pair容量：254/256，99.21875%，GO；
- `patch-only`：28,717参数，OOF准确率76.78%至81.53%；
- `patch-only` calibration：阈值1.6568973064422607，147执行、110 rescue、0 harmful；
- `patch-band`：46,345参数，OOF准确率80.39%至82.61%；
- `patch-band` calibration：58执行、43 rescue、0 harmful；
- winner固定为`patch-only`，checkpoint SHA256为
  `a5fb6356a442957f67177288512c1af3f5560b6e00a5d80625fa46ca65956687`。

holdout固定为738 hard与2,776 easy。脚本只允许读取一次且只保存聚合结果，固定使用上述winner与
阈值；`harmful=0`且`rescues>=100`才GO。读取过程用`finally`恢复两个cache为`000`。

第一次执行已完成模型评分，但因脚本没有预先创建审计目录，在写JSON前发生`FileNotFoundError`；
结果未落盘、未打印，缓存已由`finally`恢复为`000`。机械重放只增加输出目录创建，不修改模型、
阈值、评分或门控。最终审计必须记录物理读取2次、结果只观察1次，不能声称物理单读。

## Holdout结果与最终决定

固定winner和阈值在738 hard、2,776 easy上执行193次：

- rescue：151；
- harmful：1；
- useless：41；
- 两个holdout cache运行后均恢复为`000`。

`rescues>=100`通过，但严格`harmful=0`失败，因此正式决定为
`pairwise_counterfactual_holdout_no_go`。本周期到此停止，不读取dev、confirm_v2、MR-TADF、
USPTO或封存集，也不接整图Atom/Skeleton/SDF和前端。不能在看到这个结果后调高阈值来删除唯一
harmful；下一周期若继续，必须使用新的train/calibration安全协议并保持本holdout只作历史证据。

这次NO-GO不否定pairwise视觉机制：calibration的110 rescue在holdout增长到151，且harmful从P6
原始residual的数十个降至1个。它否定的是当前checkpoint与固定阈值满足“零有害替换”的部署结论。

## 入口

```bash
.venv-train/bin/python -m \
  chemvector.evaluation.audits.atoms.build_g3_p7e0_pairwise_counterfactual_data

.venv-train/bin/python -m \
  chemvector.training.atoms.train_g3_p7_pairwise_counterfactual
```
