# G3 P4-E3 Global Peak Rank Pilot20k

> 日期：2026-08-07
>
> 状态：预注册，待训练
>
> 模型：`chemvector-g3-p4e3-global-peak-rank-pilot20k`

## 目标

P4-E2只比较GT中心与3×3邻域，20k训练后dev top128仅多4个Atom。P4-E3保持网络、初始化、
数据、optimizer和训练量不变，只增加已经通过零步审计的全图候选边界排序损失，验证直接优化
top128必要条件能否恢复被整图假峰挤出的GT。

## 固定训练协议

- 初始化仍为P4-E2之前的train25k checkpoint，SHA256：
  `9e9d7725e9e9ffecde7bcbe9fbbb095f09204a480b18acbb014d88ef5ac83668`；
- 不从P4-E2 final继续，以免把额外epoch与新目标混为一项；
- train为P4-E0 pilot20k，5,000个分子、20,000张图；dev为旧P3-E21 dev640；
- batch4、1 epoch、5,000 steps、20,000 exposures、fixed-final-only；
- AdamW，lr `1e-4`、weight decay `0.01`、grad clip `1`；
- 只训练stride-1 center field和最后一个stride-4 vision block；
- one-hot center、offset权重8、局部peak margin权重1/margin1保持不变；
- 新增全图peak-rank权重10、候选预算128、边界窗口16、margin1；
- 权重10的依据：pilot初始batch中旧目标梯度范数2.926，新目标为0.275；乘10后为2.751，
  两个目标量级接近；50个pilot batch的新损失均值0.0438、最大0.2243；
- 不使用固定像素聚类、键长删除或后处理规则。

confirm_v2、fresh、USPTO、MR-TADF和sealed_test均不得读取。

## GO/NO-GO门

- dev top128 Atom recall相对step0至少增加0.1个百分点；
- dev top128 exact至少增加5/640；
- known-Count recall回退不超过0.1个百分点；
- 固定步数、exposure、训练顺序、optimizer范围、冻结state和重复dev全部通过。

门与P4-E2相同，因此通过时也必然超过P4-E2 final。无论GO或NO-GO，都保存final checkpoint，
提交结果，并创建与模型名完全一致的annotated tag。NO-GO时禁止追加相同协议epoch。

## 预期产物

- `outputs/checkpoints/chemvector-g3-p4e3-global-peak-rank-pilot20k/step-005000.pt`
- `outputs/checkpoints/chemvector-g3-p4e3-global-peak-rank-pilot20k/final_metrics.json`
- `outputs/checkpoints/chemvector-g3-p4e3-global-peak-rank-pilot20k/audit.json`
