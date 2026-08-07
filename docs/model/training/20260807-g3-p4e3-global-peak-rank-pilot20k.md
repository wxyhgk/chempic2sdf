# G3 P4-E3 Global Peak Rank Pilot20k

> 日期：2026-08-07
>
> 状态：已完成，协议通过 / 能力GO
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

## 实际结果

固定5,000步、20,000 exposures完整运行，训练耗时550.88秒；包含两次final dev重放和写盘的
总耗时623.98秒。全部协议检查通过，两次dev指标和candidate source digest逐项相同。

| dev指标 | step0 | P4-E2 final | P4-E3 final | P4-E3相对step0 |
|---|---:|---:|---:|---:|
| top128 matched | 32,098 | 32,102 | **32,195** | **+97** |
| top128 recall | 99.0679% | 99.0802% | **99.3673%** | **+0.2994pp** |
| top128 exact | 517/640 | 518/640 | **550/640** | **+33** |
| known-Count matched | 31,092 | 31,163 | **31,332** | **+240** |
| known-Count recall | 95.9630% | 96.1821% | **96.7037%** | **+0.7407pp** |
| known-Count exact | 270/640 | 277/640 | **296/640** | **+26** |

相对只增加同目标训练量的P4-E2，P4-E3多恢复93个top128 Atom和32张整图exact。三项能力门
全部通过，说明直接监督整图候选预算边界，而不是继续增加局部3×3损失的数据量，确实命中了
P3-E31/P4-E2定位的candidate ranking断点。

known-Count按variant的变化全部为正：

| variant | matched变化 | exact变化 |
|---|---:|---:|
| coordgen | +51 | +3 |
| original | +63 | +6 |
| rdkit | +56 | +9 |
| stress | **+70** | **+8** |

stress仍是最弱桶，但恢复量最大，符合P3-E31中tail主要来自stress的诊断。

## 风险与边界

regional maxima平均数从30,279增至56,763/图，p90从32,524增至85,310，最大值为102,894。
top128排序显著改善，但连续场变得更碎；这会增加完整plateau提取的CPU成本，也可能让跨checkpoint
候选并集带入更多低价值峰。下一步必须先做冻结cache的可达性与峰数/耗时审计，不能仅凭本轮GO
直接扩大训练或进入SDF。

本轮没有读取confirm_v2、fresh、USPTO、MR-TADF或sealed_test。能力GO只说明新目标可以进入
下一候选周期，不代表端到端模型已部署。

## 产物与决策

- 最终决策：`global_peak_rank_pilot20k_go`；
- checkpoint SHA256：
  `1c358d0c936c687fa0e40ea5e1d9364dd98a663bb762d294db76b43d5f009593`；
- audit SHA256：
  `da98b8bb4a576359936a7a73b0fc65ffbcbb67080501d074aaa4e8d59db61c48`；
- final metrics SHA256：
  `bfcdbf2dbb7fd931f857cbebfad5563f916cb40418a9653faffac13e3c720cc4`。

结果提交后创建同名annotated tag：
`chemvector-g3-p4e3-global-peak-rank-pilot20k`。
