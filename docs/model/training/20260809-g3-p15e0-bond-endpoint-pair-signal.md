# G3 P15-E0 Bond Endpoint Pair Signal

> 日期：2026-08-09
>
> 状态：已完成，NO-GO

P13和P14分别否定局部峰集合、完整top128几何/score集合足以安全决定KEEP/MOVE。P15只考察真正
新增的视觉信息：冻结BondObject在source与P6 top8 MOVE位置附近输出的端点支持。

本轮是零训练信号审计，不产生模型checkpoint。只重扫P11-E0的236张pilot困难图，为2,152个source
缓存9个位置（KEEP source + 8 MOVE）的18维连续端点特征。特征包括最近端点距离/score、4/8/12/20
像素尺度的连续支持及8类键型支持；不使用GT bond、Render JSON或semantic layer。

## 固定评测

- pair仍使用P11-E1严格分层的281 rescue + 281 harmful；
- 只用train scaffold做固定三折OOF，完整scaffold不能跨fold；
- baseline为MOVE几何、P6 peak logit及source/alternative center-field点值；
- augmented只追加source、alternative及差值三组Bond端点特征；
- calibration PNG只允许提前缓存无标签特征，本轮不读取calibration标签或结果；
- 不读取任何holdout、confirm、MR-TADF或USPTO。

三项门必须同时通过：augmented OOF AUC不低于0.70、相对baseline提升不低于0.03、rescue MOVE
位置6px内预测端点覆盖率不低于50%。通过才允许训练P15；失败则停止当前冻结BondObject的局部
端点路线，不用调sigma、阈值或增加相同模型数据。

## 结果

- 2,152行、236图、source与top8共9个位置的端点特征全部有限且mask对齐；
- endpoint cache SHA256：`7c598206c1d8fe7ab132c3e43a1a77fa5fde86777d37628107d563eedb9e079a`；
- 特征构建耗时22.29秒；
- train-only三折OOF baseline AUC：0.847431，macro AUC：0.853631；
- augmented Bond AUC：0.845506，macro AUC：0.852366；
- AUC lift：-0.001925，未达到+0.03门；
- rescue MOVE位置6px端点覆盖率：69.04%，覆盖门通过；
- calibration标签、holdout、confirm和外部集均未读取。

P15-E0正式NO-GO，不训练Bond增强模型。冻结Bond端点在该任务上覆盖了多数rescue位置，但其信息
已被MOVE几何、P6 peak与center-field点值解释，追加后没有跨scaffold增益。

同时出现了一个必须单独验证的新现象：仅7维显式baseline的OOF AUC达到0.847，而P14的68,877
参数模型在calibration高分区仍由harmful主导。下一周期应冻结同一7维输入，训练确定性L2线性
决策器，直接验证低容量模型能否获得安全calibration；不得把它写成P15 Bond路线继续训练。
