# G3 P15-E0 Bond Endpoint Pair Signal

> 日期：2026-08-09
>
> 状态：已预注册，尚未运行

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
