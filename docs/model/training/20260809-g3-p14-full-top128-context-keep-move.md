# G3 P14 Full Top128 Context KEEP/MOVE

> 日期：2026-08-09
>
> 状态：已预注册，尚未训练

P13加入局部top-16区域峰集合后仍只有零执行安全阈值。P14不再增加单点标量或patch，而是检验
P13尚未覆盖的机制：KEEP/MOVE是否必须知道当前source在完整P4-E3 top128候选集合中的全局竞争
关系。

## P14-E0 数据门

冻结P4-E3，只重扫P11-E0涉及的236张困难图，为既有2,152个真实source行补充完整集合上下文。
每行固定129个token：重放的top128，加一个按dense `base_source_id`读取的显式source。该设计保留
了29个在重放时跌出top128边界的旧source，不按最近坐标替换。

- 2,152行全部只写一次；
- source坐标与`GT - offset_target`逐行对齐，最大误差0像素；
- 每行恰好一个显式source token；
- cache SHA256：`282ea57292f71c450f017981502c2b31ff7f478372dc4a47b52d01696dde9e40`；
- 构建耗时20.71秒，未读取任何holdout、confirm或外部集。

## 固定模型与训练

- 模型名：`chemvector-g3-p14-full-top128-context-keep-move-fit562`；
- 保留P13的patch、显式center-field点值、局部top-16峰集合；
- 新增129-token共享MLP与masked mean/max DeepSets；token包含相对source/candidate坐标、score、
  score rank、source身份和candidate距离；
- token顺序任意置换时输出不变；交换KEEP/MOVE时输出严格反对称；
- 参数量低于300k；
- 数据、分层、损失、batch、optimizer与P11-E1相同。

先用32 rescue+32 harmful训练1,500步，准确率至少99%才进入正式训练。正式训练固定281 rescue、
281 harmful及equivalent池，6,000步。calibration至少20 rescue且0 harmful才GO；否则不扫描full80k
集合上下文，不读取历史holdout、confirm、MR-TADF或USPTO。正式checkpoint无论结果均记录SHA256
并创建与模型名相同的annotated tag。

若容量门通过但calibration失败，说明完整top128几何/score集合仍不足以判断安全MOVE；下一路线
必须加入联合Atom选择或Bond incidence，而不是继续扩大同一KEEP/MOVE头。
