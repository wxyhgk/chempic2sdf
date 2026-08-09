# G3 P11-E0 Predicted Source Pilot

> 日期：2026-08-09
>
> 状态：已预注册，尚未运行

P10证明合成source路线不能接入整图：真实top128大多已经正确，而P6--P9默认source偏离GT并必须
移动。P11改用冻结P4-E3真实预测source，并将原位置零偏移作为显式KEEP候选。

第一阶段只审计已有P4-E9 fit中的264个真实ADD_REFINE动作，不训练。重放236张动作图片，为每个
MOVE source生成P6-E1 top8候选；同图中已经6px匹配GT的source按哈希最多选8个KEEP。缓存保存
真实source crop、center field、P6候选、GT offset以及role/style/degree元数据。

数据门固定为：264个MOVE全部重放、至少180个MOVE的top8可达GT 6px、至少1,500个KEEP、至少
720个KEEP harmful pair且MOVE覆盖四种variant。失败则P6候选本身不适合真实source；通过才训练
显式`KEEP(offset=0) vs MOVE(top8)`模型。历史P7 holdout、confirm和外部集均不读取。

第一次子集重放按`candidate_index/base_source_id`校验失败。P4-E9原始ID来自完整20k、batch=8的
浮点排序，过滤到236图会改变batch组成和极邻近峰顺序。MOVE协议因此直接使用动作文件已冻结保存
的`base_anchor/target_anchor/base_source_id`，不从新排序反推；KEEP仍由当前冻结top128与GT匹配
得到。该修复发生在任何cache和容量结果产生前。
