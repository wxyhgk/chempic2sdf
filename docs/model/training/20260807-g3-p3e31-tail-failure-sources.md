# G3 P3-E31 Tail Failure Sources

> 日期：2026-08-07
>
> 状态：已完成，断点定位为stress风格下的candidate generator排名
>
> 审计：`chemvector-g3-p3e31-tail-failure-sources`

## 目标

P3-E29局部键端点和P3-E30全局双端点关系都无法救回27张tail-required图片。本轮不训练，直接
审计其中33个top128遗漏、top192可达的GT Atom，区分断点究竟来自文字、junction、某个variant，
还是一般candidate排序。

每个case记录Atom角色、degree、元素、variant、字体、candidate rank、最近top128距离和相对拥挤
度。GT bond只用于离线degree和中位键长计算，不产生模型输入。另为27张图生成叠图：红圈为tail
GT Atom，绿圈为兼容的top129-192 candidate，橙圈为最近top128 candidate。

## 实际分布

共33个case、27张图、32个不同的molecule-atom：

| 维度 | 分布 |
|---|---|
| variant | stress 23、original 7、rdkit 2、coordgen 1 |
| stress operation | crossed_bonds 13、crowded_subgraph 5、near_nonbonded 5、非stress 10 |
| role | implicit chain 12、visible text 8、junction 7、endpoint 6 |
| element | C 25、O 7、P 1 |
| degree | degree1 10、degree2 15、degree3 6、degree4 2 |
| candidate rank | 129-144为16、145-160为6、161-176为7、177-192为4 |
| crowd ratio | 小于0.5为14、0.5-0.75为9、0.75-1为9、大于等于1为1 |

stress占69.70%，达到预注册诊断中的variant majority；可见文字只占24.24%，junction只占21.21%，
都不是多数。tail candidate距GT平均2.71像素，最近top128 candidate却平均相距14.61像素，说明
空间候选并未消失，而是正确峰在压缩、交叉或近邻风格下被排到了平均151名。

自动方向判定为`variant_conditioned_candidate_generator`，更精确地说应先验证“更多唯一stress训练
图是否改善center-field候选排名”。仓库已有train25k stride-1 center field；它在旧route100上从
train5k的3073/3080提高到3074/3080，但因旧门差1个目标正式NO-GO。下一步应先冻结A/B该checkpoint
在当前Micro160的top128/top192与tail rank，不能直接重训，也不能修改P3-E29/E30的selector门。

## 产物

- audit SHA256：`bf810cd51bfc75dc93266845c3bdbb026c559319856f93900331a4b76df9f905`；
- cases SHA256：`96f0a9258935fb0e43dd2db88d93426a9d9d6df4c04fb1fe6ff9c17da9bd6ca1`；
- 27张叠图SHA256：`fe69cf457b201dfa8287638d9095a8181d2ff43f23f453d8dd211a9a6f1d4a72`；
- 目录：`outputs/audits/chemvector-g3-p3e31-tail-failure-sources/overlays`；
- dev、confirm、fresh、USPTO、MR-TADF和sealed_test均未读取。

本轮零训练，不创建tag。
