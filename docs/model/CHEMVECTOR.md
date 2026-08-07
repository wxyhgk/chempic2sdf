# ChemVector

ChemVector 是面向化学结构图片识别的轻量视觉到结构模型。v0.8 主线直接预测
BondObject 和 TextObject，汇聚为 ChemScene，再由确定性编译器写出 SDF。

本页只维护当前状态、架构说明和训练档案索引。每次训练的完整命令、指标、失败
配置和结论统一放在 [`training/`](training/README.md) 中，不再追加到本页。

2026-08-07 的 G3 P3-E22 零训练审计已将 Atom candidate 固定预算从128扩到192：
train/dev/confirm recall为99.2409%/99.3549%/99.4125%，三组均通过预注册99%门。
下一周期先验证旧base/expert和P3-E16对动态候选长度的兼容性，再恢复五路线完整SDF比较；
top256不采用，`sealed_test`仍保持封存。详见
[P3-E22 训练档案](training/20260807-g3-p3e22-candidate-budget-audit.md)。

随后的 [P3-E23 恢复审计](training/20260807-g3-p3e23-top192-cycle-recovery.md) 证明旧
base/expert权重可严格加载并输出128×192 pointer，但P3-E16 Count在新evidence上dev exact
从28.59%降到2.50%、MAE从3.51升到17.70，正式NO-GO。五路线尚未训练；当前先审计
pointer/known-count是否稳定，再决定top192 Count适配范围。

[P3-E24 输入协议 A/B](training/20260807-g3-p3e24-top192-input-protocol-ab.md) 进一步证明
global/prefix top192虽使dev router matched净增203/248，但在30张新增可达图中均0 rescue，并
破坏19/18张旧正确图；所以旧query没有真正利用新增容量。当前不做Count-only修补，转为训练
直接读取192候选的动态一一匹配occupancy/query。

[P3-E25 Hard Latent Occupancy](training/20260807-g3-p3e25-hard-latent-occupancy-micro160.md)
在Micro160训练后known-count micro F1达95.67%，但Count/known/predicted exact仅76/47/36，
新增可达0/27，正式NO-GO。hard单一identity会对兼容替代峰产生矛盾正负监督；下一路改为
identity-invariant soft coverage与兼容分量质量约束。

[P3-E26 Soft Coverage](training/20260807-g3-p3e26-soft-coverage-occupancy-micro160.md)只把
known-count exact从47提高到49，Count从76降到69，新增可达仍0/27，正式NO-GO。两个冻结
evidence candidate occupancy头都未利用tail；下一路线解冻128-query×192-candidate pointer本体。

[P3-E27 Query-slot](training/20260807-g3-p3e27-query-slot-occupancy-micro160.md)将Count做到
160/160，但Atom exact仅11/160、existence boundary 0/160、新增可达0/27，证明断点是稳定
query/pointer集合而非Count容量。下一步先审计是否存在可合法冻结的incident Bond视觉证据。

[P3-E28 五路线闭环](training/20260807-g3-p3e28-five-route-cycle-closure.md)确认前三路0通过，
hybrid受known-count最高49/160上界限制；bond即使通过也只能使maximum possible passes=1，低于
至少2路Micro门。原周期正式NO-GO，未进入dev、双profile SDF、confirm、外部或sealed阶段。

[P3-E29 Bond Incident Signal](training/20260807-g3-p3e29-bond-incident-signal-micro160.md)进一步
确认冻结键端点局部证据虽有0.9608 candidate AUC，却无法救回任何tail-required图片（0/27）；
停止独立候选局部融合，下一步转向同时解释一根键两个端点的全局pair consistency审计。

[P3-E30 Bond Pair Consistency](training/20260807-g3-p3e30-bond-pair-consistency-micro160.md)确认全局
双端点关系也只能把整体recall从95.92%增至96.18%，tail exact仍为0/27；停止当前冻结BondObject
驱动的selector路线，转而审计33个tail-only Atom的角色与视觉来源。

[P3-E31 Tail Failure Sources](training/20260807-g3-p3e31-tail-failure-sources.md)定位到23/33漏点来自
stress，角色分布混合、文字仅8/33；正确candidate仍存在但平均落到第151名。下一步冻结A/B已有
train25k stride-1 center field，验证更多唯一stress训练图是否改善当前Micro tail排名。

[P3-E32 Train25k Candidate Source A/B](training/20260807-g3-p3e32-train25k-candidate-source-ab.md)
恢复13/27旧tail图，使top128 exact从133增至145、stress从21/40增至29/40；但其自身top192
只有155/160，直接替换正式NO-GO。两个中心场错误互补，下一步按共同source_id审计候选并集。

[P3-E33 Candidate Source Union](training/20260807-g3-p3e33-candidate-source-union.md)确认source-id并集
在top128保留145/160和13张tail恢复，top192仅154/160、正式NO-GO，但top256达到160/160。
下一周期固定256 winner，并从排除本周期train/dev/confirm的style数据训练可泛化selector/center field。

[P4-E0 Source Union Cycle Manifests](training/20260807-g3-p4e0-source-union-cycle-manifests.md)已
通过：pilot20k、full80k、confirm_v2 700图与旧周期及彼此的molecule/scaffold边界全部满足，物理
tar顺序与四variant完整；下一步构建pilot20k的256双源cache。

[P4-E1 Pilot Union Cache](training/20260807-g3-p4e1-pilot-union-cache.md)完成256×72 cache，但
pilot/dev candidate recall仅99.883%/99.423%，虽Micro仍160/160，Candidate Oracle正式NO-GO。
暂停selector训练，先从train25k checkpoint继续训练center field并重建union。

[P4-E2 Center Field Pilot20k](training/20260807-g3-p4e2-center-field-pilot20k.md)完成5,000步：
known-Count多匹配71个Atom，但dev top128只多4个Atom、exact只+1，能力NO-GO；协议全部通过。
不追加同损失epoch，下一路线必须直接监督整图regional-maxima排序。

[P4-E3 全图峰排序损失审计](training/20260807-g3-p4e3-global-peak-rank-loss-audit.md)已经通过：
新目标要求GT中心超过top128边界附近的整图假regional maxima，峰位置detach而选中logit保留
梯度，相邻GT不作为负样本。batch4、768² BF16前后向仅1.42ms、约99MiB，允许进入受控pilot20k。

[P4-E3 Global Peak Rank Pilot20k](training/20260807-g3-p4e3-global-peak-rank-pilot20k.md)已预注册：
回到P4-E2之前的同一train25k初始化，在相同pilot20k加入权重10的全图排序项；固定5,000步和
相同能力门，构成只改变目标函数的受控A/B。confirm_v2和外部/封存集继续未读。

## 当前版本

- 当前完成的研究 tag：`v0.10.21`。
- 当前部署默认：`v0.10.10`；当前公开能力基线：`v0.9.41`。
- 当前开发包版本：`v0.10.21`；当前研究实验编号：`v0.10.21`。
- v0.10.15 structural-twin 已通过结构门；完整 SDF 回归前默认仍为 `v0.10.10`。
- v0.10.16 已将 structural-twin 接入 OCR/SDF 统一推理。固定 40 张开发集 A/B 的 SDF valid
  `18→24`、非立体 exact `9→12`、chemical exact `5→6`，零回退；工程 GO，仍为非推荐
  shadow profile。
- v0.10.17 回到 v0.9.41 语义 BondObject 主干并接入 v0.10.9 开放 CTC。USPTO clean 1k
  Skeleton 保持 72.7%，SDF valid `70.4%→78.5%`、chemical exact `16.0%→21.5%`；存在
  17/20 个 valid/chemical paired break，因此只作为能力 shadow，不自动切换默认。
- v0.10.18 以 v0.8.8 为主 OCR、v0.10.9 为专家，内部 4k teacher 训练保守 utility router。
  固定 `0.95` 在 USPTO clean 1k 只 route 12 次，Atom-typed `293→301`、chemical exact
  `160→162`，Skeleton/valid/exact 全部 paired break 为 0；作为保守 shadow。
- v0.10.19 为双 OCR router 加入 144 个成对视觉特征。validation break-0 rescue 达 394，
  但独立 fresh 2k confirm 为 185 rescue/3 break，三次均是 `O/OH`；NO-GO，不接前端。
- v0.10.20 再加入两模型对两候选的 CTC 交叉分数；独立 fresh 2k 为 348 rescue/5 break，
  且最高错误概率 0.994。单对象概率门正式停止，转图级双候选反事实验证。
- v0.10.21 invalid-only 双分支已接统一推理。USPTO 1k valid `704→802`、chemical exact
  `160→187`，全部 0 break；作为非推荐 shadow profile，默认模型不变。
- 完整 MR-TADF v1 已改为主应用 benchmark。v0.10.21 的 valid 65.91%、Skeleton exact
  48.09%、Chemical exact 24.00%；count-pair 正确条件下 Skeleton exact 97.96%，下一主线
  是整图 AtomObject/BondObject 计数与去重。USPTO 只保留为公开 OOD 护栏。
- G3 P1-F确认top-128固定candidate身份是系统性错误监督：train5k中89.13%的representable
  Atom有多个6px等价候选。P1-G改为组内动态一一匹配后，train几何exact
  `18.68%→24.18%`、route `17→22/100`，但未过预注册+10pp因果门；当前不追加相同loss或
  epoch。P1-H否定后期assignment振荡；P1-I的3px target-free聚合将候选`128→67.20`、
  多component GT `89.13%→20.01%`，train覆盖exact 99.74%、route 100%。P1-J选择
  base-softmax单坐标，零训练component base top-M已将train exact `24.18%→64.22%`、route
  `22%→53%`，直接确认重复candidate竞争是当前Atom集合主瓶颈。P1-K再训练60,609参数
  component ranker后，train exact 78.06%、route 68%，因果GO；train recall 98.945%距99%
  能力门差0.055pp。P1-N随后抛弃固定像素聚类，将三槽压成连续cell field并以完整plateau
  regional maxima生成对象；零训练train known exact达到87.94%、route历史诊断79%，相对
  3px基线分别只break 5/0张。机制GO，下一步训练轻量单通道Atom center field，正式替换仍需
  新fresh确认。P1-O的771参数线性头和P1-P的8,643参数局部残差均未通过overfit32严格门；
  P1-P已将top128提高到1421/1424、30/32，并定位剩余失败为192网格对角相邻Atom。P1-Q将
  场升到384²但双线性冻结特征仍为30/32且known回退，证明下一步必须直接使用高分辨率视觉
  feature或解冻高分辨率block，不能继续扩大同一后处理adapter。P1-R用7,628参数
  PixelShuffle直接从冻结stroke feature学习四个固定子格phase，但因丢失P1-A强base，top128
  回退到1394/1424、21/32且产生约3.2万峰/图；容量NO-GO。下一步只允许保留P1-A base并学习
  零初始化四相位残差，以隔离验证子格相位价值。P1-S完成该隔离后top128仍只有
  1420/1424、29/32，且约3.18万峰/图；确认phase直接作为logit会破坏连续场局部拓扑。冻结
  小头路线停止，下一步必须生成连续384 feature再统一卷积出center，并联合解冻最后一个
  stride-4视觉block。P1-T完成该架构后，92,035参数在overfit32严格门达到known/top128均
  `1424/1424、32/32`，四种布局全部100%，首次证明连续高分辨率feature加最小视觉联合微调
  具备整组容量。P1-U随后从相同初始化交替训练原32图与近邻专项32图4,800步：原域仍100%，
  但专项域known为1461/1473、20/32，top128为1468/1473、27/32。审计确认384 one-center
  target仍在3张图各丢一个Atom，而768网格零碰撞；P1-U容量NO-GO，不进入5k，下一步只测试
  连续stride-1中心场。P1-V的adjacent32 top128达到1473/1473、32/32，7个碰撞对全部由不同
  峰命中，确认stride-1表示机制GO；但known仅1447/1473、18/32，且原域回退到22/32。当前
  瓶颈已转为约9.8万高召回峰的整组选择。P1-W冻结审计找到首个双域全覆盖协议：stride-2
  top96加stride-1 top32，共128候选，在overfit和adjacent均达到全Atom、32/32。P1-X固定
  该输入并以动态一对一集合匹配训练105,089参数selector，最终两个域均达到全Atom、32/32，
  独立确定性重放逐位一致，容量GO。P1-Y随后证明capacity64中心场在train5k的96+32候选仅
  3346/5000 exact；P1-Z/P2-A把中心场正式训练到5k后，route top128分别提升到96/100和
  98/100，证实训练数据不足是上游主因。P2-E扩大selector训练后train提高但route退化到61/100，
  动态硬匹配boundary loss已经NO-GO。P2-F在相同exposure下将5k重复改成25k唯一图，route只从
  3073/3080、98/100增至3074/3080、99/100，未过预注册门；下一步改进密集/交叉局部的center
  target，不直接上100k，也不追加同一selector epoch。P2-G读取真实文字层做可恢复性审计后，
  eligible route为3003/3003、98/98；唯一severe stress图在top128为47/53，扩大到top1024仍
  只有49/53。P2-F正式NO-GO不变，但中心候选主线停止扩量，转未知Count下的Atom整组选择。
  P2-H的291,394参数query集合在capacity64把Count做到64/64，但唯一pointer输出的easy/hard
  exact只有9/32、8/32；训练期允许query pointer重复、推理期才强制唯一是明确协议缺口。下一步
  将全局唯一pointer matching直接纳入训练，不追加同一loss或直接扩到5k。P2-I完成该A/B后
  easy/hard为12/32、5/32，仍NO-GO；但GT-linked query的同一pointer输出达到32/32、28/32，
  而部署existence top-N与GT-linked query只重合92.35%/93.31%。下一步不再改pointer，而是用
  兼容候选pointer质量替代不可靠的概率加权平均坐标，定义query-to-GT matching。P2-J完成该
  单变量A/B后大幅提高到28/32、26/32，pointer-linked query双域均32/32；step3200到4800仍
  连续提高，因此允许固定optimizer续训到8,000步。P2-K final只到30/32、27/32，中间最好也
  只有30/32、29/32且后程振荡；停止追加epoch。P2-L从同初始化加入动态极值existence边界后
  崩到0/32、0/32，明确NO-GO；下一步冻结pointer/query，只训练65参数existence head，并让
  冻结pointer mass独立定义query标签。P2-M只训65参数head仍为0/32、0/32，证明独立线性
  existence结果不足；P2-N把NULL并入pointer第129类后也只到0/32、6/32。复核发现P2-M同时
  改了冻结范围与极值loss，尚不能归因线性容量；P2-O补做冻结65参数head的BCE-only后为
  26/32、24/32。P2-P增加到8,577参数MLP后反而只有20/32、22/32；
  P2-Q加入pointer utility也仅22/32、23/32。随后复核发现M至Q删除existence tie-break后，正query
  identity与P2-J只重合约90%，此前容量归因无效。下一步冻结P2-J原identity做Action Cache，再
  只训练existence head完成正确M-step。P2-R固定原identity后提高到29/32、27/32；强制cache
  identity解码则双域32/32，剩余断点已收窄为existence整组排序，而不是pointer或候选点。
  P2-S增加平滑全正负对listwise目标仍只有28/32、26/32；P2-T换成8,577参数MLP提高到
  30/32、28/32但未过门；P2-U组合MLP与listwise又回到28/32、26/32。下一步在正确identity
  上重做显式pointer utility。P2-V达到31/32、28/32，确认utility机制有效；只允许完整恢复
  optimizer续到固定4,800步。P2-W得到31/32、32/32并停止续训；将query选择与唯一pointer
  分配合并为固定基数MAP后，P2-X正式重复审计双域32/32，Atom query容量Gate关闭，转入
  未见样本泛化审计。P2-Y在分子隔离route100上确认候选oracle 98/100，但Count仅3/100；
  即使真实Count，joint也只有4/100。P3-A只训Count head后route仍仅8/100，train5k自身也只有
  14.54%，证明冻结表示不足。P3-B全量联合训练将route提高到11/100、known-count 30/100，
  但capacity64崩到5/32、0/32；动态Hungarian身份与fixed-cardinality MAP推理目标冲突，禁止
  继续同协议扩量。P3-C固定4,937图assignment并做25% capacity replay后恢复到22/32、10/32，
  但route回落为0/100；说明固定身份有效而factorized loss仍不对齐joint MAP，下一步进入结构化匹配loss。
  P3-D结构化hinge将route known-count提高到34/100，但固定唯一正matching仍使capacity只有16/32、
  2/32；下一步正分支改为GT-compatible latent matching，不再固定等价query排列。
  P3-E0随后验证精确latent正solver可行，并量化query身份变化而candidate集合稳定；E0b的精确
  best-invalid Hall/Hungarian solver也通过穷举和延迟门。P3-E1将general192从9/192提高到
  155/192，但hard32从32/32崩到2/32。E1b确认固定`mean margin=1`把只漏1个Atom的错误按整图
  K倍尺度继续强推，正式否决该margin；下一步保留latent solver，改做coverage-calibrated A/B，
  并把同分子多风格样本与molecule-disjoint能力验证分开。P3-E2随后证明即使只训已正确的
  capacity64，固定margin也会把hard降到12/32，coverage margin的microbatch4更会级联到2/32。
  P3-E3改为整组梯度累积后完整保住双域32/32，确认约束同一步可见是必要条件；但P3-E4混合
  general后仍只有hard 8/32、general 111/192。下一步不再调margin或追加epoch，而是把capacity
  作为保护域执行A-GEM受约束更新。
  P3-E5中A-GEM按预期多次把冲突点积投影到0，但final hard仅5/32、general 110/192；一阶安全
  不能保证AdamW有限步不跨过离散exact边界。下一步必须检查实际候选更新，break即回滚并缩步，
  不再调gradient-surgery权重。P3-E6事务更新完整保持capacity双域32/32，并把general从9/192
  提高到37/192，但30次中仅9次更新被接受；共享参数可行域过窄。下一步冻结P2-W作为安全base，
  general learner改为独立expert，先审计两者oracle并集和可证安全路由，不继续细分共享步长。
  P3-E7随后在512个molecule-disjoint selector样本上固定单阈值路由，confirm从base 13/256提高到
  111/256，98 rescue/0 break；冻结策略在历史route100也从4/100提高到36/100，32/0。
  base+expert方向正式GO，但当前仍是known-count Atom selector口径，必须补predicted Count与完整流水线。
  P3-E8固定阈值切到P2-W predicted Count后，confirm从111/256跌到4/256、route100从36/100跌到
  0/100；Count exact仅9/256与3/100。当前首要瓶颈已从selector切换为Count，下一步单独训练读取
  candidate/query分布统计的DeepSets Count，不再使用`memory.mean -> linear`。
  P3-E9独立DeepSets Count将confirm从12/256、MAE8.65提高到81/256、1.56，route100从3/100
  提高到34/100；但train85.2%而confirm31.6%，正式NO-GO。下一步保持证据特征不变，扩大训练
  molecule并缩小Count头，避免继续增加共享模型复杂度。
  P3-E10把Count训练扩到829个molecule/3316图后，h64 confirm提高到108/256、MAE1.27，
  route100为37/100、1.57；h32 confirm为109/256、1.16但route100 exact退到29%。数据扩量
  因果有效，缩容出现欠拟合；下一步保留h64并加入Count有序距离监督。
  P3-E11有序期望损失使confirm从108/256退到91/256，正式NO-GO；route100的39/100不能用于
  反向选择checkpoint。下一步恢复普通CE，按molecule组织四variant batch并加入风格一致性监督。
  P3-E12四variant一致性得到confirm 106/256、MAE1.25，未超过E10的108/256；正式NO-GO。
  DeepSets loss/正则支线停止，下一步用不同K的最大匹配分数增量构造结构化cardinality curve，
  直接学习是否继续加入一条Atom matching。
  P3-E13随后对`coarse ±4`九个K分别求base/expert精确一一匹配。窗口覆盖confirm 243/256，
  但refiner只有101/256、MAE1.30，低于P3-E10的108/256、1.27；正式NO-GO。当前Count winner
  保持P3-E10 h64，下一步固定协议扩到style_train_100k做数据量因果A/B，不继续调curve小头。
  P3-E14从style100k严格排除全部评估/路由molecule及scaffold，保留22,733个分子、90,932图。
  在与P3-E10完全相同的99,480 exposure和1,560 step下，confirm提高到117/256、MAE1.08，
  dev提高到120/256，route exact持平37但MAE降到1.28。能力门NO-GO、数据规模因果GO；
  P3-E14成为当前Count研究checkpoint，下一步固定协议增加到3个完整unique-data epoch。
  P3-E15恢复P3-E14模型与AdamW后再完整遍历style90k两次。dev达到134/256、MAE0.87，confirm
  达到120/256、MAE0.91，route100达到39/100、MAE1.12；相对P3-E14全部改善。能力门仍
  NO-GO、exposure因果GO；P3-E15成为当前Count研究checkpoint，下一步固定续到6次完整遍历。
  P3-E16再增加三个完整遍历后，dev148/256、confirm128/256且MAE0.926、route100 41/100，
  首次通过全部四个独立Count门。分桶残差集中在Count41–60和stress；停止追加同一Count
  epoch，下一步冻结P3-E16并接回P3-E7 base/expert router验证predicted-count Atom set。
  P3-E17不重选P3-E7阈值，只接入P3-E16 Count后，route_confirm predicted Atom set从旧
  4/256提高到79/256，历史route100从0/100提高到21/100；known-count 111/256、36/100精确
  复现。Count正确条件expert为107/124而router仅79/124，下一步先做fresh_dev跨字体固定审计，
  不用已消费的route_confirm调router。P3-E18在held-out字体fresh_dev8k上得到candidate recall
  99.6877%、known-count Router 2467/8000，均过预注册门；但Count仅2901/8000、MAE1.98，
  predicted Router为1477/8000，总Gate NO-GO。训练域与fresh_dev在相同Count区间的准确率接近，
  真正断点是style90k高度集中在21–40原子而61–80仅2372图。下一步固定证据做Count区间平衡
  训练，不继续追加同分布epoch；Router跨域利用率不足作为第二断点单独处理。P3-E19精确恢复
  P3-E16后只改逆频率加权CE并完整遍历style90k一次，fresh exact从36.26%提高到39.21%、
  41–60提高7.5pp，但61–80仅提高0.85pp，未过预注册能力门。分布失衡只解释部分残差；全局
  129类Count分类正式停止，下一步改为候选occupancy概率相加的Poisson-binomial Count表示。
  P3-E20在五个Count区间平衡的2560图训练、1280图molecule-disjoint dev上验证该表示：MAE从
  P3-E16的3.42降到2.76，但exact仅28.36%→29.22%，训练集也只有29.34%，严格NO-GO。
  Count-only标签能约束概率总和却不能将128个候选极化为0/1对象；下一步保留Poisson-binomial，
  加入等价候选集合下的动态一一匹配presence监督，先过小容量门再扩量。
- 版本从 `v0.7` 开始，每次发布增加 `0.0.1`。
- 根目录 `VERSION` 是开发版本的唯一来源。

## 当前路线

```text
768×768 图片
    -> target-independent raw LSD evidence anchors
    -> candidate-relative ordinary/text/hashed/polygon router（待训练）
    -> ordinary boundary pair + singleton candidates
    -> pair-local complete-PNG crop + candidate mask
    -> learned relative soft gain + endpoint incidence + disjoint exact matching
    -> bounded normal offset + discrete left/right extent
    -> conservative decoded-line precluster（诊断/后备）
    -> typed affinity + DROP exact-cover（待验证）
    -> frozen physical lines / typed bond groups
    -> Visual SVG v0.2 / q2047 physical primitives
    -> 联合 AtomObject Set：anchor、role、文字区域和 chemical state
    -> 联合 BondObject Set：两个可见端口和键语义
    -> endpoint-to-atom incidence：直接引用 atom query
    -> Pixel OCR：开放字符串转写接口
    -> ChemScene v0.2
    -> invalid-only 结构化原子状态搜索
    -> ChemSVG v0.4 确定性中间编译
    -> RDKit 校验与 SDF
```

关键约束：

- 不使用键长阈值删除候选边。
- 不通过隐藏 atom ID 恢复连接关系。
- OCR 输出开放 UTF-8 字符串，不在 OCR 阶段执行化学理解。
- `Ph/R1/A/X` 等都作为普通字符串处理。
- MR-TADF 不进入训练；从 v0.9.64 开始作为当前结构路线的主要能力测试集。
- locked internal100 保留为回归护栏，不再单独代表目标复杂分子的主要表现。
- 正确、错误和无效预测都保留对应 SDF。
- 双键、三键和楔形键各自是一个 BondObject，内部线段不产生额外 Junction。
- v0.8.33 只完成 known-count 条件下的排序实验，不代表上述链路已接入稳定推理。
- v0.9.2-v0.9.7 仍处于固定 100 图 capacity 阶段，不代表未见分子或 SDF 准确率。
- v0.9.5 已否决 192 个匿名 free group slots，v0.9.6 已否决 latent root/top-k pointer；
  后续直接预测无序 proposal partition，不再人为建立 group identity。
- v0.9.19/v0.9.20 证明 pair-local raster 和空间位置有效，但 3-fold OOF cross 仍为 25/24；
  下一步拆成单向 veto 与受约束 rescue，不将双向 residual 直接接入 shadow。
- v0.9.21 veto 验证单向安全边界；v0.9.22 否决 target group 类别加权。最后 hard errors 需要
  不降采样的 normal/longitudinal junction profile，而不是继续调阈值或 loss 权重。
- v0.9.23 证明无池化 profile 能零误删地减少 cross，但逐 pair 阈值前沿无法安全到零；下一步
  必须转近共线 carrier neighborhood 和组件级 boundary exact-cover。
- v0.9.24 的 ordered neighborhood veto 被 v0.9.23 严格支配；v0.9.25 组件 oracle 将 cross
  16→8、pairable exact 提至 98.39%，下一模型必须直接训练整组 matching，孤立 pair 旁路。
- v0.9.26 已训练整组 matching，但 baseline/raw gain 形成 active-score 捷径，cross 保持 16；
  下一版 forward 必须 score-blind，并通过 counterfactual activation 制造训练期活跃 negative。
- v0.9.27 删除所有 score 输入并反事实激活全量多节点 negative 后，cross 仍保持 16 且 exact
  回退；pair-stage residual/veto/structured component 支线正式停止，下一步回到完整对象 shadow。
- v0.9.90 在 v0.9.89 后增加 warm-start 第二阶段整图 residual move。validation 的
  presence/chemical exact `720/532→726/535`，locked internal test `730/527→734/530`；但
  full-v0 chemical `288→287` 回退 1 张，因此组件 GO、默认部署 NO-GO，前端继续使用 v0.9.89。
- v0.9.91 冻结 presence，只训练已存在边的 `KEEP / RETYPE`。4,500 张 train-only 审计的
  typed exact 为 2,466，一次 bounded RETYPE 可达 2,823。三个 separate seed、累计 9k joint
  softmax 和 global safety gate 的固定 train-dev 最佳均保持 `551/996`；action top-1 虽提高到
  43.75%，却没有净 rescue。正式 NO-GO，不运行 official/locked internal；下一步训练读取
  具体边和目标键型的候选级安全 verifier。
- v0.9.92 对每个具体 RETYPE 动作读取局部 group、当前/目标键型和 action margin，并直接监督
  动作执行后整图是否 exact。official validation chemical exact `532→537`，locked internal
  `527→533`；internal 两域 `239→244`、`288→289`，presence 不变且 0 break。默认结构
  profile 切换到 `structure-v0992-carbon`，OCR 仍未接入。
- v0.9.93 将候选级 RETYPE verifier 的曝光从 4.5k 扩到 9k winner。train-dev 提高到
  `551→558`，但 official validation 仍为 `532→537`，没有超过 v0.9.92。跨风格配对审计中，
  同一分子的四种渲染有 67.7% 出现 presence 结果分裂、56.5% 出现 chemical 结果分裂；主要
  差异来自 layout/stress、multiple-bond offset 和 rotation，而不只是字体。实验 NO-GO，默认
  profile 保持 `structure-v0992-carbon`。
- v0.9.94 首次以同一 molecule 的四种渲染训练 AtomPair 风格一致性。训练期用 GT 匹配把
  匿名 atom query 临时映射到目标原子对；该映射不进入推理。全 context 微调将 mixed outcome
  `86→83`，但 official ChemVision presence `348→340`；129 参数零初始化 adapter 最多只将
  style exact `348→349`，同时 ChemVision `348→347`。`0.00–1.00` scale calibration 无安全
  区间。根因是 style validation 仅 389/498 可由现有 proposal 表达，stress 仅 47/124。
  实验 NO-GO，下一步将 paired consistency 上移到 Atom/Bond proposal 集合。
- v0.9.95 的逐层审计确认 Bond proposal 是 style representability 的第一瓶颈。37,391 参数
  零初始化 heatmap/endpoint residual 将 style 与 official 双域整图 Bond coverage 各提高
  0.40pp；但四渲染 mixed 不变，locked full-v0 deployed/full exact `251/210→244/208`。
  组件 GO、部署 NO-GO；下一步重训与 candidate proposal 几何对齐的 set/incidence 小头。
- v0.9.96 让同一个 set/incidence 同时读取 v0.9.95 新 proposal 和原 proposal。style full exact
  `172→175`、新路径 official pooled `385→387`，但原路径 `376→373`；teacher 10 与
  `0.25–1.0` weight scale 均无逐域零回退解。协议 GO、checkpoint NO-GO；下一步只训练输入
  proposal delta 的零初始化 residual，从结构上保持原路径不变。
- v0.9.97 冻结共享 set/incidence，只训练 2,306 参数 proposal-delta presence/incidence
  residual。epoch 1 在 style 与 official 双域均净增，原 proposal 路径由 bypass 保持；但
  deterministic locked full-v0 Bond/Full `258/221→257/220`。架构/official GO、部署
  NO-GO；下一步用 train-only action verifier 约束 residual 的执行范围。
- v0.9.98 在排除 v0.9.97 replay 后审计 8,000 张 train-only action，仅得到 4 rescue、15
  break，且 target-free 特征高度重叠。安全 verifier 缺少可学习正例，正式 NO-GO；
  v0.9.95-v0.9.98 residual 补丁支线停止，转回大规模 Bond/set/incidence 联合训练与 fresh dev。
- v0.9.99 固定 100k style train、8k fresh dev、4k sealed test 数据协议，并打通三域平衡的
  Bond + set/incidence 联合梯度与 graph-exact surrogate。当前只完成 6 图工程 smoke，fresh
  数据未到位，不作能力结论、不切换默认前端。
- v0.10.0 在完整 8k fresh-dev 上得到 98.19% Bond proposal recall、28.71% Bond exact 和
  20.95% Full object exact 基线。直接联合训练 10,002 张后，同一 1k probe 的 Bond exact
  从 32.9% 崩到 2.8%，而 proposal recall 只下降 0.26pp，证明 Bond 分布移动与 set 决策边界
  失配；本轮 NO-GO，下一步改为 set-first 两时间尺度训练。
- v0.10.1 严格冻结 Bond、只训练 set/incidence 10k。完整 fresh 8k 的 Bond exact 仅
  28.7125%→28.9000%，Full exact 20.9500%→21.1875%，旧域无遗忘。它证明 set-first
  能消除联合漂移，但单独扩量已接近平台；稳定性 GO、能力/部署 NO-GO。
- v0.10.2 从 v0.10.1 热启动，以 `1e-7` Bond 学习率联合训练 10k。fresh probe 的 proposal
  recall 保持 95.35%，但 Bond exact 32.9%→23.1%、Full exact 26.0%→18.3%，旧域同步
  回退。纯时间尺度调整 NO-GO，下一步转向冻结 teacher 的显式 Bond 输出 anchoring。
- v0.10.3 冻结 vision、Atom、旧 Dense Bond 和 set，仅复用最终 AtomObject。新 2.27M
  Direct AtomPair Bond 对全部无序 Atom 对读取 stride-4 `9x32` 二维 ribbon，直接输出
  `None/SINGLE/DOUBLE/TRIPLE/AROMATIC`；不设邻居数、键长或空白中点过滤。6 图工程 smoke
  和正式 10k 均完成。独立 fresh 1k Bond exact `30.6%→45.3%`、edge recall
  `95.15%→95.61%`，通过 `+5pp/-1pp` 门控；架构 GO，推理编译接入前默认前端不切换。
- v0.10.4 不改 v0.10.3 权重或骨架，只对预测图中完整闭合且所有边均有 aromatic 支撑的
  5/6 元环执行整环 chemistry 提升。阈值由 fresh calibration 500 锁定后，独立 fresh 1k
  chemical Bond exact `18.3%→28.2%`、边级键型 accuracy `93.48%→94.95%`；presence 与
  skeleton 逐位不变。编译器 GO，开放 OCR 接入前默认前端仍不切换。
- v0.10.5 针对旧 Atom role head 在 fresh 1k 上 TEXT recall 仅 `0.0094%` 的明确缺口，冻结
  v0.10.4 全部结构模块，新增 anchor-centered 12/20/32px 原始像素角色门。6 图 smoke 已通过，
  正式通用三域 10k 将 role exact `0%→44.7%`、role accuracy `72.92%→99.62%`、TEXT
  precision/recall 提至 `99.47%/99.11%`。角色门 GO，下一步只对 TEXT 输出运行现有 CTC。
- v0.10.6 先否决 v0.8.8 CTC 直接适配：fresh calibration 500 最佳原子状态 accuracy 仅
  56.3%。随后固定 v0.10.5 角色门与预测 extent 1.5× crop，从 v0.8.8 初始化微调部署 CTC；
  2.5k 和 5k 均塌缩为全 blank，TEXT accuracy 0%，提前停止于 6k。正式 NO-GO，独立 fresh
  eval 1k 未读取；下一版拆分真实 TEXT CTC 与 blank gate。
- v0.10.7 从干净 v0.8.8 重新初始化，保持 v0.10.5 角色门，只用已匹配且 GT 字符串非空的
  部署 crop 训练 CTC，移除全部 blank 目标。没有 blank collapse，但 7.5k 最佳 TEXT raw 仅
  10.11%、state accuracy 56.45%、exact 0.2%。v0.10.8 复核确认该结果来自空间 AtomObject
  与源 atom_id 标签错配，且 Pixel Encoder 实际未解冻；checkpoint 已废弃。
- v0.10.8 修复原子身份协议后，同一 v0.8.8 的当前 crop TEXT 为 69.34%，Oracle bbox 为
  85.23%。冻结结构/角色/OCR训练 274k anchor-local bbox 头，fresh 1k 的 TEXT 提至 82.93%、
  state 95.24%、exact 18.9%；bbox 组件 GO，最终 20% exact 组合门仍未通过。
- v0.10.9 从干净 v0.8.8 初始化，按正确 ChemScene 标签、v0.10.8 learned bbox 真正解冻
  758,945 参数 CTC。fresh 1k 的 TEXT 达到 93.13%、state 98.00%、整图 exact 33.3%；
  6/6 Pixel Encoder tensor 均变化，最终 OCR 组合门 GO。
- v0.10.10 无新训练，将 Direct AtomPair、整环芳香编译、role、learned bbox 和 v0.10.9 CTC
  接入统一图片到 SDF 推理；前端推荐 profile 已动态切换。GDG906 的 B/B/N/N/F 全部识别正确，
  结构仍有 85/107 对 84/104 的残差，下一步回到结构集合 benchmark。
- v0.10.11 证明近重复 AtomObject 约解释 fresh 1k 四分之一的集合/骨架失败。27k learned
  affinity 将 Skeleton `45.2%→50.1%`、49 rescue/0 skeleton break，但误合并 3 个合法近邻并
  造成 1 个 Atom break；机制 GO、部署 NO-GO，v0.10.10 仍是默认。
- v0.10.12 扩至 40k safe winner 和 4k/3,974 route select/confirm。confirm 的 Atom exact
  `51.21%→55.28%`、Skeleton `46.53%→50.15%`，162/144 rescue、0 break；但 identity
  precision 99.33% 未过预注册 99.9%，默认仍 NO-GO。下一目标转 safe-suppression action。
- v0.10.13 的方向性 single-delete oracle 在全新 8k 上将 Atom exact `50.43%→61.40%`、
  Skeleton `45.95%→55.04%`，证明动作空间有效；63k learned action head 训练 50k 后仍无法
  同时达到零 harmful、Skeleton `+2.5pp` 和 25% recall，route-select NO-GO，confirm 未读。
- v0.10.14 将 unsafe certificate 与 utility ranking 解耦并运行 molecule-grouped nested OOF。
  三 seed 保守 ensemble 将 Atom/Skeleton 提高 `+2.15pp/+1.84pp`，但 407 次执行仍有 3 次
  harmful，winner recall 17.03%；正式 NO-GO，confirm 未读。失败取证转向逐邻边键级一致的
  structural-twin certificate。
- v0.10.15 用 2px structural-twin 规则保护逐邻接和逐边键级，再由冻结 ranking 每图最多删除
  一次。未见 42k confirm 执行 4,605 次，0 harmful/coverage loss，Atom/Skeleton exact
  `+5.99pp/+5.13pp`，95% safe precision 下界 99.93497%，结构组件 GO。完整 SDF 回归前默认
  仍为 v0.10.10。
- v0.10.16 在 Direct AtomPair 后、OCR/SDF 前执行获证删除并同步过滤入射 pair。20 action 与
  20 control 的端到端 pilot 协议全部一致，SDF valid `18→24`、非立体 exact `9→12`、
  chemical exact `5→6`，0 break；工程 GO，shadow 可用。
- v0.10.17 不训练新权重，将 v0.10.9 正确标签 CTC 接到 v0.9.41 语义 BondObject 主线。
  两个内部 held-out 域 matched OCR 提高 `+9.51pp/+8.02pp`；USPTO 1k valid/chemical exact
  净增 81/55，Skeleton 逐图不变。完整 1k 仍有 17/20 个 paired break，能力 GO、部署 shadow。
- v0.10.18 用 4,000 张内部 teacher 训练单 Text Object 双 OCR utility router。首次 validation
  break-0 阈值在 internal test 有 2 break，未接公开；固定 `0.95` 后 fresh 2k 为 10/0。
  USPTO 1k 的 Atom-typed/labeled/chemical 分别净增 8/7/2，所有 paired break 为 0。
- v0.9.28 冻结八组件完整对象链路。USPTO clean 100 的 SDF valid 为 77%、非立体 exact 21%、
  化学 exact 19%；GD906 无效预测也保留 diagnostic SDF 与渲染图。可试用，但不替换稳定默认。
- v0.9.29 在 USPTO clean 1k 上将 SDF 图分解为四层：纯骨架 72.4%、原子类型+骨架 29.2%、
  键级+骨架 50.5%、完整非立体图 21.2%。数量正确后的连接拓扑为 724/727；下一训练主目标
  改为开放文字/原子类型，cardinality 和键级分别作为第二、第三目标。
- v0.9.30 的对象级 micro 为单原子 96.45%、单键级 89.04%；90.9% 键级错误来自芳香键变单键。
  Open Text 1500k 直接接入和 graph corrector 全关闭均 NO-GO；invalid-only 将 valid 提高 2pp
  但 exact 不变。下一步做最小键级改动的结构化 sanitize 搜索。
- v0.9.31 冻结 Vertex/Junction，以现有端点对 line-band reader 替换 dense 键级。USPTO 100
  的键级整图从 61% 降至 18%、单键级从 96.15% 降至 80.02%，证明窄墨线带仍是错误归纳偏置。
  下一路线为唯一 pair relation + 宽二维对象语义 + 环级 aromatic 编译。
- v0.9.32 的 1.68M 双尺度无序 relation 模型在 degraded GT-pair chemical accuracy 达到 92.99%。
  保守完整环融合将 USPTO 1k 键级整图 50.5%→57.1%、单键级 89.04%→96.51%、完整非立体图
  21.2%→23.6%，验证整体 Bond 语义路线；valid 66.3%，暂不替换 shadow。下一步训练通用
  relation existence，键长只作分层评测，并用最小结构修复补 valid。
- v0.9.33 的 existence head 只读取完整二维 relation view，不读取显式长度。sampled-pair F1 达到
  99.39%，但 100 张内部 dense 全点对 relation set exact 为 70%，因此不直接接入公开推理。
  下一步由模型分数挖掘视觉 hard non-bond，并扩大唯一图覆盖；禁止长键专项训练和键长删除阈值。
- v0.9.34/v0.9.35 用模型分数挖掘视觉 hard non-bond，并将唯一图从 500 扩到 2,000；内部 dense
  clean/degraded exact 提至 78%/76%。但冻结 junction-aware 顶点接入 USPTO clean 100 后
  skeleton 只有 69%，低于旧版 76%，且 paired audit 没有新增修正、回退 7 张。下一步训练输入必须
  改为冻结生成器产生的 predicted Junction，不再用纯 GT point 训练后直接部署。
- v0.9.36 在冻结生成器产生的 predicted Junction 上训练，内部 dense exact 达到 91%/91%，
  USPTO clean 100 skeleton 从 69% 回升到 74%、bond-typed 从 55% 提到 61%。但旧版正确 76%，
  paired 新增修正仍为 0。剩余公开骨架错误主要是 Junction 数量错误，下一步由 Bond 端点证据反向
  提出或删除 Junction；不再继续盲目扩大 pair classifier。
- v0.9.37-v0.9.39 验证绘图器风格适配：MolGrapher 单域专家将 topology 10%→20%，但共享
  权重遗忘 ChemVision。保留双专家后，来源风格 gate 在内部把 MolGrapher 提到 17% 且保持
  ChemVision 55%；然而 USPTO clean 1k 的 skeleton 70.5%→69.3%，56 张路由样本只救回 1 张、
  破坏 13 张。逐图回退工程成立，来源分类目标 NO-GO；下一步改做专家收益可分性审计。
- v0.9.40 utility oracle 在 MolGrapher/ChemVision 各 100 张上得到 topology 原专家 65、
  风格专家 60、并集 88，证明双专家有可利用互补性。后续 route teacher 必须由内部真值定义
  rescue/break，并读取专家分歧和结构自一致性；不使用 USPTO 标签训练。
- v0.9.41 用 2,000 张内部双专家 teacher 训练 no-style utility router。固定 validation 阈值在
  未见内部 996 张上 rescue 93/break 1；USPTO clean 1k skeleton 70.5%→72.7%、valid
  69.1%→70.4%、chemical exact 15.7%→16.0%。进入 shadow，默认替换等待保守 veto。
- v0.9.42 根据 skeleton 72.7% 与 atom-typed skeleton 29.3% 的差距，停止独立 Junction
  和末端最近邻 OCR 附着，建立联合 AtomObject + BondObject 协议。validation/train 的
  1,494 张有效样本全部完成 SDF/InChIKey 无损往返，未知失败为 0；模型权重尚未训练。
- v0.9.43 的双 free-query 联合模型即使训练到 5000 step，Atom 6px 也只有 18.93%、
  Bond port pair 8.22%、incidence endpoint 19.78%，整图 exact 为 0，证明 count 100%
  不能替代对象空间接地。v0.9.44 改为每个 stride-4 cell 直接预测完整 AtomObject，固定
  100 图达到 anchor/role 100%、文字中心 97.54%、known-count exact 85%；空间对象 GO，
  全局平均 count head NO-GO。v0.9.45 冻结空间 Atom/Bond proposal，只训练 74,113 参数的
  endpoint-to-AtomObject incidence head；GT-associated 100 图在 step 500 达到 image exact
  92%，step 1000 endpoint/bond/image 全部 100%。表示容量 GO，下一步必须验证完全不依赖
  GT 对齐的 predicted proposal，尚不替换稳定推理。v0.9.46 使用 target-independent top-k
  proposal，在独立 validation100 将 eligible endpoint 95.36%→97.69%、bond 91.49%→95.58%、
  image exact 13%→46%。学习式 incidence GO；proposal 全覆盖上限仅 64%，下一步训练最终
  Atom/Bond presence set。v0.9.47 的 train100 集合头 validation 校准 full exact 只有 8%；
  v0.9.48 保持架构不变扩到 train1k 后提高到 24%，固定 internal test100 为 19%。数据覆盖
  方向 GO。v0.9.49 使用全部 8,974 条实际 train 记录，validation 校准 full exact 35%，
  固定 internal test100 为 24%。v0.9.50 将对象集合编译为 ChemScene v0.2/SDF；Oracle
  原子化学状态下 scene valid 100%、SDF valid 56%、非立体 exact 30%。44 个 invalid 主要
  来自 aromatic/kekulize。v0.9.51-v0.9.52 的非环芳香键/原子一致性将 valid 提到 82%、
  非立体 exact 提到 36%。v0.9.53 Oracle 匹配键语义只把 exact 提到 40%；v0.9.55 中
  Atom/Bond presence 与 incidence 同时 Oracle 可到 51%。v0.9.54 closure loss 和
  v0.9.56 endpoint rescue 在锁定 internal100 均未提高 exact。v0.9.57 将三类错误放进
  整图最坏项 surrogate 后，locked exact 反而 36%→35%，说明固定 proposal 标签上叠 loss
  仍不够。v0.9.58 使用一次稀疏 MILP 联合决定 Atom/Bond presence 与 endpoint incidence，
  locked internal100 的 SDF valid/非立体 exact/chemical exact 达到 83%/41%/36%，完整正确
  结构新增 5、回退 0。联合解码 GO。v0.9.59 在同一结构上使用匹配键 chemistry Oracle 后
  达到 90%/46%/40%，键语义提供约 5pp 上限。v0.9.60 学习五个全局 joint potential 标量，
  482 个 train-only 可表达图 exact 保持 64.94%，线性势 NO-GO。v0.9.61 非线性头在固定
  train100 将 SDF exact 61%→70%，达到 70/76 可表达图；v0.9.62 的全 9k representable hard
  replay 将 locked internal exact 41%→46%、chemical 36%→41%。v0.9.63 用旧 checkpoint
  invalid-only fallback 将 valid 80% 恢复到 85%，exact 不回退，成为当前结构 shadow。

## 模型规模

- 早期自回归 ChemMolSeq 基线：49,974,768 参数。
- v0.5 完整训练 checkpoint：约 51.49M 参数。
- 删除 decoder 后的并行部署路径：约 15.18M 参数。
- v0.8.33 presence selector：108,289 个参数，v0.8.32 geometry 全部冻结。
- v0.9.2 flat primitive set predictor：19,185,237 参数；endpoint-only 阶段训练
  150,082 个参数。
- v0.9.5 free-slot grouping：2,428,472 个参数；flat primitive model 完全冻结。该分支
  已判定 NO-GO，不进入稳定推理。
- v0.9.6 proposal pointer：1,166,644 个参数；flat primitive model 完全冻结。局部 affinity
  有效，但 independent root selection 已判定 NO-GO。
- v0.9.19 pair-local residual：15,353 参数；v0.9.20 spatial residual：37,313 参数。
  二者均只训练 fit7200 pair stage，生产模型和完整 SDF 推理未修改。
- v0.9.23 profile veto：60,505 参数；v0.9.24 neighborhood veto：108,033 参数。两者均为
  OOF 研究头；v0.9.26 structured component model：154,497 参数，正式结果 NO-GO。
- v0.9.27 score-blind component model：154,369 参数；硬门槛 NO-GO，不进入稳定推理。
- v0.9.45 spatial incidence head：74,113 参数；共享视觉、AtomObject 和 BondObject 全冻结。
  v0.9.46 继续只训练同一 head，在 target-independent predicted proposal 上完成内部独立验证；
  尚未形成部署模型。
- v0.9.47/v0.9.48 proposal object set head：178,565 参数，包含 incidence 与双向
  Atom/Bond presence 一致性；视觉和两个 proposal detector 仍冻结。
- 输入：768×768 RGB。
- 早期共享视觉 token：24×24，共 576 个。

v0.6 之后的部署推理不再需要自回归 decoder，CPU 500 张验证平均约 0.202 秒/张。
当前优先提升模型能力，量化和进一步加速后置。

v0.8.0 已完成 ChemScene v0.1 协议与编译边界。500 张 validation 中 498 张完成
Scene -> ChemSVG -> RDKit -> SDF 无损往返，2 张同坐标冲突被隔离；首个 3,000 张
训练分片中 2,983 张无损往返，8 张源 E/Z 描绘冲突、9 张同坐标冲突，未知失败为 0。
本版尚未训练新权重，稳定推理和 shadow 均不变。

v0.8.1 否决跨图归零的自由 Bond Query，改用 stride-4 空间中心、3 个碰撞 slot、
`cos 2θ/sin 2θ` 无向角度和半长度表达直线对象。固定 100 图 SDF exact 达到 98%；
9k/498 的 object accuracy 达到 67.55%，较自由 query 的 0.0546% 提高约三个数量级，
但校准后的 SDF exact 仍只有 2.61%，因此组件 GO、部署 NO-GO。

v0.8.2 通过零初始化 adapter 融合 stride-8/16 上下文，9k/498 object accuracy
达到 70.66%、SDF exact 达到 4.42%，长键和双键改善明显。v0.8.3 将已确认键裁成
二维线带，只学习可见 display 图元；局部 object accuracy 达到 71.22%，但整图 SDF
exact 回退到 2.61%。因此多尺度几何保留，逐键类型替换 NO-GO，下一阶段加入环级和
分子级一致性目标。

v0.8.4 在冻结的 v0.7.39 Junction 与 v0.8.2 BondObject 预测上训练带 NULL 类别的
端点分配器。2k/498 的 decoded pair exact 从 nearest 标签基线 91.38% 提高到
98.18%，但完整端到端中，激进 learned 模式把 SDF 有效率从 53.41% 提高到 59.84%
的同时，将完整化学图一致率从 19.48% 降到 19.08%；保守 rescue 模式则逐项追平
nearest，没有新增正确结构。因此 assignment MLP 组件成立，但默认替换 NO-GO。

v0.8.5 在预测 Junction/BondObject 构成的整图上做三层节点-边消息传递，只修正
display/chemical 类型时，内部 498 张 SDF 有效率从 53.41% 提高到 55.22%，完整
化学图一致率从 19.48% 提高到 19.68%；paired 审计为新增 2 张、回退 1 张。启用
edge keep 后虽然有效率达到 61.85%，完整一致率却降到 9.24%，因此删边 NO-GO，
type-only 小幅 GO，尚不替换稳定版。

v0.8.6 对 v0.7.8 OCR adapter 做 500-step 修饰符和彩色续训。内部 500 张彩色/黑白
validation 的可见 exact 从 85.62% 提高到 89.16%，但真实蓝色 N 图片仍把四个 N
识别为 `NH/N+/N+/N+`。因此彩色增强机制和 checkpoint 保留，端到端接入 NO-GO。

v0.8.7 从零训练 bbox ROI byte-level CTC。10k crop 最佳 exact 47.15%、修饰符 exact
28.57%，明显低于 v0.8.6；CTC 能抑制部分后缀幻觉，但视觉表征从零学习不足，NO-GO。

v0.8.8 复用 v0.8.6 的 Pixel OCR 三层卷积和 byte 分类权重，保留 bbox 横向 48
个时间步，只训练 50.7 万参数的非自回归 CTC 上下文。10k crop 达到 86.20% exact、
80.70% 修饰符 exact；同一真实彩色图不再产生错误正电荷，但四个普通 N 仍有三个
原始识别为 NH。预测 TextObject bbox 的 500 图协议只有 64.12%，真值 bbox 同分布
对照为 74.80%，说明当前瓶颈同时包含预测框偏差和长尾字符串覆盖。组件路线 GO，
端到端替换仍为 NO-GO。

v0.8.9 将原子 anchor 和文字 bbox 中心拆成两个坐标。结构图继续使用原子 anchor，
新增 130 参数的偏移头只负责 OCR 裁框；同时比较 anchor crop 与偏移 crop，偏移框的
归一化 CTC 分数至少高 0.25 才采用。498 图部署 OCR exact 从 64.74% 提高到
66.58%，修饰符从 21.97% 提高到 27.84%，simple 从 81.29% 提高到 81.58%。
真实彩色图不再把 B 回退为 S，但普通 N 的 NH 幻觉仍未改善，因此组件和实验 profile
GO，稳定替换 NO-GO。

v0.8.10 使用真值 `.text.png` 做四臂 Oracle probe。字符串均衡后的 RGB 基线已把
exact-string macro 提到 55.13%；加入 text mask、anchor 和二维垂直注意力后提高到
57.77%，supported-label macro 从 66.08% 提高到 69.43%。约 3 个百分点的收益证明
二维条件输入有价值，但远低于预设的 10pp 门槛；最大收益来自字符串采样协议，不是
语义 mask。Oracle mask 不进入正式推理，下一阶段优先建设开放字符串数据。

v0.8.11 使用 Open Text v0.1 的 108k train crop 训练 byte CTC。250 条固定样本容量
审计在 epoch 160 达到 100%，证明模型可以表达长度 1-8 的开放字符串。完整解冻 Pixel
Encoder 并分三段累计训练 1.5M 次样本后，validation 真实预测框 exact 从 27.00%
提高到 81.50%，supported-label macro 达到 74.15%；但 modifier 67.49%、常见单字符
86.43%、假修饰符 5.56%，开放随机串 exact 仅 5.62%。容量与真实框协议 GO，正式
接入 NO-GO，下一步转向同风格反事实后缀训练。

v0.8.23 不训练新权重，只在 RDKit sanitize 失败时搜索有限 OCR 原子状态，并强制
Junction/Bond 拓扑不变。internal 498 的 SDF valid 从 52.41% 提高到 67.07%，完整
化学一致从 13.05% 提高到 16.47%。该能力作为确定性 invalid-only 修复保留。

v0.8.24 用相同初始化和 100k 曝光比较 8,974 与 50,000 unique 图片。scaffold 独立
full1k 上，9k/50k 的 SDF exact 为 57.4%/55.7%，topology exact 为 61.8%/59.3%。
50k 只略增候选多样性，没有提高部署 exact，因此停止盲扩到 100k unique。

v0.8.25 将每批固定为 8 coverage + 8 replay，在 6,250 个满 batch 内混合 50k 数据
与旧 9k 能力。SDF exact 恢复到 57.5%，但 topology 与 9k 持平、chemical 低 0.1pp，
internal 几何/拓扑仍比初始化低 4.42pp/5.42pp。sampler GO，checkpoint NO-GO；下一步
转 existence/ranking、cardinality 与 duplicate suppression。

v0.8.26 抽出训练/推理共用的唯一 Junction-pair projection，并增加 detector/selector
显式 A/B。Internal 498 与 fixed Full1k 的 topology exact 分别保持 66.87% 和 60.00%，
Edge F1 只提高 0.039pp/0.013pp；工程协议 GO，能力提升 NO-GO。下一阶段转向投影后的
pair-level existence、quality ranking 和 representable cardinality。

v0.8.27 复核出旧 selector 将同一 GT 键的 23.6 万个替代候选当作普通负样本，并
实现训练专用的 GT atom-pair 等价组 overlay。全候选先 grouping 会把 validation
candidate oracle 从 88.76% 降到约 65%，因此推理顺序继续保持 selector -> grouper。
winner-anchored loss 在固定旧阈值 0.9 下，将 498 张 validation topology 从 70.88%
提高到 71.69%，Edge F1 从 95.72% 提高到 97.34%，同组多选从 0.382 降到 0.269/图；
但未达到预设 72.9% topology 和 76% count 门槛，checkpoint 不接入稳定推理。

v0.8.28 确认 Dense BondObject 同 cell 固定 slot 存在真实置换冲突，并实现 cell-local
Hungarian 匹配、动态正 heatmap、cell-count 与跨 slot duplicate loss。11 项测试证明
目标顺序和单键端点反转不改变 loss；但 400 collision + 400 replay 的正式对照中，
collision geometry exact 仅从 fixed 的 6% 提高到 8%，secondary recall 仅 23.58%，
普通 validation known-object accuracy 反而从 70.06% 降到 66.55%。标签修复保留，固定
栅格多槽位架构按强停止规则 NO-GO；下一步转每根键一个 query、两个无序端点和键级属性
的全局 Bond query/group-query，不再继续调整 slot 排序和权重。

v0.8.29 将每根键改为全局三 token Bond Group Query。matched20 中训练 secondary
达到 50.23%，但 holdout secondary 只有 7.55%、cell exact 只有 1%，说明全局 token
能够记忆训练组，却缺少定位次级碰撞键所需的局部视觉证据，正式判定 NO-GO。

v0.8.30 为每个 base endpoint 从 stride-4 特征采样 `5×5` patch，并将局部残差与全局
路径解耦。容量实验达到 85.15% cell exact、94.4% secondary；冻结全局路径、只训练
196,578 个 local 参数时，holdout 达到 16% cell exact 和 52.83% secondary，证明端点
局部视觉归纳偏置可以泛化。该几何机制 GO。

v0.8.31 沿候选键在 stride-4 特征上增加 line-band。容量为 82.18% cell exact、91.2%
secondary，但 holdout 最佳仅 14%/51.89%，低于 v0.8.30；它增强拟合而没有改善泛化，
因此保留为可选实验分支，不进入下一阶段初始化。

v0.8.32 直接按 Bond proposal 从原图采样 `25×64` 定向墨线带，并与 endpoint patch 独立
叠加。冻结全局路径、只训练 243,812 个参数后，holdout cell exact 达到 18%，secondary
recall 50.94%，collision bond recall 43.69%，protected retention 98.79%，ordinary
known-object 71.20%；所有非几何输出保持不变，正式 `go_checks.go = true`。分桶审计显示
occupancy 2 为 18/94 exact，occupancy 3 仍为 0/6。全部 refined candidates 的 collision
recall 为 43.69%，旧 presence 的 known-count top-k 仅为 9.71%，说明当前明确瓶颈已经
转为排序。该 checkpoint 只作为 geometry 实验基线，稳定推理仍为 v0.7.10。

v0.8.33 在 v0.8.32 best 上冻结 vision、dense detector、Group Query 以及 endpoint
patch/raw-ink geometry，只新增 108,289 参数的局部证据 presence selector。capacity 2k
尚未过门槛，按预先定义的 4k 容量控制继续训练后，representable group recall 达到
99.43%、exactly-one 达到 98.89%、set exact 达到 83%，容量实验 GO。

正式 safe-select 仍只使用 400 collision + 400 ordinary replay，并在固定 100 张 collision
holdout 和 498 张 ordinary validation 上选取 checkpoint。collision known-count recall 从
9.71% 提高到 25.73%，known-count cell exact 从 2/100 提高到 6/100；ordinary geometry
recall 从 95.37% 提高到 95.69%，增加 0.32pp，image exact 从 58.03% 变为 57.83%，仅
回退 0.20pp。vision、dense 和 v0.8.32 geometry 的 tensor hash 逐位不变，全部正式门控
通过，因此 presence 排序方向 GO。

该结论仍有明确边界：训练和 checkpoint 选择使用 GT-known-count 隔离排序瓶颈，没有
训练可部署的 count，也没有更新 display/chemical type，更没有完成 ChemScene -> SDF
端到端评测。因此 v0.8.33 不替换稳定 v0.7.10。下一步应独立训练 representable count
与解码，再固定整条预测链运行 ChemScene 拓扑、SDF 有效率和化学完全一致评测。

v0.7.38 保留 v0.7.37 的 Render Layer 真值、候选等价组和同角色 6px 去重，对
v0.7.36 Bond Object 加入符合真实预测误差分布的端点 jitter。9k/498 上角色严格
vertex exact 保持 56.02%，strict skeleton exact 从 41.37% 提高到 51.00%，匹配
端点条件 Edge recall 为 99.56%，`>96px` 长键条件 recall 从 92.84% 提高到
96.08%。该组件进入 shadow，稳定默认仍为 v0.7.10。

v0.7.39 对 9 个 selector checkpoint 做 Text/Implicit 独立阈值扫描，并加入真组
绝对分数下限与困难背景上限。完整 selector 的 100 图容量达到 88% 部署 exact，
但 9k/498 仅将 role vertex exact 提高到 59.64%、strict skeleton exact 提高到
53.61%，没有通过 62%/57% 正式门槛，且 `>70` 原子组退化。本轮默认 NO-GO，
下一步转向完整邻接 node-edge 消息传递。

v0.9.2 将 AR Visual SVG 改为 unordered primitive set。固定 100 图上，只训练 stride-4
endpoint refiner 后，line recall@8q 从 step-6,000 的 12.92% 提高到 step-11,000 的
85.45%，endpoint p50/p90 为 2.24/18.03q，Bond IoU 为 66.88%。全模型同步续训、
coarse-head-only 和 flat output-head 长训均未通过对照门槛。残差主要集中在 hashed wedge，
因此 flat refiner 机制保留，但主表示转向 group-aware Visual SVG v0.2。

v0.9.3 的 canonical-member group head 和 v0.9.4 的 direct carrier coordinate head 均已
NO-GO。决定性 oracle 表明，保留 flat proposal 坐标后可达到 line r@8q/r@32q
`85.54%/96.18%`、Bond IoU `70.62%`；重新回归 carrier 坐标只有 `4.92%/55.69%` 和
`17.95%` IoU。

v0.9.5 冻结 flat 坐标，仅训练 192 个 free group slots。capacity100 step500 的正式
bond-only r@8q/r@32q 为 `41.14%/42.87%`、Bond IoU 为 `38.10%`，group type/member
count 全部塌为 `single_line/count1`。membership-only 隔离虽把 GT 对齐 group exact 提到
`69.48%`，parallel-2 仍只有 `63.76%`、hashed 只有 `8.11%`。free slots 正式 NO-GO；
下一步使用 proposal-anchored latent representative/pointer，继续保持坐标冻结。

v0.9.6 删除 free slots，训练 1.17M proposal pointer。known-count、oracle usable pool 下
membership F1 提高到 `69.81%`，但 3,378 个 root 中有 548 个重复落在已覆盖 group，
root collision 为 `16.22%`，p2/hashed typed exact 仅 `8.73%/5.41%`，sample exact 为 0。
99.08% 的跨 checkpoint root 稳定率证明主要问题是稳定的全局 root 选择错误，而非训练
抖动。v0.9.6 正式 NO-GO；v0.9.7 删除 root，只训练 typed pair affinity，并用 exact
cover 直接求 proposal 的合法无序分区。

v0.9.7 的 candidate oracle 已通过：perfect GT pair affinity 经共享候选生成器后，3,378
个 group 全部进入候选集。1.10M typed-affinity 训练 500 step 后，exact-cover formal
membership group F1 为 96.12%、partition exact 77%、pair F1 94.37%、line r@8q 85.37%、
Bond IoU 70.17%，invalid 与 fit failure 均为 0，grouping capacity 正式 GO。当前 frozen
primitive 在独立 validation100 的 line F1 只有 0.055%、IoU 7.84%，所以下一步转向
pilot 9k primitive-set 训练，不把上游泛化失败归因于 group decoder。

## 当前最佳结果

当前 G3 独立 Count checkpoint 为 P3-E16 style90932 epoch6 h64：

```text
outputs/smoke/chemvector-g3-p3e16-deepsets-count-style90932-epoch6-h64/step-008665.pt
```

SHA256：`50b7f07a723468f1ef50b21011cc65320af9bad7c6ecfd88be3f5549444ef236`。
confirm256 为128/256、MAE0.926，dev256为148/256、MAE0.902，historical route100为
41/100、MAE1.04，首次通过全部独立Count门。它尚未完成predicted-count Atom set、Skeleton
或SDF验证，不能表述为端到端GO。

v0.7.10 沿用 v0.7.9 checkpoint：

```text
outputs/checkpoints/chemvector-v079-relation-all-pairs-9k/last.pt
```

SHA256：

```text
553fbb80a823ce08b338d1c2b2263a5084ce1b42d714eb4deca7e4728e2427c9
```

在 1,100 个完全未进入训练的 MR-TADF 分子上：

- 有效 SDF：92.09%。
- 化学完全一致：79.73%。
- 原子数一致：82.18%。
- 键数一致：82.18%。
- 平均 CUDA 推理：0.118 秒/张。

当前最强实验 geometry checkpoint 为：

```text
outputs/checkpoints/chemvector-v0832-bond-group-query-ink-line-local-only-early4k/best.pt
```

SHA256：

```text
05c2a7d59f4be9272c55a299c199a301b91081610159e2d1593580c5021b8484
```

它仅通过 collision geometry 受控门控，尚未完成端到端稳定推理验收，不能替代上面的
v0.7.10 checkpoint。

在该 geometry 上冻结训练的 v0.8.33 safe-select checkpoint 为：

```text
outputs/checkpoints/chemvector-v0833-bond-group-presence-generalization-safe-select/best.pt
```

SHA256：

```text
17a394b5a128fd2e20cb6b4d0a5f962992482f1f75c98db7af0620706afb9c29
```

它只完成 known-count 排序验收，尚未包含 deployable count、键型训练或端到端 SDF
验收，不能替代稳定 checkpoint。

v0.7.9 在 v0.7.8 上训练 146,308 个 relation/pair 参数。全点对骨架训练将真值
顶点下的骨架完全一致从 37.35% 提高到 49.40%。v0.7.10 不训练新权重，保留全部
点对概率用于断图桥接；USPTO clean 1k 的有效 SDF从 177 提高到 192，完全一致
保持 22，MR-TADF 完全一致保持 79.73%。

v0.7.11 结构 MoE 已复核为 NO-GO。USPTO/MR-TADF 的结果来自完全回退 v0.7.10，
MolGrapher routed detected skeleton exact 只有 1.3%，不能视为通用能力提升。

v0.7.12 在固定 100 张审计后 MolGrapher 图片上验证集合式顶点：纯 heatmap head
过拟合的 vertex exact 为 87%，解冻一个 stride-4 block 为 90%；Atom Query 负责
计数、dense peak 负责一对一坐标吸附后，Count exact 为 100%，4px/6px vertex exact
为 95%。接入现有 relation 后 skeleton exact 最高 23%，当前第一瓶颈已转为检测后
关系判断。该分支仍处于 shadow mode，不替换 v0.7.10 推理。

v0.7.13 直接在检测顶点上建立 Hungarian 标签映射，保留全部真边并挖掘当前最高分
非键。只训练 4,420 个 relation 参数时 skeleton exact 为 54%；同时训练 146,308 个
pair encoder/relation 参数后达到 89%，超过 85% 过拟合门槛。结果来自固定 100 图
训练内评测，只证明训练协议和容量足够；在完成 MolGrapher 渲染审计及 9k 顶点重训
前，不接入正式推理。上游低概率显示原子序号是主动增强，低精度像素筛查不作为
自动删样依据。

v0.7.14 将顶点分支扩到 MolGrapher 9k/1k。Atom Query 的概率和 top-k Count exact
最高 17.9%，显式全局计数头和峰值计数校准均未通过。dense head 只训练 92,548
参数，在 6px 达到 89.48% F1，但 Count exact 约 20%，vertex exact 最高 6.3%；
追加相同配置曝光已进入平台。因此 relation 扩量继续暂停，下一步受控解冻最后一个
stride-4 视觉 block。

v0.7.15 只解冻最后一个 stride-4 block，将同一 validation 的 6px vertex exact
提高到 9.5%。固定 `19×19` NMS 可达到 23.1%，说明 dense heatmap 中存在大量稳定
次峰；但 USPTO 前 200 张出现更多断图，证明固定像素抑制半径不具备跨尺度泛化能力。
该参数只作为诊断接口，默认仍为 `3×3`。下一步用一对一匹配训练 proposal 保留分类器，
不再依赖全局 count head 或固定宽 NMS。

v0.7.16 用冻结 dense peaks 加一层 56,673 参数 proposal keep head，并以真实顶点
Hungarian 一对一匹配监督重复峰。6px vertex exact 达到 22.0%，Count exact 38.2%，
USPTO clean 1k 有效 SDF 由同权重基线 63 提高到 107。该增益可以跨域，但整个 shadow
checkpoint 仍低于稳定 v0.7.10 的 192 个有效 SDF 和 22 个完全一致，因此 proposal
head 暂不成为默认生产路径。

v0.7.17 将同一 keep 架构直接训练在稳定 v0.7.10 特征上，并加入 `0.65–1.0` 在线
尺度增强。USPTO clean 1k 有效 SDF 从 192 提高到 226，原子数一致从 8.7% 提高到
10.1%；但化学完全一致从 22 降到 19，断图从 192 增加到 320。该结果证明删重复峰
有效，但低置信真顶点仍被过删，因此 keep 继续作为可选 shadow 组件，不改变默认基线。

v0.7.18 不再先删点后看边。它对最多 192 个 proposals 的全部点对采样原图线带，
用真实 ChemMol adjacency 监督二元 edge head，再把最强 4 条 incident edge support
作为 residual rescue 输入节点。内部 1k 的 6px vertex exact 为 7.5%、edge F1 为
78.35%。未参与调参的 USPTO clean 1k 有效 SDF 为 241、化学完全一致 29，超过
v0.7.10 的 192/22 和 v0.7.17 的 226/19。断图仍为 254，高于稳定基线 192，因此
当前作为推荐 shadow 候选，不静默替换稳定默认。

v0.7.19 尝试在 `model_connect` repair 阶段用 graph edge score 提高补桥排序。固定内部
阈值 `0.85` 后，USPTO clean 1k 只改变 2 个样本：有效 SDF 241→242、完全一致仍为
29、断图 254→252、价态失败 436→437。该后处理方案已否决并默认关闭。下一步必须
在 relation 训练阶段学习融合，而不是继续在 repair 末端或公开集上调阈值。

v0.7.20 在 192 个 proposals 上增加两层全局 transformer，并允许 node-edge 联合
反向传播。held-out 字体 validation 的 6px vertex exact 从 v0.7.18 的 32.93% 提高
到 36.75%，MolGrapher 保持 7.70%；USPTO clean 1k 有效 SDF/完全一致提高到
258/30。但两个内部域选出的统一 threshold `0.90` 在大型 MR-TADF 上过删顶点，完全
一致从稳定版 79.73% 降到 49.00%。因此全局图组件 GO，固定阈值部署和默认替换 NO-GO。

v0.7.21 将顶点主线改为集合式 Atom Query 直接输出原子数量与坐标。固定 100 图
能力测试中，低学习率视觉微调和尾部坐标损失把纯 query 的 6px vertex exact 从
60% 提高到 74%–75%；增加第二次、64px 半径的可微局部 refinement 后达到 97%，
Count exact 100%，不再依赖 dense peak 吸附。该结果只证明架构容量和训练协议通过，
尚未完成独立字体/scaffold 泛化门控，不替换 v0.7.10。随后在 MolGrapher 9k 训练、
scaffold 独立 1k 验证上，6px F1 从旧 Atom Query 的 20.40% 提高到 39.95%，16px
F1 从 54.72% 提高到 68.46%；Count exact 仅 20.2%，整图 exact 仍为 0。坐标主干
已明显改善。Poisson-binomial 计数分布只训练 193 个 presence 参数时 Count exact
为 21.4%；让计数损失进入最后两层 query decoder 后达到 23.9%、MAE 1.648，6px
F1 同时提高到 40.14%。仍没有整图 exact，当前主瓶颈是需要显式全局 cardinality
token 的集合数量建模。

v0.7.22 实现显式全局 cardinality token：两层注意力先读取 atom query 集合，再读取
整图视觉 memory，同时保留 Poisson-binomial set count loss。固定 100 图在曝光
2,000 张时 Count exact 达到 100%，证明结构容量成立；但 MolGrapher scaffold 独立
1k 上，token-only 分类最高 18.1%，连续回归最高 17.0%，token 与最后两层 query
decoder 联合训练后 token 分类为 21.5%、set count 为 22.7%，均低于 v0.7.21 的
23.9%。6px F1 保持在 40.22%，没有坐标回退。

本轮新增的 ground-truth count Oracle 给出了更重要的修正：即使直接提供真实原子数，
6px F1 仍只有 40.22%、vertex exact 仍为 0；16px vertex exact 也只有 0.1%。因此
先前“失败已由坐标主导转为 cardinality 主导”的判断不成立。计数不准主要是 query
候选覆盖、排序和坐标错误的结果，而不是独立计数头可以解决的原因。显式 cardinality
token 单独方向判定 NO-GO；下一阶段应直接提高 query set 的 Oracle-count 上限。

v0.7.23 对 Atom Query 的正样本协议做了隔离审计。原训练匹配使用归一化坐标 L1
距离减 `0.2 × presence`，高分但更远的 query 会持续抢走正样本；删除 presence 后，
同一候选池的 6px Hungarian F1 从 44.72% 提高到 62.50%，L2 为 62.70%。正式训练
改为纯坐标 L2 匹配，并把计数 presence、最终坐标感知 selection、冻结的旧
ranking presence 副本拆为独立分支。MolGrapher scaffold 独立 1k 上，4k 坐标适配将 coordinate Oracle
提高到 65.39%；冻结坐标训练 selector 后，known-count F1 为 55.10%，现有 duplicate
penalty 最高 55.45%。独立校准 193 个计数参数后的部署型 set-count F1 为 54.79%，
Count exact 为 19.6%。该版本证明分配与排序修复有效，但整图 exact 仍未过门槛，
不替换稳定 v0.7.10。完整记录见
[v0.7.23 训练档案](training/20260716-v0.7.23-coordinate-query-selection.md)。

v0.7.24 对 ranking 分数组件做零训练消融：旧 ranking prior 单独为 41.03%，仅
local+context 为 47.93%，完整组合为 55.12%，因此旧 prior 当前仍有正贡献，不能
直接删除。随后把已有 stride-4 dense vertex head 打包进 query checkpoint，复用同一
视觉主干和同一次 stroke feature 前向，只增加 92,678 个参数。对 query 选出的顶点做
一对一 dense peak 吸附后，MolGrapher 1k 的部署型 6px F1 从 54.79% 提高到 76.61%，
coordinate Oracle 从 65.39% 提高到 93.62%，Oracle vertex exact 从 1.8% 提高到
40.2%。跨字体 internal 500 的部署型 F1 为 76.77%，没有同域专属回退。完整骨架
1k 评测中顶点 precision/recall 为 74.37%/72.38%，但 relation edge recall 最高只有
53.81%，skeleton exact 仍为 0，因此 v0.7.24 仅判定顶点精定位组件 GO，不替换稳定
推理。完整记录见
[v0.7.24 训练档案](training/20260716-v0.7.24-shared-dense-query-snap.md)。

v0.7.25 固定 v0.7.24 的 query、共享视觉主干和 packed dense head，只训练 146,308 个
pair encoder/relation 参数。MolGrapher scaffold validation 1k 上，旧 relation 的
最佳 Edge F1 为 52.56%，新 relation 为 62.70%，假边从 11,065 降到 4,408；跨字体
internal 500 的 Edge F1 从 53.99% 提高到 58.28%。因此 relation 适配组件判定 GO。
但 vertex exact 和 Count exact 仍使 skeleton exact 为 0，当前 checkpoint 不替换稳定
v0.7.10。完整记录见
[v0.7.25 训练档案](training/20260716-v0.7.25-shared-detected-relation.md)。

v0.7.26 否决了会让低质量 query 抢峰的全 query Hungarian 吸附，改用 dense 峰作为
固定候选实体，并为每个峰聚合附近 Atom Query 的全局上下文。MolGrapher validation
1k 的 6px 顶点 F1 从 76.61% 提高到 90.02%，Count exact 提高到 27.5%，vertex exact
提高到 6.1%；跨字体 internal 500 顶点 F1 为 92.12%。接入冻结的 v0.7.25 relation
后，两套数据首次得到 5.7% 和 4.2% skeleton exact。完整记录见
[v0.7.26 训练档案](training/20260716-v0.7.26-dense-candidate-query-fusion.md)。

v0.7.27 让 v0.7.26 的最多 128 个候选点进行全无向点对原图线带采样，并加入中间
候选 blocker 特征。固定 100 图两阶段容量实验先把 Edge F1 从 8.57% 提高到 80.44%，
再冻结 edge reader 训练有界节点残差；真实数量已知时，vertex exact 从 12% 提高到
18%。部署型 set-count 仍偏向少选点，因此本轮只判定 edge reader 与候选排序容量
GO，不扩到 9k，也不替换稳定推理。下一步单独校准 count。
[v0.7.27 训练档案](training/20260717-v0.7.27-grounded-visible-line-graph.md)。

v0.7.28 冻结 v0.7.27 的候选排序，新增只预测 top-k 数量的独立 Count head。压缩为
mean/max/sum 的第一版固定 100 图只有 23% Count accuracy，已否决；保留完整排序
节点/边分布后，在 12,000 次样本达到 100% Count accuracy，calibrated vertex exact
与同路径 known-count 一致为 16%。Count 架构容量 GO，下一步先训练 9k graph 上游。
[v0.7.28 训练档案](training/20260717-v0.7.28-independent-grounded-count.md)。

v0.7.29 发现并修复高精度矩阵乘法模式下 `torch.cdist` 的近坐标相消误差，将坐标
监督与评测统一为 6px，并用最弱真点/最强假点 whole-set margin 直接优化整图集合。
固定 100 图 Candidate Oracle exact 为 57%，known-count exact 从 12% 提高到 45%；
接回独立 Count 后 batch-1 calibrated exact 为 41%。当前第一上限重新明确为候选池。
[v0.7.29 训练档案](training/20260717-v0.7.29-whole-set-exact-objective.md)。

v0.7.36 否决自由 Bond Queries 后，改为对双分支端点的全部无向点对读取 25x64
二维宽线带，每个端点对只输出一个 Bond Object。真实端点的 9k/498 泛化达到
99.16% Edge F1、78.31% skeleton exact，双键/三键 edge recall 为 99.04%/100%。
接入预测端点后，匹配端点条件 Edge recall 仍为 98.70%，但 9.84% vertex exact 将
strict skeleton exact 限制在 7.83%。键对象方向 GO，下一步仍优先解决整组顶点，并
为芳香/Kekule 键型增加环级上下文。
[v0.7.36 训练档案](training/20260717-v0.7.36-node-conditioned-bond-object.md)。

v0.7.7 字体扩展实验将内部可见 OCR 提高到 96.89%，但整体微调 OCR head 导致
MR-TADF 完全一致降至 71.27%；插值和新旧字体混合训练也未通过回归门槛。因此
三个 v0.7.7 checkpoint 均已否决。v0.7.8 已改用冻结基线和零初始化残差适配器。
角色头继续保持影子模式。当前优先改进顶点集合、Count exact 和漏点恢复，OCR 暂时
冻结；relation 已完成对 shared dense-snap 顶点分布的适配，不再继续单独扩量。

## 模块边界

```text
image_io.py             图片准备、PIL/Tensor 转换
detection.py            vertex 检测与锚点 OCR
stroke.py               点对监督与线段分类
parallel_v6.py          无 decoder 并行结构预测
repair_v4.py            墨线拓扑与化学图修复
sdf_io.py               SDF 写入及立体往返校验
evaluation_metrics.py   纯图比较、叠加图与汇总指标
benchmark_diagnostics.py 图级错误分类与回填
infer_*.py              参数解析与结果落盘
evaluate_detected_v5.py 数据迭代与 pipeline 调度
```

核心模块不依赖 `infer_*` CLI；repair 不依赖 generation；并行推理通过公开检测 API
运行，不引用 generation 内部函数。

## 数据与协议

- [ChemMolSeq v0.2](../../Data/sdf/CHEMMOLSEQ_V0_2.md)：最早直接 graph 序列。
- [ChemMolSeq v0.3](../../Data/sdf/CHEMMOLSEQ_V0_3.md)：空间排序短序列基线。
- [ChemSVG v0.4](../../Data/sdf/CHEMSVG_V0_4.md)：当前主协议。
- [ChemScene v0.1](../../Data/sdf/CHEMSCENE_V0_1.md)：v0.8 对象场景协议。
- [ChemScene v0.2](../../Data/sdf/CHEMSCENE_V0_2.md)：v0.9.42 联合 AtomObject/BondObject 协议。
- [WebDataset 数据说明](../../Data/sdf/README.md)：训练样本和加载器协议。
- [MR-TADF benchmark](../benchmarks/MR_TADF_V1.md)：外部测试、历史结果和错误诊断。

## 训练档案

完整时间线见 [训练档案索引](training/README.md)。主要阶段：

| 阶段 | 简要说明 |
|---|---|
| [001](training/001-grounding-100.md) | 100 样本视觉接地，证明模型确实读取图片 |
| [002](training/002-chemmolseq-v03-1k.md) | ChemMolSeq v0.3 短序列和空间排序 |
| [003](training/003-chemsvg-v04-initial.md) | ChemSVG v0.4 点线端点协议 |
| [004](training/004-v05-ocr-decoupling.md) | 几何与开放字符串 OCR 解耦 |
| [005](training/005-v05-visual-relation-repair.md) | 全点对连线与真实墨线修复 |
| [006](training/006-v06-parallel-cpu.md) | 无 decoder 并行 CPU 推理 |
| [007](training/007-v072-factorized-relation.md) | 因子化 Edge/Bond head，端到端未通过 |
| [008](training/008-v072-replacement-ocr.md) | 平衡字符替换 OCR |
| [009](training/009-v073-pixel-ocr.md) | 原图高分辨率 Pixel OCR |
| [010](training/010-v075-chemvision-multiscale-ocr.md) | ChemVision 三尺度 OCR |
| [011](training/20260715-v0.7.6-vertex-recall-pilot.md) | 9k vertex-head pilot，完成 |
| [012](training/20260715-v0.7.7-font-expanded-ocr.md) | 字体扩展有效，但整体微调遗忘旧分布，checkpoint 全部否决 |
| [013](training/20260715-v0.7.8-ocr-residual-adapter.md) | 冻结旧 OCR 的 29k 残差适配器，完成 |
| [014](training/20260715-v0.7.9-skeleton-all-pairs.md) | 全点对 relation 与概率约束 repair |
| [015](training/20260715-v0.7.10-probability-bridge.md) | 使用连续点对概率恢复断图，当前最佳 |
| [019](training/20260715-v0.7.11-conservative-structure-moe.md) | 风格 gate 只实现回退，已判定 NO-GO |
| [020](training/20260716-v0.7.12-atom-query-vertex.md) | Atom Query 集合计数与 dense 几何吸附，顶点方向 GO |
| [021](training/20260716-v0.7.13-detected-relation-hard-negatives.md) | 检测顶点 Hungarian 监督与困难负边，100 图过拟合 GO |
| [022](training/20260716-v0.7.14-molgrapher-9k-vertex-scale.md) | MolGrapher 9k 顶点扩量，dense 局部 GO、整图 NO-GO |
| [026](training/20260716-v0.7.18-joint-proposal-graph.md) | 原图线带联合 node-edge proposal graph，组件与 shadow GO |
| [027](training/20260716-v0.7.19-graph-edge-prior.md) | repair 末端 graph edge prior 覆盖过低，NO-GO |
| [028](training/20260716-v0.7.20-style-global-proposal.md) | 跨字体全局 proposal graph，组件 GO、固定阈值部署 NO-GO |
| [029](training/20260716-v0.7.21-trainable-vision-atom-query.md) | 双阶段 query：100 图 97%；独立 1k F1 40.14%，Count exact 23.9% |
| [030](training/20260716-v0.7.22-global-cardinality-token.md) | 全局计数 token 未过 30% 门槛；真实计数 Oracle 证明瓶颈仍是 query 候选质量 |
| [031](training/20260716-v0.7.23-coordinate-query-selection.md) | 纯坐标分配与独立排序，部署型 6px F1 提高到 54.79% |
| [032](training/20260716-v0.7.24-shared-dense-query-snap.md) | 共享 dense 顶点吸附，部署型 6px F1 提高到 76.61% |
| [033](training/20260716-v0.7.25-shared-detected-relation.md) | 新检测顶点分布 relation 重训，MolGrapher Edge F1 提高到 62.70% |
| [034](training/20260716-v0.7.26-dense-candidate-query-fusion.md) | Dense 候选与 Query 上下文融合，顶点 F1 90.02%，首次得到非零 skeleton exact |
| [035](training/20260717-v0.7.27-grounded-visible-line-graph.md) | 全候选点对读取真实线条，100 图 Edge F1 80.44%，known-count vertex exact 18% |
| [036](training/20260717-v0.7.28-independent-grounded-count.md) | 独立 Count 在 100 图达到 100% accuracy，并追平 known-count vertex exact |
| [037](training/20260717-v0.7.29-whole-set-exact-objective.md) | 修复 TF32 距离监督；whole-set known-count exact 12%→45%，接 Count 后 41% |
| [038](training/20260717-v0.7.30-high-recall-candidate-visual.md) | 256 候选 Oracle exact 88.7%；3x3 坐标对齐视觉将 9k/1k strict vertex exact 提至 4.4% |
| [039](training/20260717-v0.7.31-extreme-ranking-no-go.md) | 候选身份极值损失误伤 6px 内等价候选，20k exact 4.4%→0.5%，NO-GO |
| [040](training/20260717-v0.7.32-semantic-vertex-oracle.md) | 文字对象中心使跨字体 F1 75.69%→85.19%、exact 0.2%→2.6%；bond-only 无额外收益 |
| [041](training/20260717-v0.7.33-text-object-center.md) | 独立文字对象分支 internal F1 94.30%、exact 55.60%；合并顶点 F1 75.04%→83.14% |
| [042](training/20260717-v0.7.34-dual-vertex-branches.md) | 文字/隐式双分支 internal F1 92.62%、exact 9.6%；原始 dense exact 仅 0.4% |
| [043](training/20260717-v0.7.35-bond-object-pilot.md) | 自由 Bond Queries 100 图端点 recall 96.65%，9k 仅 4.62%；转向双分支端点条件化 |
| [044](training/20260717-v0.7.36-node-conditioned-bond-object.md) | 全端点对宽线带 Edge F1 99.16%；预测端点条件 Edge recall 98.70%，顶点仍为瓶颈 |
| [045](training/20260717-v0.7.37-render-label-group-selector.md) | 角色严格 vertex exact 56.02%；接 Bond Object 后 strict skeleton exact 41.37% |
| [046](training/20260717-v0.7.38-endpoint-jitter-bond.md) | 端点 jitter 将 strict skeleton exact 提至 51.00%，长键条件 recall 提至 96.08% |
| [047](training/20260717-v0.7.39-deployment-boundary-calibration.md) | 部署边界将 role vertex exact 提至 59.64%，但未过门槛，默认 NO-GO |
| [048](training/20260717-v0.7.40-full-adjacency-graph-denoiser.md) | 完整邻接容量 89%，9k skeleton 52.81%；强边 teacher 无收益 |
| [049](training/20260717-v0.7.41-joint-node-edge-denoising.md) | 联合 node-edge 将 role exact/skeleton 提至 59.44%/54.62%，线内假点成为主问题 |
| [050](training/20260718-v0.7.42-endpoint-ray-evidence.md) | 射线证据将 3px 线内假点 162→107，但新增漏点，默认 NO-GO |
| [051](training/20260718-v0.7.43-straight-through-context.md) | 共线角度摘要只在漏点/假点间折中，正式指标未提升 |
| [052](training/20260718-v0.7.44-candidate-patch-ab.md) | 原图 patch 线内假点减少但漏点增加，受控 A/B 判定 NO-GO |
| [053](training/20260718-v0.7.45-semantic-pass-through-context.md) | 独立穿越分类的 100 图容量 87%，未进入 2k |
| [054](training/20260718-v0.7.46-candidate-coverage-grid.md) | 降阈值和 384 容量几乎不抬 Oracle，上游候选头成为明确缺口 |
| [055](training/20260718-v0.8.0-chemscene-bond-first.md) | ChemScene 对象协议与 SDF 编译完成，下一步进入 Bond Query 容量实验 |
| [056](training/20260718-v0.8.1-spatial-bond-object.md) | 空间线对象解决自由 query 跨图归零，9k 局部 67.55%，整图仍待提升 |
| [057](training/20260718-v0.8.2-multiscale-bond-object.md) | 多尺度上下文将 object 提至 70.66%、SDF exact 提至 4.42% |
| [058](training/20260718-v0.8.3-line-band-type-refiner.md) | 线带提高局部分类但损害整图 exact，转向结构化类型一致性 |
| [059](training/20260718-v0.8.4-junction-bond-assembler.md) | NULL 分配提高有效率但未提高端到端 exact，默认 NO-GO |
| [060](training/20260718-v0.8.5-chemscene-graph-corrector.md) | 图级 type-only 首次净增完整正确结构，删边 NO-GO |
| [061](training/20260718-v0.8.6-color-modifier-ocr.md) | 彩色 crop 提升但真实彩色分子未泛化，接入 NO-GO |
| [062](training/20260718-v0.8.7-bbox-ctc-pilot.md) | 从零 CTC 只有 47.15%，下一步复用 Pixel OCR encoder |
| [063](training/20260718-v0.8.8-pretrained-bbox-ctc.md) | 复用视觉 encoder 后 crop exact 86.20%，预测框协议仅 64.12%，暂不替换默认 OCR |
| [064](training/20260719-v0.8.9-asymmetric-text-bbox.md) | anchor/bbox 中心解耦后部署 OCR 64.74%→66.58%，真实图无回退但 NH 仍未解决 |
| [065](training/20260719-v0.8.10-oracle-text-layer-ab.md) | Oracle mask+anchor 将 macro 55.13%→57.77%，主要收益来自 balanced sampling |
| [066](training/20260720-v0.8.11-open-text-ctc.md) | 容量审计 100%，真实预测框 81.50%；modifier、假后缀和开放随机串仍未过门槛 |
| [069](training/20260720-v0.8.15-line-band-bond-set-selector.md) | line-band 能区分真/假键，但 learned count 泛化失败，9k 拓扑 33.73% |
| [070](training/20260720-v0.8.16-bond-endpoint-junction.md) | 每个 Bond 固定两个语义端点，结构拓扑 67.67%；端到端 OCR 附着失败使化学一致仅 12.45% |
| [073](training/20260720-v0.8.19-endpoint-junction-grouper.md) | 修复亚像素距离审计后，纯几何 Junction grouping validation 拓扑为 75.70% |
| [074](training/20260720-v0.8.20-visual-endpoint-junction-grouper.md) | 方向对齐 endpoint patch 将 validation 拓扑提高到 77.91%，通过 75.30% 门槛 |
| [075](training/20260720-v0.8.21-junction-aware-bond-selector.md) | 无 GT Junction、无 learned count 的 9k selector 将 validation 拓扑提高到 70.88% |
| [076](training/20260720-v0.8.22-junction-aware-inference.md) | 正式 internal 498：Junction exact 68.67%、SDF valid 52.41%、完整化学一致 13.05% |
| [077](training/20260721-v0.8.23-structured-atom-decoder.md) | invalid-only 结构化原子搜索将 SDF valid 提至 67.07%、完整化学一致提至 16.47% |
| [078](training/20260721-v0.8.24-data-scale-matched-compute.md) | full1k：9k SDF 57.4%，50k 55.7%；matched-compute 扩量 NO-GO |
| [079](training/20260721-v0.8.25-coverage-replay.md) | 50k+9k replay 恢复到 57.5% SDF，但跨风格遗忘仍在，checkpoint NO-GO |
| [080](training/20260721-v0.8.26-junction-pair-projection-consistency.md) | 统一 selector-score pair 投影；Internal/Full1k topology exact 净增益均为 0，工程 GO / 能力 NO-GO |
| [081](training/20260721-v0.8.27-equivalence-bond-selector.md) | winner-anchored 等价组监督将 validation topology 70.88%→71.69%、extras 0.382→0.269/图；正式门槛未过 |
| [082](training/20260721-v0.8.28-dense-slot-hungarian.md) | slot 置换冲突修复通过不变性测试，但 collision exact 仅 6%→8%、secondary recall 23.58%；固定多槽位架构 NO-GO |
| [083](training/20260721-v0.8.29-bond-group-query.md) | 全局三 token matched20 明显记忆；holdout secondary 7.55%、cell exact 1%，NO-GO |
| [084](training/20260721-v0.8.30-endpoint-local-refinement.md) | stride-4 endpoint patch 的 local-only holdout 达到 16% cell exact、52.83% secondary，geometry GO |
| [085](training/20260721-v0.8.31-feature-line-band.md) | 特征 line-band 提高容量但 holdout 仅 14%/51.89%，泛化 NO-GO |
| [086](training/20260721-v0.8.32-raw-ink-line-view.md) | 原图定向线带将 holdout cell exact 提至 18%，通过 geometry 门控；不替换稳定推理 |
| [087](training/20260721-v0.8.33-local-evidence-presence-selector.md) | 局部证据 presence 排序提高 collision recall，但 count/type 未训练，不替换稳定推理 |
| [088](training/20260721-v0.9.0-raster-to-visual-primitives-capacity.md) | 建立 q2047 Visual SVG；旧 cross-attention 视觉未接地 |
| [089](training/20260721-v0.9.1-visual-prefix-grounding-and-capacity.md) | Visual Prefix 固定 100 图容量 GO |
| [090](training/20260721-v0.9.1-unseen-molecule-generalization.md) | AR token sequence 未见分子门控 NO-GO，转 unordered set |
| [091](training/20260721-v0.9.2-unordered-primitive-set-capacity.md) | flat set endpoint refiner 达到 85.45% recall@8q；coarse/output-head 分支 NO-GO；v0.2 group 协议 GO |
| [092](training/20260721-v0.9.3-bond-group-unordered-set-p1.md) | 两阶段 group set 达到 r@8q 9.50%、IoU 18.32%；canonical 32-member head NO-GO，转 typed carrier/envelope |
| [093](training/20260721-v0.9.4-carrier-and-flat-proposal-oracle.md) | direct carrier 坐标回归 NO-GO；flat proposal + oracle grouping 保留 85.54% r@8q、70.62% IoU |
| [094](training/20260721-v0.9.5-frozen-proposal-free-group-slots.md) | free group slots 类型/成员数塌缩；正式 r@8q 41.14%、IoU 38.10%，转 proposal-anchored pointer |
| [095](training/20260722-v0.9.6-proposal-anchored-pointer.md) | pointer 学到局部 affinity，但 root collision 16.22%、p2 exact 8.73%、sample exact 0，root 正式 NO-GO |
| [096](training/20260722-v0.9.7-typed-affinity-packing.md) | exact-cover formal group F1 96.12%、partition 77%、r@8q 85.37%、IoU 70.17%；grouping capacity GO，上游 9k 待训 |
| [097](training/20260722-v0.9.8-primitive-set-pilot9k.md) | fresh full-model primitive-set 扩到 pilot 9k，首轮 1,000-step 泛化门控训练中 |
| [098](training/20260722-v0.9.9-evidence-carrier-extent.md) | evidence carrier extent best step900：q8 member 98.39%、平移 32px 下降 0.30pp；sample exact 72.4%，局部 GO / 完整 P0 未过 |
| [099](training/20260722-v0.9.9-p05-target-free-decoded-line-clustering.md) | target-free complete-link 零误合并、零 q8 损失，压缩 51.32%；pair F1 98.07%、partition exact 79%，仅保守预聚类 GO |
| [100](training/20260722-v0.9.9-p1-candidate-relative-router-audit.md) | P1 选择 645 参数 candidate router 显式分流 text/hashed/polygon，不在 LSD 前擦除像素 |
| [101](training/20260722-v0.9.9-boundary-stroke-pairing.md) | molecule-disjoint 边界 pair recall 99.45%，但 hard merge 仍有 cross-target、one-positive 与 p2 误合并，转 pair/singleton soft matching |
## 常用入口

创建训练环境：

```bash
python -m venv --system-site-packages .venv-train
.venv-train/bin/pip install -r requirements-train.txt
```

运行全部测试：

```bash
.venv-train/bin/python -m unittest discover -v
```

查看 benchmark：

```bash
.venv-train/bin/python frontend/server.py
```

服务默认监听 `0.0.0.0:8765`；浏览器使用服务器 IP 和端口 `8765` 访问。

## 后续文档维护

新训练不得继续扩写本页。使用
[`training/TEMPLATE.md`](training/TEMPLATE.md) 新建独立记录，完成后只更新：

1. [`training/README.md`](training/README.md) 的时间线和当前下一步；
2. 本页“当前版本”“当前最佳结果”和训练档案表中的一行摘要；
3. 对应 benchmark 文档中的外部评测结果。
