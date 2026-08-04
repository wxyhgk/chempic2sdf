# ChemVector 训练档案

本目录保存每一次已执行训练或候选实验的独立记录。模型主文档只维护当前架构、
当前最佳结果和本目录索引，不再承载完整训练日志。

## 文档规则

1. 每次开始新的正式训练、消融实验或候选版本时，新建一个 Markdown 文件。
2. 新文件命名为 `YYYYMMDD-vX.Y.Z-topic.md`；历史实验使用已有实验编号命名，
   不补造未知日期。
3. 已结束实验的命令、指标和结论不得被后续结果覆盖。需要修正时，在原文增加
   “复核”小节，并说明修正原因。
4. 每篇必须记录数据范围、初始化 checkpoint、可训练参数、训练命令、门控指标、
   输出目录、失败方案和最终决策。
5. 新实验完成后，同时更新本页时间线和上一级 [ChemVector 主文档](../CHEMVECTOR.md)
   的一行摘要。
6. MR-TADF 只作为外部 benchmark，不得写成训练数据。
7. 每个产生正式 checkpoint 的模型版本必须先提交对应代码、配置和训练记录，再创建与模型
   版本名完全一致的 annotated Git tag，例如模型 `v0.9.90` 对应 tag `v0.9.90`。
8. tag 注释必须记录 checkpoint 路径、SHA256、训练样本数以及 GO/NO-GO 部署结论。pilot、
   smoke 和中间 step 不单独占用正式版本 tag；它们统一记录在对应版本的训练文档中。
9. 禁止把 `.pt` checkpoint、训练数据或生成图片提交到 Git。tag 固定的是可复现代码和协议，
   模型权重通过训练文档中的路径与 SHA256 对应。

新建文档时复制 [训练记录模板](TEMPLATE.md)。

## 训练时间线

当前已完成实验：[G3 P3-E11 Ordinal Count](20260804-g3-p3e11-ordinal-count.md)：
`CE+0.1*期望Count Huber`使dev提高但confirm从108/256退到91/256，route100为39/100；正式
NO-GO并恢复普通CE。下一步按molecule组织四variant batch，加入风格一致性监督。

当前已完成实验：[G3 P3-E10 Count 数据量 / 容量 A/B](20260804-g3-p3e10-count-scale-ab.md)：
训练从1760扩到3316图后，h64 confirm从81/256、MAE1.56提高到108/256、1.27；h32为109/256、
1.16但route100 exact退到29%。两臂严格NO-GO；下一步保持h64，只加入Count有序距离损失。

当前已完成实验：[G3 P3-E9 DeepSets Count Train1760](20260804-g3-p3e9-deepsets-count.md)：
独立集合统计Count将confirm从12/256、MAE8.65提高到81/256、1.56，route100从3/100提高到
34/100；但train85.2%对confirm31.6%，严格NO-GO。下一步扩molecule并缩小Count头，输入特征不变。

当前已完成审计：[G3 P3-E8 Predicted Count Router](20260804-g3-p3e8-predicted-count-router.md)：
固定P3-E7路由从known Count切到P2-W Count后，confirm从111/256跌到4/256，route100从36/100
跌到0/100；Count exact仅9/256、3/100。Count已成为明确首要瓶颈，下一步重做DeepSets统计Count。

当前已完成审计：[G3 P3-E7 Base / Expert Router](20260804-g3-p3e7-base-expert-router-audit.md)：
512个molecule-disjoint样本上，固定单阈值路由在confirm将known-count exact从base 13/256提高到
111/256，98 rescue/0 break；冻结后在历史route100从4/100提高到36/100，32/0。路由方向GO，
下一步实现shadow组件并补predicted-count口径；不得表述为端到端结果。

当前已完成实验：[G3 P3-E6 Transactional Capacity Guard](20260804-g3-p3e6-transactional-capacity.md)：
实际更新后break即回滚，成功完整保持capacity双域32/32，并将general从9/192提高到37/192；
但只有9/30次更新可接受，最小1/128步长连续21次被拒，严格NO-GO。下一步冻结base并训练独立expert。

当前已完成实验：[G3 P3-E5 Capacity-Anchored A-GEM](20260804-g3-p3e5-capacity-anchored-agem.md)：
投影确实将冲突梯度点积压到约0，但AdamW有限步仍跨过离散exact边界；final easy/hard为29/32、
5/32，general为6079/6202、110/192，正式NO-GO。下一步改成实际更新后检查capacity并事务回滚。

当前已完成实验：[G3 P3-E4 Mixed Coverage Full-set](20260804-g3-p3e4-mixed-fullset.md)：
256图每epoch只做一次整组更新后，general达到6077/6202、111/192，hard较P3-E1从2/32改善到
8/32，但仍未保护capacity，正式NO-GO。下一步将capacity作为硬保护域做A-GEM，而不是继续调margin。

当前已完成实验：[G3 P3-E3 Capacity Coverage Full-set](20260804-g3-p3e3-capacity-fullset.md)：
相同30次曝光下，把64张capacity梯度累积后每epoch只更新一次，easy/hard完整保持967/967、
807/807和双域32/32；coverage margin仅62/64，严格Gate仍NO-GO。证明整组约束同一步可见是必要条件。

当前已完成实验：[G3 P3-E2 Margin A/B Capacity64](20260804-g3-p3e2-margin-ab-capacity64.md)：
不使用general数据时，旧固定margin仍把hard从32/32破坏到12/32，因果否决；coverage margin
在microbatch4下仍级联到2/32。下一步不是继续调margin，而是扩大单次更新的约束可见范围。

当前已完成实验：[G3 P3-E1 Latent Structured Pilot256](20260804-g3-p3e1-latent-structured-pilot256.md)：
latent valid-vs-best-invalid 将 general192 从5663/6202、9/192提高到6082/6202、155/192，
但 hard32 从807/807、32/32崩到704/807、2/32，正式NO-GO。E1b确认capacity起点的best-invalid
全部只错1个GT，真实局部margin已63/64满足，而固定mean-margin=1仅1/64满足；当前整图margin
尺度被否决。下一步做coverage-calibrated margin A/B，并单独使用molecule-disjoint能力样本。

当前已完成审计：[G3 P3-E0 Latent Solver 与梯度审计](20260804-g3-p3e0-latent-solver-gradient-audit.md)：
精确latent-positive solver在4,937图上最大仅32种归属、p95约1.6ms，正式GO。BF16/FP32下
P2-W query exact仅190/256但candidate-set exact为249/256，确认等价query身份不稳定；同时
梯度审计发现正确capacity梯度接近0、general梯度巨大，历史目标几乎全程强clip。下一步先补
等价类不变的best-invalid margin oracle；E0b随后通过穷举、重复性和延迟门，Count与结构彻底拆开。

当前已完成实验：[G3 P3-D Structured Matching Replay Train4937](20260804-g3-p3d-structured-matching-replay-train4937.md)：
loss-augmented joint matching将route known-count提高到目前最高34/100，但capacity仅16/32、2/32，
仍NO-GO。结构目标有效，错误在把一个任意query-candidate排列当唯一正解；下一步改为GT-compatible
latent positive matching，保留排列等价性。

当前已完成实验：[G3 P3-C Fixed Assignment Replay Train4937](20260804-g3-p3c-fixed-assignment-replay-train4937.md)：
4,937张图固定assignment并每批25% capacity replay后，capacity从P3-B的5/32、0/32恢复到
22/32、10/32，但route predicted exact回到0/100，仍NO-GO。固定身份有效但factorized loss与
joint MAP不对齐；下一步改为loss-augmented固定基数结构化匹配目标，不再调replay比例。

当前已完成实验：[G3 P3-B Joint Atom Query Train4937](20260804-g3-p3b-joint-atom-query-train4937.md)：
全量解冻后route100从0提高到11/100 exact、known-count到30/100，但capacity64从双域32/32
崩到5/32、0/32。容量样本已在训练集，故不是普通遗忘，而是动态Hungarian身份与joint MAP
推理目标冲突。停止同协议扩量；下一步固定identity replay并直接对齐结构解码目标。

当前已完成实验：[G3 P3-A Count Head Train5k](20260804-g3-p3a-count-head-train5k.md)：
冻结P2-W表示只训8,513参数Count head，route100 exact从3/100升至8/100、MAE降至4.59，
但train5k自身也仅14.54% exact，正式NO-GO。下一步必须联合适配encoder/decoder与三类head。

当前已完成审计：[G3 P2-Y Joint Cardinality Decode Route100](20260804-g3-p2y-joint-cardinality-decode-route100.md)：
route100与train5k分子重叠0；候选oracle 3073/3080、98/100。P2-W joint在预测Count下
2399/3080、0/100，Count仅3/100；真实Count下也只有2827/3080、4/100。下一步先Count-only
因果隔离，再扩训query/pointer，禁止把两类断点混在一起解释。

当前已完成审计：[G3 P2-X Joint Cardinality Decode Capacity64](20260804-g3-p2x-joint-cardinality-decode-capacity64.md)：
不训练新模型，将query选择与唯一pointer分配合并为无超参固定基数MAP，P2-W easy/hard从
31/32、32/32提高到双域32/32，容量Gate正式GO。下一步进入未见样本泛化审计。

当前已完成实验：[G3 P2-W Fixed Identity Pointer Utility Step4800](20260804-g3-p2w-fixed-identity-pointer-utility-step4800.md)：
完整恢复P2-V optimizer续到固定4,800步，hard达到32/32，easy仍31/32，独立top-N解码NO-GO。
唯一残差来自先选query、后配pointer的决策割裂；无GT联合固定基数MAP解码预验证双域32/32。

当前已完成实验：[G3 P2-V Fixed Identity Pointer Utility Capacity64](20260804-g3-p2v-fixed-identity-pointer-utility-capacity64.md)：
P2-Q改用正确固定identity后，10维pointer utility将easy/hard提高到31/32、28/32，机制GO但
严格容量NO-GO。曲线在1,600步仍改善；下一步完整恢复optimizer并预先固定4,800步终点。

当前已完成实验：[G3 P2-U Fixed Identity Listwise MLP Capacity64](20260804-g3-p2u-fixed-identity-listwise-mlp-capacity64.md)：
正确identity上组合8,577参数MLP与平滑listwise后仅28/32、26/32，NO-GO。停止调纯query-hidden
head；下一步用正确identity重做P2-Q的10维pointer utility实验。

当前已完成实验：[G3 P2-T Fixed Identity Existence MLP Capacity64](20260804-g3-p2t-fixed-identity-existence-mlp-capacity64.md)：
正确identity下只把65参数线性head换成8,577参数MLP，easy/hard提高到30/32、28/32但仍NO-GO；
固定identity exact从step800后基本平台。下一步组合MLP与平滑listwise排序，不追加相同BCE epoch。

当前已完成实验：[G3 P2-S Fixed Identity Listwise Existence Capacity64](20260804-g3-p2s-fixed-identity-listwise-existence-capacity64.md)：
相同固定identity与P2-J初始化下加入平滑全正负对listwise排序，结果仍为28/32、26/32，NO-GO。
线性head的两类目标均未过容量门；下一步只替换为8,577参数MLP，正确隔离非线性容量。

当前已完成实验：[G3 P2-R P2-J Identity Existence Capacity64](20260804-g3-p2r-p2j-identity-existence-capacity64.md)：
固定P2-J原始existence-aware identity后，65参数head的weighted BCE得到easy/hard 29/32、27/32，
仍为NO-GO；但强制cache identity解码两域均32/32，已把剩余断点严格定位为existence整组排序。
下一步在同一缓存、同一初始化上测试平滑全正负对pairwise排序损失。

当前已完成实验：[G3 P2-Q Pointer Utility Existence Capacity64](20260804-g3-p2q-pointer-utility-existence-capacity64.md)：
显式加入10个pointer utility后仅22/32、23/32。复核确认P2-M至Q删除existence tie-break后，
正query identity与P2-J原身份只重合90.49%/89.47%，导致校准head被迫学习另一套等价排列。
下一步冻结P2-J existence-aware assignment做Action Cache，再执行正确的固定identity M-step。

当前已完成实验：[G3 P2-P Frozen Pointer Existence MLP Capacity64](20260804-g3-p2p-frozen-pointer-existence-mlp-capacity64.md)：
冻结同一P2-J表示，将65参数线性head换成8,577参数MLP，BCE降到0.125但easy/hard仅
952/967、20/32和795/807、22/32；P2-Q复核后归因修正为pointer-only identity协议NO-GO，
不能据此判定P2-J原identity的hidden容量。

当前已完成实验：[G3 P2-O Frozen Pointer Existence BCE Capacity64](20260804-g3-p2o-frozen-pointer-existence-bce-capacity64.md)：
冻结P2-J全部表示、pointer-only固定标签，仅训练65参数线性head和weighted BCE，final为
960/967、26/32与795/807、24/32，略低于step0。该严格隔离确认线性head容量NO-GO；下一步
只增加existence MLP容量；P2-P随后证明纯query-hidden非线性容量也未过门。

当前已完成实验：[G3 P2-N Null Pointer Query Set Capacity64](20260804-g3-p2n-null-pointer-query-set-capacity64.md)：
用candidate-or-NULL统一pointer与existence后，easy/hard仅894/967、0/32和756/807、6/32，NO-GO。
同时复核发现P2-M把冻结head与极值boundary一起改变，不能归因线性容量；下一步补做P2-J冻结表示、
65参数线性head、pointer固定标签、weighted-BCE-only的严格隔离；P2-O随后确认线性容量未过门。

当前已完成实验：[G3 P2-M Frozen Pointer Existence Head Capacity64](20260804-g3-p2m-frozen-pointer-existence-head-capacity64.md)：
加载P2-J并冻结291,329参数，仅训练65参数existence head；pointer-only固定标签下仍退到
803/967、0/32和637/807、0/32；复核后只判“冻结head + 极值boundary”NO-GO，不能单独归因
线性容量。P2-N测试NULL语言后，下一步补weighted-BCE-only隔离。

当前已完成实验：[G3 P2-L Existence Boundary Query Set Capacity64](20260804-g3-p2l-existence-boundary-query-set-capacity64.md)：
在P2-J同初始化上加入最低正query对最高负query的极值margin，boundary全程0/4 batch exact，
easy/hard崩到603/967、0/32和456/807、0/32。动态身份下极值边界明确NO-GO；下一步冻结
pointer/query，只训练65参数existence head；P2-M随后证明独立线性head本身也不够。

当前已完成实验：[G3 P2-K Pointer-Matched Query Set Step8000](20260804-g3-p2k-pointer-matched-query-set-step8000.md)：
从P2-J完整恢复optimizer续训到8,000步，final easy/hard为965/967、30/32和801/807、27/32；
中间最好观察也仅30/32、29/32且后程振荡。严格NO-GO并停止追加epoch，下一步从相同随机初始化
重开，在object BCE外加入整图existence正负边界损失；P2-L随后证明该动态极值目标会塌缩。

当前已完成实验：[G3 P2-J Pointer-Matched Query Set Capacity64](20260804-g3-p2j-pointer-matched-query-set-capacity64.md)：
只把query-to-GT代价换成兼容候选pointer mass，easy/hard从P2-I的12/32、5/32大幅提高到
28/32、26/32；pointer-linked query两域均已32/32。4,800步严格NO-GO，但step3200→4800
持续单调提高；P2-K固定optimizer续训到8,000步后仍未过门，转existence boundary目标。

当前已完成实验：[G3 P2-I Unique Pointer Set Capacity64](20260804-g3-p2i-unique-pointer-set-capacity64.md)：
相同初始化下将pointer监督改成训练期全局唯一匹配，easy/hard为943/967、12/32和768/807、5/32，
仍NO-GO；但GT-linked query pointer达到967/967、32/32和802/807、28/32。existence top-N只与
GT-linked query重合92.35%/93.31%；P2-J随后改用pointer质量做query-to-GT matching。

当前已完成实验：[G3 P2-H Atom Query Set Capacity64](20260804-g3-p2h-atom-query-set-capacity64.md)：
291,394参数query集合在easy/hard各32图上将Count做到64/64，但known-count整组仅9/32、8/32。
训练时set-valued pointer允许多个query偏好同一peak，推理期唯一assignment出现148/102次独立
pointer冲突。Count机制GO、pointer容量NO-GO；P2-I随后把唯一pointer matching移入训练目标。

当前已完成审计：[G3 P2-G 渲染可恢复性审计](20260804-g3-p2g-render-recoverability-audit.md)：
基于真实`layers.visible_text`将route100的7张粗几何标记收敛为2张困难图、1张severe；冻结P2-F
重放显示eligible为3003/3003、98/98，唯一severe为47/53且top1024仍只49/53。P2-F正式NO-GO
不变，但最后残差已隔离到不可安全监督的stress渲染；P2-H随后进入未知Count下的Atom整组选择。

当前已完成实验：[G3 P2-F Stride-1 Center Field Train25k](20260804-g3-p2f-stride1-center-field-train25k.md)：
保持6250 updates/25000 exposures不变，将5k×5轮改为25k唯一图×1轮；route从3073/3080、
98/100小幅提高到3074/3080、99/100，但未达到预注册3075门，正式NO-GO。数据扩量仍有边际
收益，但不支持直接上100k；P2-G随后确认剩余失败属于severe stress渲染。

当前已完成实验：[G3 P2-E Stride-1 Top128 Selector Train5k](20260804-g3-p2e-stride1-top128-selector-train5k.md)：
冻结P2-A候选后，selector在train5k从152423/155216、3344/5000提高到152789/155216、
3492/5000，但route从3022/3080、63/100退化到3018/3080、61/100；全部中间checkpoint
均未同时改善matched与exact，route因果NO-GO，停止该动态硬匹配boundary loss。

当前已完成实验：[G3 P2-A Stride-1 Center Field Train5k](20260804-g3-p2a-stride1-center-field-train5k.md)：
route top128从2912/3080、26/100提高到3073/3080、98/100，因果GO。总预算128重新审计后，
纯stride-1 top128上限最高；唯一剩余route失败是`crossed_bonds` stress样本的5个不可达密集点。

当前已完成实验：[G3 P1-Z Stride-2 Center Field Train5k](20260804-g3-p1z-stride2-center-field-train5k.md)：
route top128从2989/3080、49/100提高到3071/3080、96/100，因果GO，证明capacity64中心场的
主要限制确实是训练分布太小。

当前已完成审计：[G3 P1-Y Train5k Candidate Reachability](20260804-g3-p1y-train5k-candidate-reachability.md)：
capacity64中心场的固定96+32候选在train5k仅3346/5000 exact、route仅69/100；即使每尺度
top1024仍不全覆盖，因此先训练中心场而非扩大selector token。

当前已完成实验：[G3 P1-X Multiscale Peak Selector Capacity64](20260804-g3-p1x-multiscale-peak-selector-capacity64.md)：
固定`stride2 top96 + stride1 top32`候选，用105,089参数集合selector和动态最大基数一对一监督，
overfit32从1199/1424、0/32提高到1424/1424、32/32，adjacent32从1225/1473、0/32提高到
1473/1473、32/32；独立确定性重放逐位一致，双域容量GO。下一步固定协议进入P1-Y train5k，
容量结果不得表述为公开分布泛化。

当前已完成审计：[G3 P1-W Multiscale Peak Rank Audit](20260804-g3-p1w-multiscale-peak-rank-audit.md)：
冻结P1-U/P1-V后，首个双域100%固定协议是`stride2 top96 + stride1 top32`：只用128候选即达到
overfit 1424/1424、32/32和adjacent 1473/1473、32/32。top128内同GT等价峰很多，下一P1-X
selector必须动态一对一集合匹配，禁止固定candidate身份监督和半径聚类。

当前已完成实验：[G3 P1-V Continuous Stride-1 Capacity64](20260804-g3-p1v-continuous-stride1-capacity64.md)：
连续768 feature使adjacent32 top128达到1473/1473、32/32，7个碰撞对全部由不同峰命中，证明
stride-1表示机制GO；但known仅1447/1473、18/32，原overfit32也回退到1399/1424、22/32，
双域容量NO-GO。下一步冻结P1-U/P1-V做多尺度峰rank与最小充分K审计，不追加相同训练。

当前已完成实验：[G3 P1-U Adjacent Atom Capacity64](20260804-g3-p1u-adjacent-atom-capacity64.md)：
从P1-T相同初始化交替训练原32图和近邻专项32图4,800步。原域保持1424/1424、32/32；专项域
known为1461/1473、20/32，top128为1468/1473、27/32，双域容量`NO-GO`。target审计确认384
网格仍有3图各丢1个Atom，而768网格零碰撞；下一步只测试连续stride-1中心场，不追加同一训练。

当前已完成数据审计：[G3 Adjacent Atom Capacity32 Manifest](20260803-g3-adjacent-atom-capacity32-manifest.md)：
从固定train5k选出32张近邻Atom困难图，包含全部7张same-stride4-cell碰撞图；距离范围
`0.75px..6.713px`，manifest SHA为`6e4eca...7442f`。P1-T第二阶段将从相同初始化重开4,800步，
要求原32图与专项32图同时100%，未过不进入5k。

当前已完成实验：[G3 P1-T Continuous Stride-2 Center Field Overfit32](20260803-g3-p1t-continuous-stride2-center-field-overfit32.md)：
PixelShuffle先生成连续32通道384² feature，再统一卷积出center/offset；仅联合解冻最后一个
stride-4视觉block，共92,035参数。严格容量达到known与top128均`1424/1424、32/32`，四种布局
全部100%，第一阶段`容量GO`。尚不能进5k；下一关是原32图加相邻Atom专项32图的双域100%门。

当前已完成实验：[G3 P1-S Phase Residual Center Field Overfit32](20260803-g3-p1s-phase-residual-center-field-overfit32.md)：
保留P1-A双线性384强base，只学习零初始化四相位残差；top128为1420/1424、29/32，known
1369/1424、15/32，仍产生约3.18万峰/图。容量`NO-GO`，确认phase不能直接作为最终logit；
下一步先生成连续384 feature并统一卷积输出中心场，同时解冻最后一个stride-4视觉block。

当前已完成实验：[G3 P1-R Virtual Stride-2 Center Field Overfit32](20260803-g3-p1r-virtual-stride2-center-field-overfit32.md)：
7,628参数PixelShuffle头从冻结stroke feature直接预测四个固定phase的384²中心场。top128仅
1394/1424、21/32，known为1222/1424、3/32，并产生约3.2万峰/图；容量`NO-GO`，不进入5k。
该结果定位到随机新头丢失P1-A强排序基线，下一步只测试P1-A base加零初始化四相位残差。

当前已完成实验：[G3 P1-Q High Resolution Center Field Overfit32](20260803-g3-p1q-high-resolution-center-field-overfit32.md)：
将中心场升到384²后top128为1422/1424、30/32，但known exact回退到19/32；双线性放大冻结
192特征没有创造高分辨率信息，容量`NO-GO`。下一步必须读取stride-2视觉特征或解冻最后一个
高分辨率block做可学习上采样，不再扩大同一adapter。

当前已完成实验：[G3 P1-P Local Residual Center Field Overfit32](20260803-g3-p1p-local-residual-center-field-overfit32.md)：
8,643参数零初始化局部残差加peak margin，将top128从1415/1424、28/32提高到
1421/1424、30/32，但known exact为25/32，严格容量`NO-GO`。剩余失败来自192网格对角相邻
真实Atom，regional maxima表示本身无法同时输出两个峰，因此停止加宽/加epoch。

当前已完成实验：[G3 P1-O Atom Center Field Overfit32](20260803-g3-p1o-atom-center-field-overfit32.md)：
冻结P1-A hidden只训练771参数线性center/offset，fixed-final known为1409/1424、26/32，
top128为1415/1424、28/32；exact相对step0不变且峰数增加，容量`NO-GO`，未进入train5k。

当前已完成审计：[G3 P1-N Regional Maxima零训练审计](20260803-g3-p1n-regional-maxima-zero-train-audit.md)：
不再使用`3px/3.5px/4px`候选聚类，将P1-A三槽压成连续cell score后用完整plateau regional
maxima直接生成唯一对象。train5k known exact `64.22%→87.94%`、recall
`98.176%→99.490%`，top-128可达exact 97.48%；route100历史诊断`53%→79%`且0 break。
正式`mechanism_go_requires_fresh_confirmation`，停止swap verifier，下一步训练轻量单通道
Atom center field；route100已消费，真正替换仍需全新fresh确认。

当前已完成审计：[G3 P1-M 一跳替换边界可达性](20260803-g3-p1m-one-move-boundary-reachability-audit.md)：
top-8 swap覆盖train一跳图95.27%，但route仅77.27%未过80%门；top-16覆盖train
775/783=98.98%、route 20/22=90.91%，正式选择`top16x16_action_candidates`。下一步最多
生成256个swap/图，以执行后整图exact为标签，并要求当前exact图零有害替换。

当前已完成审计：[G3 P1-L 最小Component替换审计](20260803-g3-p1l-minimum-component-replacement-audit.md)：
P1-K train的1,081张oracle可达失败图中，783张只需替换一个component，占72.43%；route为
22/32=68.75%，单/双替换累计90.63%。16张train图因representative覆盖不可达。按预注册选择
`one_move_action_verifier`；下一步先审计正swap在top-K分数边界的覆盖，再决定动作枚举范围。

当前已完成实验：[G3 P1-K Component Set Ranker Train5k](20260803-g3-p1k-component-set-ranker-train5k.md)：
60,609参数masked component ranker完成25,000 exposures并通过独立全量重放。train known
exact `64.22%→78.06%`、recall `98.176%→98.945%`；route exact `53→68/100`、recall
`97.461%→98.405%`。因果门全过，但train recall距99%能力门差0.055pp，正式
`protocol passed / causal GO / capability NO-GO`。不追加同一epoch，下一步审计失败图的最小
component替换数。

当前已完成审计：[G3 P1-J Component Representative审计](20260803-g3-p1j-component-representative-audit.md)：
四种3px component单坐标方案全部过门，按train reachability选择base-softmax，train
recall/exact 99.990%/99.68%，route 100%/100%。零训练component base top-M已将train几何
exact `24.18%→64.22%`、route `22%→53%`，Atom recall升至98.18%/97.46%。这直接定位
重复candidate竞争为主要瓶颈；下一步训练component-level presence ranker。

当前已完成审计：[G3 P1-I Target-free候选聚合可达性](20260803-g3-p1i-target-free-candidate-clustering-audit.md)：
3px预测anchor连通分量将每图候选`128→67.20`，多component GT `89.13%→20.01%`，同时
train一一覆盖recall 99.992%、exact 99.74%，route为100%/100%；3px与4px过门，按协议选择
更保守的3px，正式`target_free_radius_clustering_go`。下一步先审计component representative
坐标，再训练component-level ranker。

当前已完成审计：[G3 P1-H 动态指派稳定性审计](20260803-g3-p1h-dynamic-assignment-stability-audit.md)：
零训练同时重放P1-G五个epoch checkpoint；最后两轮仅666/154,911个Atom更换candidate代表，
turnover 0.43%，五轮始终稳定Atom达98.29%。冻结几何exact从20.68%单调升至24.18%，正式
`late_assignment_instability_not_primary`。否定后期硬匹配振荡主因，下一步审计target-free
几何候选聚合能否减少重复身份且不合并真实Atom。

当前已完成实验：[G3 P1-G 等价集合动态匹配 Train5k](20260803-g3-p1g-equivalent-set-matching-train5k.md)：
保持P1-D的60,545参数ranker与25,000 exposures，只将固定candidate身份监督改为6px组内最大
基数动态一一匹配。独立重放通过；train几何exact `18.68%→24.18%`、Atom recall
`94.26%→95.35%`，route exact `17→22/100`、known F1 `93.31%→94.22%`。跨域收益真实但
低于预注册+10pp因果门，正式`protocol passed / causal NO-GO / capability NO-GO`；停止追加
相同loss/epoch，下一步零训练区分动态assignment稳定性与target-free候选去重可达性。

当前已完成审计：[G3 P1-F 等价候选监督审计](20260803-g3-p1f-equivalent-candidate-supervision-audit.md)：
零训练重放四个final checkpoint后确认固定身份是主要瓶颈。train5k中89.13%的representable
GT在6px内有多个top-128候选，平均3.19个；允许几何等价候选置换后，四臂whole-set exact
相对固定candidate identity提高12.54..13.96pp，route100也提高11..13pp。正式判定
`fixed_identity_major_bottleneck`；下一步改为组内动态一一匹配监督，禁止把同组全部标正。

当前已完成实验：[G3 P1-E Aligned Stroke Feature Sufficiency Train5k](20260803-g3-p1e-aligned-stroke-feature-sufficiency-train5k.md)：
同参数CAP/SHAM/ALIGN三臂各完成6,250步，24项协议检查及独立全量重放通过，但Train-Fit和
route Gate全部失败。train boundary为5.16%/5.58%/5.88%，ALIGN只比SHAM高0.30pp；同权重
patch打乱只降0.38pp。CAP将ranker扩至243万参数反而降低recall，route F1仍仅92.01%..93.28%。
正式`protocol passed / capability failed / NO-GO / no_go_indeterminate`；停止加ranker容量和
stride-4 patch，下一步审计固定hybrid身份是否错误惩罚了同一GT附近的等价候选。

当前已完成实验：[G3 P1-D Boundary-Band Set Ranker Train5k](20260803-g3-p1d-boundary-band-set-ranker-train5k.md)：
唯一正式训练完成5 epochs/6,250步；独立审计重算route100与完整train5k后确认
`protocol passed / capability failed / NO-GO`。route known F1仅93.31%、matched 2874/3080、
whole 17/100，低于P1-C；train5k的hybrid top-M recall为88.68%、whole-set 5.22%、strict
boundary exact仅5.20%。因此easy-negative loss稀释假设被否定，停止换ranking loss或追加
epoch；下一步先做同冻结token上的ranker容量A/B，再决定是否增加视觉特征。confirm400与
外部数据继续封存。

当前已完成实验：[G3 P1-C Top128 Global Set Ranker Train5k](20260802-g3-p1c-top128-global-set-ranker-train5k.md)：
固定top-128上的60,545参数单层attention完成5 epochs/6,250步；五轮candidate/positive SHA
完全一致，train标签覆盖99.80%。known F1 93.18%→93.77%、whole 14→17、canonical identity
83.80%→85.13%，独立审计纠正一次autocast bitwise误报后结论为
`protocol passed / capability failed / NO-GO`。attention有信号但当前全pair平均loss被easy pair
稀释；下一步做固定8×8 boundary-band loss-only实验，不加层或epoch。

当前已完成实验：[G3 P1-B Frozen Presence Ranker Train5k](20260802-g3-p1b-frozen-presence-ranker-train5k.md)：
冻结P1-A整个Atom head和vision，仅训练16,643参数的pointwise presence residual。唯一正式训练
完成5 epochs/6,250步，17项协议检查全部通过，但known-count F1仅93.18%→93.60%、whole-set
14→16/100、canonical identity只增加13/3080，能力正式`NO-GO`。top-128 geometry coverage
仍为3072/3080、fixed-positive localization仍为3080/3080；residual绝对值已达18.25仍无显著
迁移，停止追加局部MLP或epoch。下一步另立top-128全局set ranker协议，confirm400继续封存。

当前已完成实验：[G3 P1-A Canonical Atom Selector Train5k](20260802-g3-p1a-canonical-atom-selector-train5k.md)：
固定train5k训练5 epochs/6,250 updates后，route-select100的Atom F1从76.65%升至90.39%，
count MAE从12.36降至3.83，但count exact仅13/100、whole-set仅2/100，协议通过、能力正式
`NO-GO`。只读GT-count top-K为93.18% F1和14/100 whole-set；all-dense top-128 coverage
99.74%，canonical fixed-positive定位100%。结论是候选容量与定位基本解决，presence/ranking
是第一瓶颈，count是第二瓶颈。下一步冻结P1-A整个Atom head做P1-B selector-only；禁止追加
P1-A epoch，confirm400继续封存。

当前已完成审计：[G3 P0-M Frozen Route Select 100](20260802-g3-p0m-frozen-route-select-100.md)：
固定route-select100上0 step、0参数运行，协议全部通过，但能力正式 `NO-GO`，confirm400继续
封存。GT Atom oracle presence F1为98.42%，而positive type仅81.51%；same-slot主路Atom
count exact仅 `5/100`、recall 87.92%、Bond representable 79.18%、GT-count presence F1
59.77%、representable type 80.65%。这证明presence已有迁移信号，首要扩量对象是canonical
Atom selector，其次是semantic type；下一阶段按顺序在train_5k分开训练，不做joint。

当前已完成实验：[G3 P0-L Native Atom Feature Adapter overfit32](20260802-g3-p0l-native-feature-adapter-overfit32.md)：
P0-L 冻结 P0-I/P0-G/P0-K，只训练147,840参数的共享线性 residual adapter，以 detached
same-slot P0-G feature做SmoothL1-only蒸馏。final count/skeleton恢复 `32/32`，但feature
error只下降70.37%，未达到90% Gate；GT/deploy typed graph均仅 `7/32`。same-slot与oracle
controls全程严格 `32/32`，独立审计确认协议有效，能力正式 `NO-GO`。停止在同一32图上扩大
adapter；当前结构保留 image-derived P0-G same-slot semantic feature，下一步做未见train-split
的零训练冻结 route audit。

当前已完成实验：[G3 P0-K Predicted-Anchor Relation Alignment](20260802-g3-p0k-predicted-anchor-relation-alignment.md)：
P0-J 的 same-slot feature replay只有 `26/32`，只换 GT anchor立即恢复 `32/32`。P0-K
冻结 vision、P0-I Atom、P0-G Atom feature与count，只训练236万参数 pair head，以真实
P0-I predicted anchor/ribbon固定训练400 steps。step100达到 `31/32`，fixed-final step400
的 predicted GT/deploy skeleton与typed graph、GT Atom oracle replay均为 `32/32`，独立
checkpoint审计通过，正式 `GO`。这证明 predicted-anchor geometry shift是 P0-J presence
错误的主因；结论仍限于 overfit32容量，下一步单独处理 P0-I native feature接口。

当前已完成审计：[G3 P0-J Frozen Relation Composition](20260802-g3-p0j-frozen-relation-composition.md)：
冻结 P0-I Atom 与 P0-G relation，比较 P0-I native feature 和在同一预测 slot 上重取的
P0-G feature。GT 只在 forward 后做 6px 一对一评分映射，不进入 pair universe 或 top-K。
P0-G oracle replay 仍为 `32/32`；native feature typed graph 仅 `4/32`，slot replay 修复
type/count 后 typed graph 为 `26/32`，仍有 10 条 presence top-K 错误，正式 `NO-GO`。
冻结 2x2 残差消融显示：替换 score仍为 `26/32`，只替换 GT anchor 即恢复 `32/32`，
主因确定为 predicted-anchor geometry shift。下一步 P0-K 只对齐 P0-G pair head。

当前已完成实验：[G3 P0-I Canonical Cell Slot overfit32](20260802-g3-p0i-canonical-cell-slot-overfit32.md)：
P0-H 的 all-dense/top-128 与 positive localization 均为 `32/32`，但 GT-count top-K 只有
`1369/1424`、whole-set `5/32`，supervision identity first-to-final churn 为 `70.22%`。
P0-I 唯一改动是将同一 cell 内 GT 按 normalized `(x,y)` 升序固定映射到 slot
`0..n-1`，完全相同坐标直接拒绝；当前 `occupancy=1` 为 `1424`、collision 为 `0`。
其余严格复用 P0-H 的同初始化、Atom-only、from-scratch、2,400-step final-only 协议。
fixed-final deploy count、top-K identity、GT-count Atom set 全部达到 `32/32`，canonical
identity churn=`0%`，机器 Gate 全部通过，正式 `GO`。这证明动态 cell-slot 监督身份是
P0-H 的主要断点；但 occupancy=2/3、泛化和 P0-G feature interface 仍未验证。

当前已完成实验：[G3 P0-H Atom-only overfit32](20260802-g3-p0h-atom-only-overfit32.md)：
从与原 P0 完全相同的 Atom 初始化、seed、vision、32 图和 2,400-step 预算出发，只训练
`SpatialAtomSlotHead` 的 heatmap、anchor 与 count objective，禁止 Bond/AtomPair 梯度。
fixed-final deploy count 达到 `31/32`，但 GT-count Atom whole-set 只有 `5/32`，正式
`NO-GO`。all-dense、top-128 和 positive localization 均为 `1424/1424`，而 GT-count
top-K 只有 `1369/1424`，supervision identity churn 为 `70.22%`。这排除了候选池容量与
定位是首要瓶颈，下一轮只处理动态 slot identity 和整组 presence 选择。

当前结构实验：[G3 层级 Atom→Bond 集合图 P0 overfit32](20260801-g3-p0-overfit32.md)：
P0-B..P0-F 均为 `NO-GO`；其中 P0-F learned query 的 first-to-final identity churn 为
`99.14%`。P0-G 改用全部无序 GT AtomPair 作为稳定 Bond identity 后，fixed-final 的
GT/deploy skeleton 与 typed graph 均达到 `32/32`，pair identity churn=`0%`，正式通过
容量 Gate。这只证明给定完美 Atom set 时 Bond relation 可学；预测 Atom set 与全 pair
`O(V²)` 部署成本仍未解决。

最新因果诊断：[ChemTraverse GT Count Oracle](20260801-chemtraverse-gt-count-oracle.md)：
给定真实原子/键数量后，route100 的 count-pair 为 `68/100`，但 completion 仍为 `68/100`、
Skeleton 仍为 `0/100`。CountHead-conditioned AR 三项门槛全部失败；下一步转 decision-only
grammar loss 与 G3 Atom→Bond 集合解码。

当前已完成：[v0.10.21 Invalid-only OCR Fallback](20260731-v0.10.21-invalid-only-ocr-fallback-audit.md)
（真实 USPTO 1k：valid 704→802、chemical exact 160→187，全部 0 break；shadow GO）。

当前主应用 benchmark：MR-TADF v1 完整 1,100 张。v0.10.21 的 SDF valid 65.91%、
Skeleton exact 48.09%、Chemical exact 24.00%；count-pair 正确条件下 Skeleton exact
97.96%，下一瓶颈为整图对象计数与去重。

最新表示消融：[v0.9.41 文字框内部 Bond Veto Pilot](20260731-v0.9.41-text-interior-bond-veto-pilot.md)
（USPTO 前 100 只触发 1 张，全部指标 `0 rescue / 0 break`；NO-GO，不建立新版本）。

| 编号 | 阶段 | 主要结论 | 状态 |
|---|---|---|---|
| 001 | [100 样本视觉接地课程](001-grounding-100.md) | 证明图片到结构的训练和 SDF 往返链路成立 | 完成 |
| 002 | [ChemMolSeq v0.3 1k 课程](002-chemmolseq-v03-1k.md) | 缩短序列并改用图片空间原子顺序 | 历史基线 |
| 003 | [ChemSVG v0.4 初始训练](003-chemsvg-v04-initial.md) | 用视觉端点直接表达键，不依赖 atom ID | 当前协议基础 |
| 004 | [v0.5 几何与 OCR 解耦](004-v05-ocr-decoupling.md) | OCR 输出开放 UTF-8 字符串，局部预训练必须冻结视觉层 | 完成 |
| 005 | [v0.5 顶点、连线与墨线修复](005-v05-visual-relation-repair.md) | 全点对不设键长上限，repair 直接检查真实墨线 | 完成 |
| 006 | [v0.6 并行 CPU 推理](006-v06-parallel-cpu.md) | 删除部署 decoder，推理参数降至约 15.18M | 完成 |
| 007 | [v0.7.2 因子化 Edge/Bond head](007-v072-factorized-relation.md) | 局部指标提高但端到端无收益 | 否决 |
| 008 | [v0.7.2 平衡字符替换 OCR](008-v072-replacement-ocr.md) | 高比例扩展字符会破坏原分布，平衡 3k 有效 | 完成 |
| 009 | [v0.7.3 Pixel OCR](009-v073-pixel-ocr.md) | 原图 crop 解决 stride-8 字符细节损失 | 候选 |
| 010 | [v0.7.5 ChemVision 多尺度 OCR](010-v075-chemvision-multiscale-ocr.md) | 12/20/32px 融合将 MR-TADF 完全一致提高到 67.45% | 完成 |
| 011 | [v0.7.6 顶点召回与定位 pilot](20260715-v0.7.6-vertex-recall-pilot.md) | 只训练 92,548 个 vertex 参数，完全一致提高到 77.55% | 完成 |
| 012 | [v0.7.7 字体扩展 OCR pilot](20260715-v0.7.7-font-expanded-ocr.md) | 字体方向有效，但全头微调遗忘旧分布；保留数据，否决 checkpoint | NO-GO |
| 013 | [v0.7.8 OCR 残差适配器](20260715-v0.7.8-ocr-residual-adapter.md) | 冻结旧权重，只增 29,044 参数；USPTO 有效 88→167，MR 完全一致 77.55%→79.55% | 完成 |
| 014 | [v0.7.9 全点对骨架](20260715-v0.7.9-skeleton-all-pairs.md) | 全点对 relation + 概率约束 repair；USPTO 有效 167→177，MR 完全一致 79.55%→79.73% | 完成 |
| 015 | [v0.7.10 概率桥接修复](20260715-v0.7.10-probability-bridge.md) | 不训练新权重；保留全点对概率补桥，USPTO 有效 177→192，MR 完全一致不变 | 当前最佳 |
| 016 | [v0.7.11 MolGrapher 结构头插值](20260715-v0.7.11-molgrapher-interpolation.md) | 新域 Oracle 提升，但三档均未通过旧域联合门控 | NO-GO |
| 017 | [v0.7.11 结构残差适配器](20260715-v0.7.11-structure-residual-adapters.md) | Oracle 连线提升，detected 整图无收益，瓶颈仍是顶点集合 | NO-GO |
| 018 | [v0.7.11 Teacher Consistency](20260715-v0.7.11-teacher-consistency.md) | 新域局部指标恢复，但共享头仍严重遗忘未覆盖旧域 | NO-GO |
| 019 | [v0.7.11 保守结构 MoE](20260715-v0.7.11-conservative-structure-moe.md) | gate 只实现旧域回退，新域 detected exact 1.3% | NO-GO |
| 020 | [v0.7.12 集合式顶点](20260716-v0.7.12-atom-query-vertex.md) | Atom Query 计数 + dense 吸附在 100 图达到 95% vertex exact | 顶点 GO |
| 021 | [v0.7.13 检测顶点关系](20260716-v0.7.13-detected-relation-hard-negatives.md) | 困难负边 + pair encoder 在 100 图达到 89% skeleton exact | 过拟合 GO |
| 022 | [v0.7.14 MolGrapher 9k 顶点扩量](20260716-v0.7.14-molgrapher-9k-vertex-scale.md) | dense 6px F1 89.48%，但整图 vertex exact 最高 6.3% | 局部 GO / 整图 NO-GO |
| 023 | [v0.7.15 顶点主干适配与峰值抑制](20260716-v0.7.15-vertex-backbone-and-nms.md) | 解冻一个 stride-4 block 将 exact 提至 9.5%；宽 NMS 揭示一对一候选竞争瓶颈，但固定尺度不可部署 | 权重 GO / 固定宽 NMS NO-GO |
| 024 | [v0.7.16 一对一顶点候选筛选](20260716-v0.7.16-one-to-one-proposal-keep.md) | 56,673 参数 keep head 将 exact 提至 22.0%，USPTO 有效 SDF 63→107 | proposal GO / shadow 不替换稳定版 |
| 025 | [v0.7.17 稳定基线尺度增强 Proposal](20260716-v0.7.17-stable-scale-proposal.md) | 稳定 v0.7.10 上有效 SDF 192→226，但完全一致 22→19，过删仍增加断图 | 组件 GO / 默认 NO-GO |
| 026 | [v0.7.18 联合 Node-Edge Proposal Graph](20260716-v0.7.18-joint-proposal-graph.md) | 原图线带反向救点；USPTO 有效 241、完全一致 29，paired 净增 10 | 组件与 shadow GO / 默认待定 |
| 027 | [v0.7.19 Graph Edge Prior](20260716-v0.7.19-graph-edge-prior.md) | repair 末端仅改变 2/1000 样本，完全一致无提升 | NO-GO |
| 028 | [v0.7.20 跨字体全局 Proposal Graph](20260716-v0.7.20-style-global-proposal.md) | held-out 字体 exact 32.93%→36.75%，USPTO 241/29→258/30；固定阈值使 MR 完全一致降至 49% | 组件 GO / 默认 NO-GO |
| 029 | [v0.7.21 可训练视觉 Atom Query](20260716-v0.7.21-trainable-vision-atom-query.md) | 100 图 97%；独立 1k 的 6px F1 20.40%→40.14%，Count exact 23.9% | 坐标 GO / cardinality 待新结构 |
| 030 | [v0.7.22 全局 Cardinality Token](20260716-v0.7.22-global-cardinality-token.md) | token 联合训练 Count exact 21.5%；真实计数 Oracle 的 6px exact 仍为 0 | cardinality NO-GO / query 质量待提升 |
| 031 | [v0.7.23 纯坐标分配与独立排序](20260716-v0.7.23-coordinate-query-selection.md) | 修复 presence 自强化；部署型 6px F1 40.14%→54.79%，Oracle 65.39% | 分配/排序 GO / 计数待提升 |
| 032 | [v0.7.24 共享 Dense 顶点吸附](20260716-v0.7.24-shared-dense-query-snap.md) | 只增 92,678 参数；部署型 6px F1 54.79%→76.61%，跨字体 76.77% | 顶点精定位 GO / 端到端待 relation |
| 033 | [v0.7.25 共享顶点分布关系重训](20260716-v0.7.25-shared-detected-relation.md) | MolGrapher Edge F1 52.56%→62.70%，跨字体 53.99%→58.28%，假边显著减少 | relation GO / 稳定默认不替换 |
| 034 | [v0.7.26 Dense 候选与 Query 融合](20260716-v0.7.26-dense-candidate-query-fusion.md) | MolGrapher 顶点 F1 76.61%→90.02%，首次得到 5.7% skeleton exact；跨字体顶点 F1 92.12% | 顶点与骨架 GO / 待 relation 重训 |
| 035 | [v0.7.27 Grounded 全点对真实线条图](20260717-v0.7.27-grounded-visible-line-graph.md) | 100 图 Edge F1 8.57%→80.44%；两阶段 known-count vertex exact 12%→18%，但 set-count 仍失准 | Edge GO / 排序容量 GO / 计数 NO-GO |
| 036 | [v0.7.28 独立 Grounded Count 校准](20260717-v0.7.28-independent-grounded-count.md) | 完整排序节点/边特征在 100 图达到 100% Count accuracy，并追平 known-count vertex exact | Count 容量 GO / 待 9k graph |
| 037 | [v0.7.29 Whole-Set 整图精确目标](20260717-v0.7.29-whole-set-exact-objective.md) | 修复 TF32 距离标签；6px Oracle exact 57%，whole-set known-count 12%→45%，接 Count 后 41% | 整图目标 GO / 候选上限待提升 |
| 038 | [v0.7.30 高召回候选与坐标对齐视觉](20260717-v0.7.30-high-recall-candidate-visual.md) | 256 候选 Oracle exact 88.7%；3x3 对齐 patch 将 9k/1k F1 提至 84.72%、strict exact 提至 4.4% | 候选/对齐视觉 GO / 部署 NO-GO |
| 039 | [v0.7.31 候选身份极值排序](20260717-v0.7.31-extreme-ranking-no-go.md) | 极值损失把 6px 内等价候选误作负样本，20k 时 exact 4.4%→0.5% | NO-GO / 转向候选等价组 |
| 040 | [v0.7.32 文字对象与键线顶点 Oracle](20260717-v0.7.32-semantic-vertex-oracle.md) | 文字实例作为单一顶点后 F1 75.69%→85.19%、exact 0.2%→2.6%；bond-only 无额外收益 | 文字对象解耦 GO |
| 041 | [v0.7.33 独立文字对象中心分支](20260717-v0.7.33-text-object-center.md) | internal 500 文字 F1 94.30%、exact 55.60%；合并 F1 75.04%→83.14% | GO / 转向隐式顶点 |
| 042 | [v0.7.34 文字与隐式骨架双顶点分支](20260717-v0.7.34-dual-vertex-branches.md) | internal 500 双分支 F1 92.62%、exact 9.6%，原始 dense exact 仅 0.4% | 双分支 GO |
| 043 | [v0.7.35 Bond Object 自由查询 Pilot](20260717-v0.7.35-bond-object-pilot.md) | 100 图端点 recall 96.65%、exact 55%；9k 泛化端点 recall 仅 4.62% | 语义 GO / 自由查询 NO-GO |
| 044 | [v0.7.36 端点条件化 Bond Object](20260717-v0.7.36-node-conditioned-bond-object.md) | GT 端点 9k Edge F1 99.16%、skeleton exact 78.31%；预测端点条件 Edge recall 98.70% | 键对象 GO / 顶点仍为瓶颈 |
| 045 | [v0.7.37 渲染真值与候选等价组](20260717-v0.7.37-render-label-group-selector.md) | 角色严格 vertex exact 56.02%；接 Bond Object 后 strict skeleton exact 7.83%→41.37% | 顶点与骨架接入 GO / shadow |
| 046 | [v0.7.38 端点偏移稳健 Bond Object](20260717-v0.7.38-endpoint-jitter-bond.md) | 真实误差分布 jitter 将 strict skeleton exact 41.37%→51.00%，长键条件 recall 92.84%→96.08% | GO / shadow |
| 047 | [v0.7.39 部署边界与角色校准](20260717-v0.7.39-deployment-boundary-calibration.md) | 绝对边界将 role vertex exact 56.02%→59.64%、skeleton 51.00%→53.61%，但未过正式门槛 | 方向有效 / 默认 NO-GO |
| 048 | [v0.7.40 全候选邻接图去噪](20260717-v0.7.40-full-adjacency-graph-denoiser.md) | 100 图容量 89%，9k role exact 57.83%、skeleton 52.81%；强 Bond teacher 无收益 | 容量 GO / 默认 NO-GO |
| 049 | [v0.7.41 联合节点与边去噪](20260717-v0.7.41-joint-node-edge-denoising.md) | 联合 edge residual 将 role exact 提至 59.44%、skeleton 54.62%，但线内假点明显增加 | 方向有效 / 默认 NO-GO |
| 050 | [v0.7.42 端点射线终止与穿越证据](20260718-v0.7.42-endpoint-ray-evidence.md) | 3px 线内假点 162→107，但漏点 411→472，role exact 59.24% | 机制 GO / 默认 NO-GO |
| 051 | [v0.7.43 共线穿越节点上下文](20260718-v0.7.43-straight-through-context.md) | 角度摘要找回部分漏点，但 role exact/skeleton 仅追平 59.64%/54.62% | 容量 GO / 默认 NO-GO |
| 052 | [v0.7.44 候选二维 Patch 受控 A/B](20260718-v0.7.44-candidate-patch-ab.md) | 原图 patch 减少线内假点但增加漏点，2k exact 比 feature 基线低 1.61pp | 容量 GO / 正式 NO-GO |
| 053 | [v0.7.45 独立 PASS_THROUGH 语义上下文](20260718-v0.7.45-semantic-pass-through-context.md) | 解耦分类与 REAL 梯度后 100 图 exact 87%，未复现 88% 容量门槛 | 容量 NO-GO |
| 054 | [v0.7.46 候选覆盖阈值与容量网格](20260718-v0.7.46-candidate-coverage-grid.md) | 0.02/k384 仅找回 3 个顶点、无额外整图收益，点对成本 4.51 倍 | Oracle NO-GO |
| 055 | [v0.8.0 ChemScene Bond-First](20260718-v0.8.0-chemscene-bond-first.md) | 对象协议和 SDF 编译完成；3k 分片未知失败为 0 | 协议 GO |
| 056 | [v0.8.1 空间线对象 Bond-First](20260718-v0.8.1-spatial-bond-object.md) | 100 图 SDF exact 98%；9k object accuracy 67.55%，SDF exact 2.61% | 组件 GO / 部署 NO-GO |
| 057 | [v0.8.2 多尺度空间 BondObject](20260718-v0.8.2-multiscale-bond-object.md) | 长键和双键改善，object 70.66%，SDF exact 4.42% | 多尺度几何 GO |
| 058 | [v0.8.3 二维线带键型 Refiner](20260718-v0.8.3-line-band-type-refiner.md) | 局部 object 71.22%，但 SDF exact 回退到 2.61% | 局部机制 GO / 替换 NO-GO |
| 059 | [v0.8.4 Junction-Bond Assembler](20260718-v0.8.4-junction-bond-assembler.md) | pair exact 98.18%，有效率提高但端到端 exact 略降 | 组件 GO / 默认 NO-GO |
| 060 | [v0.8.5 ChemScene 图级修正](20260718-v0.8.5-chemscene-graph-corrector.md) | type-only exact 19.48%→19.68%，edge keep 严重过删 | type-only 小幅 GO |
| 061 | [v0.8.6 彩色修饰符 OCR](20260718-v0.8.6-color-modifier-ocr.md) | crop validation 85.62%→89.16%，真实彩色图未泛化 | 组件 GO / 接入 NO-GO |
| 062 | [v0.8.7 bbox CTC pilot](20260718-v0.8.7-bbox-ctc-pilot.md) | 10k crop 最佳 47.15%，从零视觉表征不足 | NO-GO |
| 063 | [v0.8.8 预训练 bbox CTC](20260718-v0.8.8-pretrained-bbox-ctc.md) | 10k crop 86.20%；预测框协议 64.12%，真实图仍有 NH 幻觉 | 组件 GO / 接入 NO-GO |
| 064 | [v0.8.9 非对称文字框](20260719-v0.8.9-asymmetric-text-bbox.md) | 部署 OCR 64.74%→66.58%，modifier 21.97%→27.84% | 组件 GO / 实验 profile GO |
| 065 | [v0.8.10 Oracle 文字层 A/B](20260719-v0.8.10-oracle-text-layer-ab.md) | balanced RGB macro 55.13%；mask+anchor 57.77%，未过 10pp 门槛 | 数据协议 GO / 分割主线暂缓 |
| 066 | [v0.8.11 Open Text CTC](20260720-v0.8.11-open-text-ctc.md) | 容量审计 100%；真实预测框 81.50%，modifier 67.49%，开放随机串 5.62% | 容量与真实框 GO / 接入 NO-GO |
| 067 | [v0.8.13 Bond-first 分层审计与集合选择](20260720-v0.8.13-bond-first-set-decomposition.md) | Candidate Oracle 88.76%；500 张容量拓扑 87.2%，9k 泛化 38.35% | 表示/容量 GO，冻结 selector NO-GO |
| 068 | [v0.8.14 冻结视觉 Bond Set Selector](20260720-v0.8.14-visual-bond-set-selector.md) | 中心 hidden 在 9k 将拓扑 38.35% 降到 34.34% | 冻结视觉 NO-GO，转 line-band |
| 069 | [v0.8.15 完整 Line-Band Bond Set Selector](20260720-v0.8.15-line-band-bond-set-selector.md) | line-band AUC 98.77%，但 learned count 拖累，9k 拓扑仅 33.73% | 特征有效 / count 解码 NO-GO |
| 070 | [v0.8.16 语义 Bond Endpoint 生成 Junction](20260720-v0.8.16-bond-endpoint-junction.md) | 结构拓扑 67.67%；正式端到端化学一致 12.45%，OCR 附着导致大量价态失败 | 结构 GO / 端到端 NO-GO |
| 071 | [v0.8.17 Bond Threshold 校准](20260720-v0.8.17-bond-threshold-calibration.md) | 0.20 将 USPTO 纯骨架 73.2% 提到 76.0%，但 SDF valid 降到 69.1% | 召回 GO / 固定阈值 NO-GO |
| 072 | [v0.8.18 Bond Oracle Rescue Ceiling](20260720-v0.8.18-bond-oracle-rescue-ceiling.md) | top-128 覆盖 88.76%，固定 endpoint clustering 后拓扑上限仅 75.30% | 候选 GO / 聚合待改 |
| 073 | [v0.8.19 Endpoint Same-Junction Grouper](20260720-v0.8.19-endpoint-junction-grouper.md) | 修复亚像素距离审计后，纯几何 validation 拓扑为 75.70%，只高固定半径 0.40pp | 边际 GO |
| 074 | [v0.8.20 方向对齐视觉 Endpoint Grouper](20260720-v0.8.20-visual-endpoint-junction-grouper.md) | 32px 方向对齐 patch 将 validation 拓扑提高到 77.91%，比固定半径高 2.61pp | GO / 接 selector |
| 075 | [v0.8.21 Junction-Aware Bond Selector](20260720-v0.8.21-junction-aware-bond-selector.md) | 删除 Oracle Junction 和 learned count 后，9k validation 拓扑达到 70.88%，比固定阈值高 3.21pp | 9k GO / 接实验推理 |
| 076 | [v0.8.22 Junction-Aware 正式推理接入](20260720-v0.8.22-junction-aware-inference.md) | internal 498 的 Junction exact 68.67%、SDF valid 52.41%、化学一致 13.05%；498 份预测/诊断 SDF 完整保存 | 骨架 GO / 化学层待解 |
| 077 | [v0.8.23 无效价态结构化原子解码](20260721-v0.8.23-structured-atom-decoder.md) | 固定拓扑的 invalid-only 搜索将 SDF valid 52.41%→67.07%、化学一致 13.05%→16.47% | 确定性修复 GO |
| 078 | [v0.8.24 数据规模等计算量对照](20260721-v0.8.24-data-scale-matched-compute.md) | full1k 上 9k SDF 57.4%，50k 55.7%；数据多样性有价值但 matched-compute 扩量 NO-GO | 9k GO / 50k NO-GO |
| 079 | [v0.8.25 50k Coverage 与 9k Replay](20260721-v0.8.25-coverage-replay.md) | replay 将 50k 恢复到 57.5% SDF，但跨风格几何/拓扑仍回退，净增益仅噪声量级 | sampler GO / checkpoint NO-GO |
| 080 | [v0.8.26 Junction Pair 投影一致性](20260721-v0.8.26-junction-pair-projection-consistency.md) | 统一训练/推理 winner 与坐标路径；Internal/Full1k topology exact 净增益均为 0，Edge F1 仅 +0.04/+0.01pp | 工程 GO / 能力 NO-GO |
| 081 | [v0.8.27 GT Pair 等价组 Selector](20260721-v0.8.27-equivalence-bond-selector.md) | 9k validation topology 70.88%→71.69%，extras 0.382→0.269/图；提升真实但未过 72.9% 门槛 | 监督 GO / checkpoint NO-GO |
| 082 | [v0.8.28 Dense Slot Hungarian](20260721-v0.8.28-dense-slot-hungarian.md) | 消除同 cell slot 标签置换冲突；collision exact 6%→8%，secondary recall 23.58%，普通验证明显回退 | 标签修复 GO / 固定 slot 架构 NO-GO |
| 083 | [v0.8.29 Bond Group Query](20260721-v0.8.29-bond-group-query.md) | 全局三 token 在 matched20 训练 secondary 50.23%，holdout 仅 7.55%、cell exact 1% | 容量有 / 泛化 NO-GO |
| 084 | [v0.8.30 Endpoint Local Refinement](20260721-v0.8.30-endpoint-local-refinement.md) | stride-4 端点 `5×5` patch；local-only holdout cell exact 16%、secondary 52.83% | 局部几何 GO |
| 085 | [v0.8.31 Feature Line Band](20260721-v0.8.31-feature-line-band.md) | 特征线带容量 82.18%，holdout cell exact 14%，未超过端点局部分支 | 容量 GO / 泛化 NO-GO |
| 086 | [v0.8.32 Raw Ink Line View](20260721-v0.8.32-raw-ink-line-view.md) | 原图定向线带将 holdout cell exact 提至 18%；O2 为 18/94，O3 仍为 0/6；旧排序只保留 9.71% collision recall | geometry GO / O3 NO-GO / 默认不变 |
| 087 | [v0.8.33 Local-evidence Presence Selector](20260721-v0.8.33-local-evidence-presence-selector.md) | 108,289 参数 selector 将 holdout collision known-count recall 9.71%→25.73%；safe-select 保持普通图 exact 回退仅 0.20pp | ranking GO / count、type 未训练 / 默认不变 |
| 088 | [v0.9.0 Raster-to-Visual-Primitives](20260721-v0.9.0-raster-to-visual-primitives-capacity.md) | 建立 q2047 Visual SVG 协议；旧 cross-attention 在 1,000 step 的 mismatch ratio 仍为 0.999985 | 协议 GO / cross-attention NO-GO |
| 089 | [v0.9.1 Visual Prefix Grounding 与容量](20260721-v0.9.1-visual-prefix-grounding-and-capacity.md) | content-only value 支路打通视觉梯度；clean 精调后 100 图 primitive exact 99%、数量 exact 100%、mismatch ratio 834.72 | 视觉接地与同集容量 GO / 泛化待验 |
| 090 | [v0.9.1 未见分子泛化门控](20260721-v0.9.1-unseen-molecule-generalization.md) | coordinate-aware 将 soft MAE 248.50q→236.97q，但 free completion 仅 37%、全部图元 F1 为 0、IoU 0.76%；internal test 未运行 | AR token-sequence 停止 / 转 unordered set A/B |
| 091 | [v0.9.2 无序图元集合与 Bond Group 容量](20260721-v0.9.2-unordered-primitive-set-capacity.md) | endpoint-only 将 line recall@8q 提至 85.45%、Bond IoU 提至 66.88%；coarse-only 与 flat output-head 长训 NO-GO；v0.2 group 协议 100/100 可逆 | set/refiner GO / 转 group model |
| 092 | [v0.9.3 Bond Group Unordered Set P1](20260721-v0.9.3-bond-group-unordered-set-p1.md) | 两阶段 full->refiner 将 group r@8q 提至 9.50%、IoU 18.32%，但 group exact 仍为 0；GT 证明 typed carrier 可在 1.5q 内重建 | group 方向 GO / canonical-member head NO-GO |
| 093 | [v0.9.4 Carrier 与 Flat Proposal Oracle](20260721-v0.9.4-carrier-and-flat-proposal-oracle.md) | direct carrier 仅 r@8q 4.92%/IoU 17.95%；保留 flat 坐标并做 oracle grouping 达到 85.54%/70.62% | carrier codec GO / direct 坐标回归 NO-GO |
| 094 | [v0.9.5 Frozen Proposal 与 Free Group Slots](20260721-v0.9.5-frozen-proposal-free-group-slots.md) | 2.43M free-slot 模型正式 r@8q 41.14%、IoU 38.10%；type/count 全塌为 single/count1，membership-only 仍未过门 | frozen proposal GO / free slots NO-GO |
| 095 | [v0.9.6 Proposal-Anchored Pointer](20260722-v0.9.6-proposal-anchored-pointer.md) | 局部 affinity 提升，但 16.22% root collision、p2 exact 8.73%、sample exact 0 | proposal affinity 有信号 / root NO-GO |
| 096 | [v0.9.7 Typed Affinity Packing](20260722-v0.9.7-typed-affinity-packing.md) | partition exact 77%、group F1 96.12%、r@8q 85.37%、IoU 70.17%；旧 primitive 在 unseen100 的 line F1 仅 0.055% | grouping capacity GO / 上游 9k 待训 |
| 097 | [v0.9.8 Primitive Set Pilot 9k](20260722-v0.9.8-primitive-set-pilot9k.md) | step1k line F1 仅 0.146%、IoU 11.39%；平移 16px 后 capacity F1 81.4%→0.16%，refiner 记忆绝对位置 | free-query primitive 正式 NO-GO |
| 098 | [v0.9.9 Evidence Carrier Extent](20260722-v0.9.9-evidence-carrier-extent.md) | best step900：q8 member 98.39%、single/p2 98.20%/98.25%、长键 96.15%；平移 32px 仅降 0.30pp，但 q8 sample exact 72.4% | 局部 extent GO / 完整 P0 未过 |
| 099 | [v0.9.9 P0.5 Target-Free Decoded-Line 聚类](20260722-v0.9.9-p05-target-free-decoded-line-clustering.md) | 9049 条压到 4405 clusters，cross-target merge=0、q8 零损失、p2 895/895；pair F1 98.07%、partition exact 79% | 精确去重 NO-GO / 保守预聚类 GO |
| 100 | [v0.9.9 P1 Candidate-Relative Router 审计](20260722-v0.9.9-p1-candidate-relative-router-audit.md) | 87.48% ignore 来自文字；bbox 与完美 mask 仅差 0.39pp；选择 645 参数 carrier router，不在 LSD 前擦图 | 路线 GO / 尚未训练 |
| 101 | [v0.9.9 LSD 两侧边界配对](20260722-v0.9.9-boundary-stroke-pairing.md) | molecule-disjoint192 上 multiplicity-2 pair recall 99.45%、GT-positive carrier 减少 48.84%，但仍有 2 个 cross-target、8 个 one-positive pair 和 1 个 p2 直接误合并 | pair candidate GO / 硬合并 NO-GO |
| 102 | [v0.9.10 Geometry Pair Scorer](20260722-v0.9.10-geometry-pair-scorer.md) | 全量 train-cal900：cross=0 时 precision 100%、P2 100%、p95 2.38ms，但 multiplicity-2 recall 94.56%、pairable exact 71.26%；threshold0 虽有 99.93% recall，仍产生 9 个 cross | 8D A 组 NO-GO / 转 frozen P0 visual B |
| 103 | [v0.9.10 Pair 8D 可分性与 A1 审计](20260722-v0.9.10-pair-feature-separability-audit.md) | 9 个 residual cross 的 top-10/50 邻居有 94.4%/95.3% 为 positive；top-1 重训仅 9→7，pairable exact 99.46%→97.17% | A1 仅作辅助 / 主线 B0 |
| 104 | [v0.9.10 Frozen P0 Pair Scorer](20260722-v0.9.10-frozen-p0-pair-scorer.md) | FP32 step1000 P0 + 11,873 参数 scorer：cross=0、precision/P2 100%，但 m2 recall 98.71%、pairable exact 88.53%；低阈值仅剩 1 个 surplus-anchor cross | B 组 NO-GO / 转 whole-set coverage |
| 105 | [v0.9.11 协同 Pair Coverage](20260722-v0.9.11-synergistic-pair-coverage.md) | target-free 同步 coverage veto 删除低阈值最后 1 个 cross；m2 recall 99.60%、pairable exact 95.41%、P2 100%，pipeline p95 9.49ms | post-hoc train/cal DEV-GO / 待 fresh audit |
| 106 | [v0.9.12 Fresh Paired Smoke](20260722-v0.9.12-fresh-paired-smoke.md) | train-only 8 分子/32 图 12-member 包可复现；加固后 run5 为 SMOKE-PASS，与 run1 的 32/32 功能结果一致，cross 0→0、仅 veto 3 个 same-target 冗余 pair、pair-stage p95 9.1161ms | 执行契约 GO / 非泛化 / formal dev SHA 未登记 |
| 107 | [v0.9.13 Fresh Dev 数据冻结](20260722-v0.9.13-fresh-dev-preflight.md) | 2,000 分子/8,000 图、96,000 members 完整验收；8 shard、isolation 全零重叠；commitment `67a2bffd...` 已登记 | 数据与 loader GO / sealed 未读 |
| 108 | [v0.9.14 Fresh Pair Coverage 独立审计](20260722-v0.9.14-fresh-pair-coverage-no-go.md) | fresh8k：cross 47->47，coverage veto 58 个但全为 same-target；m2 recall 99.37%、pairable exact 93.10%、P2 99.9985% | `FRESH-DEV-NO-GO` / coverage 主线停止 |
| 109 | [v0.9.15 Train-only 每图标量 Tau 可达性](20260722-v0.9.15-train-only-per-sample-tau-reachability.md) | calibration900 oracle：m2 recall 99.94527%、pairable exact 99.86505%、cross 0、P2/precision 100% | `CALIBRATION-SCALAR-ORACLE-GO` |
| 110 | [v0.9.16 Train-only 每图 Tau 预测器](20260722-v0.9.16-train-only-sample-tau-predictor.md) | 9k full-hull manifest + 1,841 参数/1,500 steps；calibration m2 99.926%、pairable 99.595%，但 cross 8、P2 violation 1 | `CALIBRATION-TAU-PREDICTOR-NO-GO` / calibration 已消费 |
| 111 | [v0.9.17 Fit-only Pair-conditioned Residual](20260722-v0.9.17-fit-only-pair-conditioned-residual.md) | m2 recall 99.58%→99.74%、pairable exact 94.84%→96.51%，但 cross 1→2 | `DEV900-PAIR-RESIDUAL-NO-GO` |
| 112 | [v0.9.18 P0 Fit-only Candidate-Pool Oracle](20260722-v0.9.18-fit-only-candidate-pool-oracle.md) | label oracle：cross 0、precision 100%、m2 recall 99.942%、pairable exact 100% | `FIT-CANDIDATE-POOL-ORACLE-GO` |
| 113 | [v0.9.18 Endpoint-Incidence Safe Residual OOF](20260722-v0.9.18-endpoint-incidence-oof.md) | OOF m2 99.55%→99.71%、pairable 94.58%→97.19%，但 cross 33→31；三折均非零 | `FIT-OOF-ENDPOINT-SAFE-RESIDUAL-NO-GO` |
| 114 | [v0.9.19 Pair-Local Raster OOF](20260722-v0.9.19-pair-local-raster-oof.md) | 完整 PNG 对齐 crop 将 cross 33→25、M2 99.55%→99.71%；全局平均池化丢失局部空间位置 | `FIT-OOF-PAIR-LOCAL-RASTER-NO-GO` |
| 115 | [v0.9.20 Pair-Local Spatial OOF](20260722-v0.9.20-pair-local-spatial-oof.md) | 3×6 空间池化将 cross 33→24、M2 99.73%、pairable 97.27%；更强表示仍会引入 OOF cross | `FIT-OOF-PAIR-LOCAL-SPATIAL-NO-GO` |
| 116 | [v0.9.21 Conservative Pair Veto OOF](20260722-v0.9.21-conservative-pair-veto-oof.md) | 单向 veto 将 cross 33→22、M2 保持 99.55%，但 exact 停在 baseline；consensus+veto 前沿可到 cross8/exact95.49% | `FIT-OOF-PAIR-LOCAL-VETO-NO-GO` |
| 117 | [v0.9.22 Junction-Negative Veto OOF](20260722-v0.9.22-junction-negative-veto-oof.md) | junction label 加权将 consensus cross 19→16，但误删 positive 21、negative 仅4 | `FIT-OOF-PAIR-JUNCTION-VETO-NO-GO` |
| 118 | [v0.9.23 High-Resolution Profile Veto OOF](20260722-v0.9.23-high-resolution-profile-veto-oof.md) | 无池化 profile 将 consensus cross 19→16，误删 positive=0；阈值前沿证明逐 pair 表示已到上限 | `FIT-OOF-PAIR-PROFILE-VETO-NO-GO` |
| 119 | [v0.9.24 有序邻域 Veto OOF](20260722-v0.9.24-ordered-neighborhood-veto-oof.md) | 12 个近共线邻居 token 将 cross 19→17，但误删 2 positive，且正确删除均与 v0.9.23 重合 | `FIT-OOF-PAIR-NEIGHBORHOOD-VETO-NO-GO` |
| 120 | [v0.9.25 多节点组件 Oracle OOF](20260722-v0.9.25-multi-component-oracle-oof.md) | 只替换多节点组件即可将 v0.9.23 cross 16→8、M2 99.83%、pairable exact 98.39% | 组件联合选择容量 GO / 非模型 |
| 121 | [v0.9.26 组件 Structured Matching OOF](20260722-v0.9.26-component-structured-oof.md) | 23,159 组件整组训练完成，但 cross 16→16、exact 下降；baseline/raw gain 造成 active-score 捷径 | `FIT-OOF-COMPONENT-STRUCTURED-NO-GO` |
| 122 | [v0.9.27 Score-Blind Counterfactual OOF](20260723-v0.9.27-score-blind-counterfactual-oof.md) | 删除全部 score 输入并反事实激活 50,595 negative，cross 仍 16→16、exact 少 1；触发硬停止规则 | `FIT-OOF-COMPONENT-SCORE-BLIND-NO-GO` / pair-stage 停止 |
| 123 | [v0.9.28 完整对象 Shadow](20260723-v0.9.28-object-shadow.md) | 冻结八组件 PNG→ChemScene→SDF；USPTO clean 100 valid 77%、非立体 exact 21%、化学 exact 19%，GD906 保留无效 SDF 与渲染图 | 完整链路 GO / 稳定替换 NO-GO |
| 124 | [v0.9.29 SDF 对象分解 Benchmark](20260723-v0.9.29-object-decomposition-benchmark.md) | USPTO clean 1k：纯骨架 72.4%、原子类型+骨架 29.2%、键级+骨架 50.5%；727 个数量正确样本仅 3 个拓扑错 | 评测协议 GO / 下一步原子类型优先 |
| 125 | [v0.9.30 OCR 与 Graph Corrector A/B](20260723-v0.9.30-ocr-and-graph-corrector-ab.md) | Open Text 1500k 直连使原子 micro 96.42%→84.63%；关闭 corrector 使键级 micro 89.47%→96.15% 但 valid -10pp；invalid-only valid +2pp、exact 不变 | 两个直接替换 NO-GO / 最小键改动搜索 |
| 126 | [v0.9.31 Line-Band Bond Semantic A/B](20260723-v0.9.31-line-band-semantic-ab.md) | 固定顶点与候选，仅用现有端点对 line-band 判键级；键级整图 61%→18%、单键级 96.15%→80.02% | 现有 line-band NO-GO / 重写 Semantic Relation |
| 127 | [v0.9.32 Semantic Bond Relation v0.1](20260723-v0.9.32-semantic-bond-relation-v01.md) | 1.68M 双尺度无序 pair 模型；degraded chemical 92.99%；USPTO 1k 键级 50.5%→57.1%、单键级 89.04%→96.51%，valid -10.6pp | relation/ring GO / shadow 替换 NO-GO |
| 128 | [v0.9.33 Semantic Bond Existence](20260723-v0.9.33-semantic-bond-existence.md) | 去除显式长度输入；sampled F1 99.39%，dense set exact 70%/66% | 表示 GO / 整图 NO-GO |
| 129 | [v0.9.34 Semantic Bond Hard Negative](20260723-v0.9.34-semantic-bond-hard-negative.md) | 模型分数挖掘困难非键，dense exact 提至 73%/73% | mining GO / 继续扩唯一图 |
| 130 | [v0.9.35 Predicted Junction Semantic Bond](20260723-v0.9.35-predicted-junction-semantic-bond.md) | 2,000 唯一图达到 78%/76%，接预测 Junction 后 USPTO skeleton 仅 69% | GT-point 分布偏移 NO-GO |
| 131 | [v0.9.36 Predicted Junction Supervision](20260723-v0.9.36-predicted-junction-supervision.md) | predicted-point dense exact 91%/91%，USPTO skeleton 69%→74%，仍低于旧 shadow 76% | 监督协议 GO / 替换 NO-GO |
| 132 | [v0.9.37 MolGrapher 风格适配](20260723-v0.9.37-molgrapher-style-adaptation.md) | MolGrapher topology 10%→20%，ChemVision 55%→40%；20% replay 仅恢复到 46% | 风格学习 GO / 遗忘 NO-GO |
| 133 | [v0.9.38 平衡双域 Replay](20260723-v0.9.38-balanced-style-replay.md) | 50/50 低学习率得到 MolGrapher 14%、ChemVision 52%，只是在双域间折中 | 共享权重 NO-GO |
| 134 | [v0.9.39 BondObject 风格 MoE](20260723-v0.9.39-bond-style-moe.md) | 内部 MolGrapher 10%→17% 且旧域不变；USPTO 1k skeleton 70.5%→69.3%，路由样本救回 1/破坏 13 | 运行时 GO / 来源 gate NO-GO |
| 135 | [v0.9.40 Bond 专家收益 Oracle](20260723-v0.9.40-bond-expert-utility-oracle.md) | 双域 topology 原专家 65/200、风格专家 60/200、oracle 并集 88/200；23 rescue / 28 break | 双专家互补 GO / 转收益可分性 |
| 136 | [v0.9.41 Bond 专家收益路由](20260723-v0.9.41-bond-utility-router.md) | no-style router 未见内部 996 张 rescue 93/break 1；USPTO 1k skeleton 70.5%→72.7%、valid 69.1%→70.4%、chemical 15.7%→16.0% | shadow GO / 默认待保守 veto |
| 137 | [v0.9.42 AtomObject + BondObject 联合协议](20260723-v0.9.42-atomobject-bondobject-protocol.md) | 删除独立 Junction 与最近邻 OCR 附着；validation/train 1494 张无损往返，未知失败 0 | 协议 GO / 模型待训练 |
| 138 | [v0.9.43 联合 Free-Query 容量](20260723-v0.9.43-joint-free-query-capacity.md) | count 100%，但 atom 6px 18.93%、bond port 8.22%、skeleton exact 0 | 双 free-query NO-GO |
| 139 | [v0.9.44 空间 AtomObject 容量](20260723-v0.9.44-spatial-atomobject-capacity.md) | anchor/role 100%，文字中心 97.54%，known-count AtomObject exact 85% | 空间完整对象 GO / count 待联合选择 |
| 140 | [v0.9.45 空间 Atom-Bond Incidence 容量](20260723-v0.9.45-spatial-incidence-capacity.md) | 74,113 参数 head；step 500 image exact 92%，step 1000 endpoint/bond/image 全部 100% | GT-associated 容量 GO / predicted proposal 待验证 |
| 141 | [v0.9.46 预测 Proposal Incidence](20260723-v0.9.46-predicted-proposal-incidence.md) | target-independent 候选；独立 validation endpoint 95.36%→97.69%、bond 91.49%→95.58%、image exact 13%→46% | Incidence GO / proposal set 待训练 |
| 142 | [v0.9.47 Atom/Bond Proposal 集合选择](20260723-v0.9.47-proposal-object-set.md) | train100 full 61%，validation 固定阈值 4%、校准 8% | 容量 GO / 100 图泛化 NO-GO |
| 143 | [v0.9.48 Proposal 集合扩量到 1k](20260723-v0.9.48-proposal-set-scale-1k.md) | validation full 8%→24%；固定 internal test100 为 19% | 1k 扩量 GO / 转全 9k |
| 144 | [v0.9.49 Proposal 集合扩量到全 9k](20260723-v0.9.49-proposal-set-scale-9k.md) | 8,974 图缓存训练；validation full 35%，固定 internal test100 为 24% | 全量扩量 GO / 接 ChemScene/SDF |
| 145 | [v0.9.50 ChemScene v0.2 Oracle Shadow](20260723-v0.9.50-chemscene-v02-oracle-shadow.md) | scene valid 100%、SDF valid 56%、非立体 SDF exact 30%；30/26/44 correct/wrong/invalid 全保存 | 编译 GO / Bond chemical type NO-GO |
| 146 | [v0.9.51 非环芳香键一致性](20260723-v0.9.51-aromatic-bond-consistency.md) | valid 56%→69%、非立体 exact 30%→36%，配对回退 0 | 编译规则 GO |
| 147 | [v0.9.52 非环芳香原子一致性](20260723-v0.9.52-aromatic-atom-consistency.md) | valid 69%→82%，清除剩余 13 个 non-ring aromatic invalid，exact 不回退 | 编译规则 GO |
| 148 | [v0.9.53 匹配键化学语义 Oracle](20260723-v0.9.53-oracle-bond-chemistry.md) | 3,198 根匹配键使用真值，valid 82%→90%、非立体 exact 36%→40% | 键语义非首要瓶颈 |
| 149 | [v0.9.54 拓扑 Closure Loss](20260723-v0.9.54-topology-closure-loss.md) | validation full 35%→37%，locked internal full 24%→24%、SDF exact 36%→36% | NO-GO |
| 150 | [v0.9.55 Object/Incidence Oracle 阶梯](20260723-v0.9.55-object-incidence-oracle-ladder.md) | 单独 Oracle 无收益；presence+incidence 同时 Oracle 使非立体 exact 36%→51% | 联合集合目标 GO |
| 151 | [v0.9.56 Endpoint 救回 AtomObject](20260723-v0.9.56-endpoint-atom-rescue.md) | validation 44%→45%，locked internal 非立体 exact 36%→36% | NO-GO |
| 152 | [v0.9.57 整图最坏项 Surrogate](20260723-v0.9.57-graph-exact-surrogate.md) | validation full 35%→37%，locked internal full 24%→24%、SDF exact 36%→35% | NO-GO / 改联合匹配协议 |
| 153 | [v0.9.58 Atom/Bond/Incidence 联合图 MILP](20260723-v0.9.58-joint-graph-milp.md) | locked internal valid 82%→83%、非立体 exact 36%→41%、chemical 31%→36%；paired exact +5/-0 | 联合解码 GO |
| 154 | [v0.9.59 联合图 Bond Chemistry Oracle](20260723-v0.9.59-joint-oracle-bond-chemistry.md) | v0.9.58 上 valid 83%→90%、非立体 exact 41%→46%、chemical 36%→40%，配对回退 0 | 键语义为第二瓶颈 |
| 155 | [v0.9.60 线性联合势函数](20260723-v0.9.60-linear-joint-potentials.md) | 482 个 train-only 可表达图 structured update 169 次，同集 exact 64.94%→64.94% | 线性势 NO-GO |
| 156 | [v0.9.61 非线性联合势容量](20260724-v0.9.61-nonlinear-joint-capacity.md) | train100 SDF exact 61%→70%，达到 70/76 可表达上限；validation full 25% | 容量 GO / 泛化 NO-GO |
| 157 | [v0.9.62 可表达难图 Replay](20260724-v0.9.62-representable-hard-replay.md) | 3,249 hard 图；validation exact 48%→50%，locked internal 41%→46%，paired +5/-0 | 能力 checkpoint GO |
| 158 | [v0.9.63 Checkpoint Invalid Fallback](20260724-v0.9.63-checkpoint-invalid-fallback.md) | 主/旧 checkpoint 分层 sanitize fallback；locked internal valid 85%、非立体 46%、chemical 41% | 当前结构 shadow GO |
| 159 | [v0.9.64 MR-TADF 主测试协议](20260724-v0.9.64-mr-tadf-main-benchmark.md) | 1,100 张主结构测试：valid 79.45%、Oracle 原子化学状态下 SDF 图 exact 55.55% | 当前主 benchmark |
| 160 | [v0.9.65 多尺度 Bond proposal 无效回退](20260724-v0.9.65-multiscale-bond-proposal-fallback.md) | 单尺度合法结果零回退；valid 88.55%、SDF 图 exact 59.27%，极端长键 +5.26pp | 当前结构主线 |
| 161 | [v0.9.66 AtomObject 近邻距离与训练量审计](20260724-v0.9.66-atom-spacing-data-audit.md) | 当前 Atom head 只训 100 张；完整 train 有 976 个 12-16px 实例，现 recall 74.69% | 原因确认 / 待全量训练 |
| 162 | [v0.9.67 AtomObject 全 9k 扩量](20260724-v0.9.67-atomobject-full9k.md) | MR-TADF Atom 整图完整率 93.73%→97.73%，但旧 selector 下 SDF exact 55.55%→48.64% | proposal GO / 直接替换 NO-GO |
| 163 | [v0.9.68 Selector 重对齐](20260724-v0.9.68-atom-selector-realignment.md) | 新 cache 重训后 validation full 33.33%，MR-TADF SDF exact 48.27% | NO-GO |
| 164 | [v0.9.69 双 Atom proposal 融合](20260724-v0.9.69-dual-atom-proposal-fusion.md) | 128+16 slot 融合，validation full 28.51%，GDG906/GDG1005 未完全正确 | NO-GO |
| 165 | [v0.9.70 Feature-Stable Atom 训练](20260724-v0.9.70-feature-stable-atom-training.md) | object feature 不变时 12-16px recall 95.65%，但 geometry/incidence 仍失配 | NO-GO / 转联合训练 |
| H01 | [历史 PaddleOCR 教师方案](historical-paddleocr-teacher.md) | 通用 OCR 不再作为训练教师 | 停止 |

## 当前下一步

v0.9.67-v0.9.70 已确认训练数据扩量能显著提高 Atom proposal：MR-TADF 顶点整图完整率
`93.73%→97.73%`，12-16px recall `89.67%→97.28%`。但直接替换、重训 selector、双头
候选融合和 feature-stable head-only 四种方案都未通过 SDF 图门控。新 Atom head 直接替换时
SDF exact `55.55%→48.64%`，说明当前分阶段训练把 Atom geometry/score 与 incidence 身份
绑定在旧联合分布上。下一步停止独立升级 Atom head，改为 Atom presence、Bond presence 和
endpoint incidence 同批联合训练，并以完整图损失监督“补回顶点后两条键同时改接”。稳定主线
仍是 v0.9.65。

v0.9.51-v0.9.56 已把当前边界定位清楚：保守芳香一致性将 SDF valid `56%→82%`、
非立体 exact `30%→36%`；匹配键 chemical type Oracle 只能到 `40%`。对象 presence 或
incidence 单独 Oracle 都不能提高 exact，二者同时 Oracle 才达到 `51%`。closure loss、
Oracle count、endpoint 单向救点和整图最坏项 surrogate 均未通过 locked internal100；
surrogate 还使 SDF exact `36%→35%`。下一版必须让 AtomObject、BondObject 与两端 incidence
在同一次联合图优化中监督和解码。v0.9.58 已用不含化学/键长规则的稀疏 MILP 验证该方向：
locked internal100 的非立体 exact `36%→41%`，完整正确新增 5、回退 0。下一版保持约束不变，
只学习适配联合解码的 Atom/Bond/incidence 势函数。v0.9.59 证明匹配键 chemistry 还能提供约
5pp 非立体 exact 上限，但优先级低于 presence/incidence 剩余约 10pp；不先接 OCR，也不使用
USPTO/MR-TADF 调参。v0.9.60 的五参数 averaged structured perceptron 在训练集 exact 零提升，
未运行 validation/internal；下一版先验证 proposal 级非线性联合势的 100 图容量。

v0.9.61 已证明非线性 proposal head 容量：固定 train100 的 SDF exact `61%→70%`，达到
70/76 可表达图。v0.9.62 在完整 8,974 图中挖出 3,249 张 representable hard 图并以 75%
replay 续训，validation `48%→50%` 后通过门控，locked internal 非立体/chemical exact
`41%/36%→46%/41%`，完整正确新增 5、回退 0。v0.9.63 仅对 sanitize 失败使用 v0.9.49
checkpoint 分层回退，将 valid `80%→85%` 且 exact 不变，现为当前结构 shadow。下一阶段转
环级 Bond chemical type，不接 OCR，不使用公开测试集调参。

v0.9.13 已完成正式 2,000 分子/8,000 图 fresh_dev。8 个 shard 的 96,000 members 全部可读，
8,000 个结构通过，双键/三键 group 为 67,996/528；final isolation 对冻结 train 暴露和 smoke
的 molecule、InChIKey、scaffold、source record、style、font、render profile overlap 全为 0。
正式 commitment SHA256 为
`67a2bffd984a95bb925b16126faa04b791a7ab2679abf0e59c92941806920d07`。

v0.9.14 在该 commitment 上完成零训练单遍 paired audit。B 的 47 个 cross-target pair 在
B+coverage 后仍为 47；coverage 删除的 58 个 pair 全部是 same-target。正式 score-frontier
对 1,068 张失败相关图片做 hash 重放：1,797 个 available miss 全部不高于 threshold，
1,771/1,797（98.5531%）按 artifact 定义严格领先 incident negatives；其中 1,322 个没有
incident negative、449 个在存在 negative 时仍领先，局部 ranking fail 为 26 个。fresh 结果
仅是 exposed diagnostic，不用于选择。

v0.9.15 已在完整 calibration900 上通过 robust scalar oracle 门禁：m2 recall
`99.94527%`、pairable exact `99.86505%`、cross 0、precision/P2 100%、support loss 0。
正式 artifact 为
`outputs/evaluations/chemvector-v0915-sample-threshold-reachability/calibration.json`，SHA256 为
`783f8a5932ef550be47eefa9ed803d459ec9589dc2352b8e32c00f0987dda28d`。这仍只是
target-aware 能力上限，不是可部署模型。

v0.9.16 已完成 8,100 fit + 900 calibration 的 full-hull interval manifest，9,000 / 9,000
样本关闭 early shortcut。只用 7,923 个 loss-eligible fit 样本训练 1,841 参数模型
1,500 步，final loss 为 `0.00226807`。正式 calibration900 中，m2 recall
`99.92612%`、pairable exact `99.59514%`、precision `99.97899%`，但产生 8 个
cross-target pair 和 1 个 P2 violation，因此结论为
`CALIBRATION-TAU-PREDICTOR-NO-GO`。

v0.9.17 已冻结 fit-only pair-conditioned residual 协议：只在原 `fit8100` 内按
molecule/source identity component 一次性拆分 7,200 train / 900 dev，训练 2,841 参数
residual scorer，并在固定 `tau0` 下与 frozen B baseline 做同次结构门禁对照。当前已完成
预注册、隔离式 split builder、模型、训练器、一次性 marker/evaluator、24 项聚焦测试、
正式 sidecar、2,000-step 训练和唯一一次 dev900 评估。candidate 将 m2 recall 从
`99.57978%` 提至 `99.73935%`、pairable exact 从 `94.83961%` 提至 `96.51325%`，但
cross-target 从 1 增至 2，因此正式结论为 `DEV900-PAIR-RESIDUAL-NO-GO`。原 calibration900、fresh_dev1、
sealed、旧 validation500、`internal_test` 和 MR-TADF 不得用于本轮训练、选择或门槛调整；
已暴露的 v0.9.17 dev900 也不得用于同版本调参或重跑，生产推理继续保持 `v0.7.10`。
fit-only 诊断同时证明现模型 forward 不含 endpoint incidence，逐 pair 平均 loss 与最终
maximum-weight matching 存在结构性错位；下一候选必须先做 endpoint-aware、matching-aware 的
OOF train-only 门禁，而不是继续扩大同一路线的数据或参数。

v0.9.18 P0 随后证明 candidate pool 本身足够：fit7200 label oracle 的 cross 为 0、precision
100%、M2 recall `99.94245%`、pairable exact 100%。因此没有回退 candidate generation。
Endpoint-incidence 6,829 参数模型完成正式 3-fold OOF 后，将 M2 recall 从 `99.55087%` 提升至
`99.71458%`、pairable exact 从 `94.58255%` 提升至 `97.18841%`，但 cross 仅从 33 降到 31，
三个 fold 分别仍有 14 / 11 / 6 个 cross，正式结论为
`FIT-OOF-ENDPOINT-SAFE-RESIDUAL-NO-GO`。

31 个 OOF selected cross 中有 24 个来自 stress、12 个由 residual 新抬过 threshold、24 个没有
任何共享 endpoint 的 positive candidate。endpoint incidence 对竞争 pair 有效，却无法判断孤立
pair 的真实局部线条关系。下一版本应从完整 PNG 提取与 candidate 几何对齐的 pair-local raster
crop，并保留 candidate mask；在新的 fit-only OOF 通过零 cross 前，不申请 `fresh_dev2`。

v0.9.19 已完成上述 pair-local raster 正式 OOF。7,200 张 fit 图共缓存 368,573 个
`2×24×48` uint8 crop，完整 PNG ink 与 candidate mask 同时输入 15,353 参数 residual。结果将
cross 从 33 降到 25，M2 recall 提至 `99.71061%`、pairable exact 提至 `96.79410%`，但三折
仍为 10 / 9 / 6 个 cross，因此仍是 NO-GO。25 个剩余错误中 22 个没有 incident positive，
13 个两端 degree 都为 1；contact sheet 显示 crop 已包含平行线、交叉线和端点干扰，但当前
global average pooling 丢失其相对位置。下一版只做 `3×6` spatial pooling A/B，复用相同 cache、
fold、loss、步数和门禁，不申请 fresh 数据。

v0.9.20 的 `3×6` spatial pooling A/B 已完成：cross 33→24、M2 recall `99.73178%`、
pairable exact `97.27413%`、support lost 31→29。空间表示提升成立，但 fold0 cross 12→14；
与 v0.9.19 对照时修掉 7 个错误、同时新引入 6 个错误。固定 `min(avg,spatial)` 共识可将 cross
降到 19，但 exact 降至 `96.38265%`，仍未达到零 cross。下一步应将双向 residual 拆为单向
veto 与受约束 rescue，分别审计 false-delete 和 introduced-cross；不再叠加第三个同类 residual。

v0.9.21 单向 veto 将 raw cross 33→22，M2 保持 `99.54954%`，但 pairable exact 仍为 baseline
的 `94.58255%`。与双模型 consensus rescue 组合后，固定 threshold=-3 的诊断达到 cross8、
M2 `99.50655%`、exact `95.49117%`、parallel violation 0；继续降低阈值会跌破 99.5% recall。
hard8 中 5 个为 parallel3/single junction、3 个为密集 single/single。

v0.9.22 尝试用 target group type 只在 loss 中强调 2,368 个 junction negatives；forward 不读取
group type。正式 consensus+veto 将 cross 19→16，但三折共误删 21 个 positive、只删除 4 个
negative，证明标签加权不能替代高分辨率 junction 证据。下一步保留 24 行 normal profile 和
left/middle/right longitudinal profile，不做 max-pool；禁止继续增加 junction loss 权重。

v0.9.23 使用 480D 不降采样 profile 和 60,505 参数 veto，在相同 consensus 上也达到 cross16，
但误删 positive 为 0、support lost 25→21、pairable exact 多正确 1 张，证明白缝与端点位置证据
有效。阈值降到 -3 时 cross 仅到10且已 veto 1,047 个 positive；降到 -4 时 cross6但 M2 只有
99.14%。因此独立 pair classifier 没有安全零-cross阈值。下一步转 carrier neighborhood / 组件级
exact-cover，同时建模相邻近共线 anchors 的 normal 排序和 left/right support。

v0.9.24 已完成 7,200 图、368,573 pairs 的 target-free ordered neighborhood sidecar 和 108,033
参数 veto。正式结果为 consensus cross 19→17、M2 `99.64083%`、pairable exact `96.39979%`，
但误删 2 个 positive；两个正确删除的 negative 均已被 v0.9.23 删除，因此该模型被 v0.9.23
严格支配。邻域信号有效，独立 pair 输出方式无效。

v0.9.25 随后完成多节点组件标签 oracle：候选图共有 23,704 个 `anchor_count>2` 组件、97,617
个 pair。只在这些组件替换 v0.9.23 score，即可将 cross 16→8、M2 提至 `99.83265%`、pairable
exact 提至 `98.38848%`，parallel violation 归零。下一实验固定为 multi-component structured
matching：孤立 pair 旁路，组件内枚举 matching，用共享-anchor 消息传递和整组结构损失训练。

v0.9.26 已冻结 23,159 个自然激活多节点组件、45,219 条边和 90,396 个 matching，并训练
154,497 参数两轮共享-anchor structured model。正式 OOF cross 保持 16，pairable exact 从
`96.39979%` 降至 `96.36551%`；共删除 477 positive 和 3 个不影响 cross 的 negative。进一步
审计显示 45,219 条活跃边只有 11 negative，且模型输入同时包含 baseline/raw gain：active positive
residual 均值 `+4.91`，active negative 也为 `+5.03`，inactive negative 才为 `-4.59`。下一版必须
移除所有 score 输入，并使用 counterfactual activation 将普通 negative 人工置于 active energy 分布。

v0.9.27 已执行最终硬门槛：154,369 参数 score-blind 模型彻底移除 baseline/raw gain，并将完整
多节点组件的 50,595 negative 统一置于 `+0.5` 反事实能量训练。正式自然 OOF 仍为 cross 16→16，
M2 `99.64116%→99.64083%`、pairable exact `96.39979%→96.38265%`。aggregate `cross<=8`
和 exact 不回归两项均失败。按照预注册规则，pair-stage residual/veto/组件头支线到此停止；下一步
返回完整 PNG 的 BondObject/TextObject→ChemScene→SDF shadow，不再新增同类 pair scorer。

v0.9.28 已冻结八个已有组件为可执行 Object Shadow，并完成 GD906 单图与 USPTO clean 100
公开门控。公开集 SDF valid 为 77%，非立体结构 exact 为 21%，化学 exact 为 19%；GD906 因
多余连接导致 C 价态 6，按协议保留 diagnostic SDF 和渲染图。完整链路与前端试用 GO，但能力
不足以替换稳定默认。下一阶段以完整对象集合和端到端公开指标训练，不再返回 pair scorer 支线。

v0.9.29 将真值和预测 SDF 拆成纯骨架、原子类型、键级和完整非立体图四层。USPTO clean 1k
中纯骨架为 72.4%，且 727 个对象数量正确样本只有 3 个连接拓扑错误；原子类型+骨架仅 29.2%，
键级+骨架为 50.5%。最大瓶颈正式定位为开放文字/原子类型，其次是对象数量和键级。该分解已
自动接入后续 ChemScene benchmark 和前端，不再用总 exact 猜测组件责任。

v0.9.30 增加对象级 micro 复核：骨架正确时单原子类型 96.45%，单键级 89.04%，键级错误的
90.9% 为 `AROMATIC -> SINGLE`。Open Text 1500k checkpoint 在 USPTO 前 100 张将原子 micro
降至 84.63%、化学 exact 归零；直接接入 NO-GO。关闭 graph corrector 虽将键级 micro 提到
96.15%，却使 valid 降 10pp。新增 invalid-only 模式可将 valid 77%→79%，但完整 exact 不变，
因此下一步转最小 chemical-type 改动搜索，不全量覆盖 BondObject 输出。

v0.9.31 按 `test1/test2` 的模糊键失败验证“一个端点对一个语义键对象”。冻结 Vertex/Junction、
候选集合和 selector 后，现有 `25×64` line-band reader 直接判键级使 USPTO 100 的键级整图
61%→18%、单键级 96.15%→80.02%。该组件仍依赖局部墨迹数量和间距，正式 NO-GO。下一版将
Relation existence、可见键语义和环级 chemical semantics 分开，芳香性不再交给单条局部线判断。

v0.9.32 新建 1.68M 参数双尺度 Semantic Bond Relation。GT pair 跨字体 validation 的 chemical
accuracy 为 92.91%，固定模糊/缩放后为 92.99%。逐键直接覆盖在 USPTO 上 NO-GO，改为完整 5/6
元环整体融合后，USPTO clean 1k 的键级整图从 50.5% 提至 57.1%、单键级从 89.04% 提至
96.51%、完整非立体图从 21.2% 提至 23.6%，纯骨架和原子层不变。SDF valid 因关闭旧全量
corrector 降至 66.3%，checkpoint 暂不替换 shadow。`test2` 剩余明确错误为 relation 漏检；
下一阶段训练通用 Relation existence，键长只用于分层评测，不调整 Vertex/Junction。

v0.9.33 将 existence 分支改为纯视觉 local/context relation，不再读取显式键长。500 张唯一图、
10k 曝光的 sampled-pair clean/degraded F1 达到 99.39%/99.36%，长度四分位召回没有断层；但新增
dense 全点对门控后，内部 100 张的整图 relation set exact 只有 70%/66%。表示方向 GO，直接生成
整图 NO-GO。下一版只做模型分数驱动的视觉 hard-negative mining 和唯一图片扩容，不做长键专项
训练，也不使用键长阈值删除候选。

v0.9.34 用模型自身分数从全部 non-bond 中挖掘 hard negative，500 张续训使 dense clean/degraded
exact 从 70%/66% 提高到 73%/73%。v0.9.35 扩到 2,000 张唯一图、20k 曝光后进一步达到
78%/76%，sampled F1 约 99.6%，最长长度分桶召回仍约 98%-99%。但接到真实预测 Junction 后，
USPTO clean 100 skeleton 为 69%，低于旧 shadow 76%；paired audit 为新增修正 0、回退 7。
因此 relation 表示 GO、直接替换 NO-GO。下一版必须用冻结 Junction 生成器产生的预测点训练，解决
GT point 到 predicted point 的分布偏移，不做长键专项训练。

v0.9.36 新建 predicted-Junction 监督协议。500 张 train 的点匹配率 99.56%、GT 键可恢复率
94.25%；100 张 validation 为 99.62%/92.55%，证明主要损失来自漏点而非坐标偏移。2k 曝光使
内部 predicted-point dense exact 从 88%/89% 提至 91%/91%；USPTO clean 100 skeleton 从
v0.9.35 的 69% 回升到 74%，bond-typed 从 55% 提到 61%。但相对旧 shadow 仍是新增修正 0、
回退 2，不能替换。pair existence 支线到此停止盲扩，下一步用 Bond 端点证据反向救回缺失
Junction，并处理多余节点；仍禁止长键专项训练和键长删边。

v0.9.37-v0.9.39 证明跨绘图器风格确实影响 BondObject：单域适配可将 MolGrapher topology
10%→20%，但共享权重会遗忘 ChemVision。双专家来源 gate 在内部达到 MolGrapher 17%、
ChemVision 55%，但 USPTO clean 1k 的 56 张路由样本只救回 1 张骨架、破坏 13 张，整体
skeleton 70.5%→69.3%。来源风格不等于专家收益，下一步只做内部 utility-oracle 和收益路由
可分性审计，不用 USPTO 标签训练 gate。

v0.9.40 的零训练 utility oracle 证明两个专家有真实互补：MolGrapher/ChemVision 合计
topology exact 从原专家 65/200 提升到 oracle 并集 88/200。下一步不是再调来源阈值，而是
在内部 train 域生成双专家 rescue/break teacher，并用专家输出分歧与结构自一致性训练收益路由。

v0.9.41 已完成 2,000 张内部 teacher 和 no-style utility router。固定 validation 零-break 阈值
在未见内部 996 张上救回 93、破坏 1；USPTO clean 1k skeleton 70.5%→72.7%、SDF valid
69.1%→70.4%、chemical exact 15.7%→16.0%。方向正式 GO，先作为 shadow；下一步只在内部
teacher 上增加保守 veto，分析外部回退但不使用公开标签重训。

v0.9.42 已停止“匿名 Junction + 最近邻 OCR 附着”，建立联合 AtomObject + BondObject
协议。ChemVision validation 500 和 train 前 1000 共 1,494 张有效样本完成 SDF/InChIKey
无损往返，2 个描绘冲突和 4 个坐标冲突被显式隔离，未知失败为 0。下一步先做 100 图
联合集合与 incidence 容量实验，不先扩大数据，也不使用 USPTO/MR-TADF 调参。

v0.9.43 的双 free-query 联合模型训练到 5000 step 后 count 均为 100%，但 Atom 6px
只有 18.93%、Bond port pair 8.22%、incidence endpoint 19.78%，整图为 0，正式
NO-GO。v0.9.44 改为同一空间 cell 直接预测完整 AtomObject，在固定 100 图上达到
anchor/role 100%、文字中心 97.54%、known-count exact 85%。下一步固定为空间
BondObject proposal 与直接 incidence，不再恢复匿名顶点或末端最近邻附着。

v0.9.45 冻结共享视觉、空间 AtomObject 和旧空间 BondObject，只训练 74,113 参数的
endpoint-to-AtomObject incidence head。GT-associated 固定 100 图在 step 500 达到
endpoint 99.51%、bond 99.11%、image exact 92%，step 1000 三项均为 100%，容量 GO。
下一步必须移除 GT proposal 对齐，在独立内部集验证 predicted-proposal incidence；该结果
不等同于未见图片或公开集准确率。

v0.9.46 使用完全 target-independent 的 top-k 局部峰 proposal，在 train100 上训练并用独立
validation100 选择 checkpoint。最佳 step300 将 eligible endpoint 95.36%→97.69%、bond
91.49%→95.58%、incidence image exact 13%→46%。但冻结 proposal 的 validation 全覆盖上限
只有 64%，且 128 个 Bond 候选尚未做最终集合选择。下一步训练 Atom/Bond presence 与
incidence 一致性，不继续追加 incidence step。

v0.9.47 联合训练 Atom/Bond presence 与 incidence，一套 178,565 参数集合头在 train100
达到 full object exact 61%，validation 固定阈值只有 4%、全局校准后 8%，证明容量成立但
100 图候选排序泛化不足。v0.9.48 保持架构不变扩到 1000 唯一训练图，validation full exact
提高到 24%，锁定阈值后的 internal test100 为 19%。数据覆盖方向 GO，下一步扩到 pilot 全 9k。

v0.9.49 将三个 train shard 的 8,974 条实际记录全部缓存并训练；0 新拒绝，3000 step 仅
61.7 秒。validation 校准 full object exact 达到 35%，锁定阈值后的 internal test100 为
24%，相对 v0.9.48 的 19% 继续提升。扩量 GO，但对象集合指标已不足以代表最终能力；下一步
接 ChemScene v0.2 和 SDF shadow。

v0.9.50 首次把 v0.9.49 对象集合编译为 ChemScene v0.2/SDF。Oracle 原子化学状态条件下，
internal test100 的 scene valid 100%、SDF valid 56%、非立体 SDF exact 30%；30 correct、
26 wrong、44 invalid 的 SDF 和渲染全部保存。invalid 中 26 个 non-ring aromatic、12 个
kekulize、6 个 valence，下一步优先修 Bond chemical type，不先接 OCR。

v0.9.84 将全图 OpenCV LSD 线段端点吸附到预测 AtomObject，并按无序 AtomPair 聚合，不设置
最大键长。MR-TADF 极端长键的 Bond proposal 整图完整率从 32.02% 提到 94.30%。正式推理只在
至少两条 raster boundary 支持、已有 learned AtomPair group 且两个端点位于不同预测分量时
提升 group presence；键型仍由 v0.9.83 group type head 输出。稳定 invalid fallback 下，外部
1,100 张 exact 从 56.55% 提到 57.55%，valid 保持 91.09%。详见
`20260724-v0.9.84-global-line-segment-rescue.md`。

v0.9.85 将旧 AtomPair group 之外的 line-only 候选追加为一等 Bond proposal。尺度不变 head
在 ChemVision/full-v0 双域 validation 上取得 82.35% presence precision，并找回 7/12 个
dense 漏键正例。冻结后运行 MR-TADF 1,100，稳定 invalid fallback 的 valid 为 91.36%，exact
为 57.64%；相对 v0.9.84 是 extreme-long 新增 2 张、standard 损失 1 张，净增 1 张。方向
边际 GO，但收益不足以继续扩大同一 MLP；下一步应提高内部 line-only rescue 正例覆盖，并把
误补边纳入整图集合目标。详见 `20260724-v0.9.85-line-only-atom-pair.md`。

v0.9.86 将版本门控从极端长键拉回普通双域结构。冻结 v0.9.83 局部 AtomPair presence/type，
新增两端邻接和整图集合上下文 residual，并以 count/degree 软监督共同训练。修复 WebDataset
shuffle 未使用 seed 和 CUDA 原子累加不可复现后，两个独立 smoke 的权重逐位一致。确定性
validation 的 presence+chemical exact `519→524`，locked internal test `518→521`，两个域均无
回退。方向 GO，长键 rescue 不进入默认 profile。详见
`20260724-v0.9.86-atom-pair-graph-context.md`。

v0.9.87 先否决全局 residual 缩放和 base-positive 规则保护：两者在 locked internal test 均只有
520 chemical exact，低于 v0.9.86 的 521。随后在训练目标中加入真键绝对正 margin，解决相对
ranking 不保证越过部署阈值的问题。validation 固定 residual scale=0.8 后 chemical exact
`524→527`；locked internal test 相对 v0.9.86 presence `715→717`、chemical `521→523`，
两域均提升。详见 `20260724-v0.9.87-positive-margin-graph-context.md`。

v0.9.88 验证候选键关系图和安全删除。18 个关系特征包括第三顶点穿越、两跳共同邻居、直接/
独占墨迹和候选线段重叠解释率。完整上下文、precision-first 和 2,000 参数 relation-only 三种
训练均未提高 chemical exact；最好 validation presence `710→711`、chemical `527→527`。
共线两跳模式边级精度为 8/9，但只在 full-v0 救回 1 张 presence，ChemVision 和 chemical 均
不变。实验 NO-GO，未消费 locked internal test，默认仍为 v0.9.87。详见
`20260725-v0.9.88-atom-pair-deletion-veto.md`。

v0.9.89 把事后逐边删除改成集合级 KEEP/ADD/DELETE/SWAP one-move。候选上限审计显示 996 张中
806 张可表达，710 张已正确，另有 65 张只需一次动作；漏键在负候选 top-16 的覆盖为 27/28，
假键在正候选 bottom-4 的覆盖为 47/47。51,844 参数动作头用 17,932 张双域图片训练，最终
validation presence/chemical `710/527→720/532`，locked internal test `717/523→730/527`，
两个域均无 chemical 回退。默认前端切换到 v0.9.89。详见
`20260725-v0.9.89-atom-pair-one-move.md`。

v0.9.90 冻结 v0.9.89 并追加第二阶段 residual move。可达上限审计表明第二次 one-move 可将
validation 的 720 张 presence exact 理论提高到 779，完整两删两加只再多覆盖 13 张，因此不做
上千动作枚举。两次 cold-start pilot 均塌缩为全 KEEP；将 v0.9.89 权重映射 warm start 后，
17,932 张双域训练把 validation presence/chemical 提到 `726/535`。locked internal test
`730/527→734/530`，但 full-v0 chemical `288→287` 回退 1 张。组件 GO、默认部署 NO-GO，
前端继续使用 v0.9.89。详见 `20260725-v0.9.90-atom-pair-second-move.md`。

v0.9.91 冻结 presence，只允许已存在边在四种键型之间 RETYPE。train-only 4,500 张的
bounded RETYPE-only 上限为 `2466→2823`，候选覆盖只漏 1 个；但 separate gate/action、
KEEP+RETYPE 联合 softmax、累计 9k 和 global safety gate 均未超过固定 train-dev 的
`551/996` chemical exact。action top-1 可提高到 43.75%，说明数据量影响排序能力；部署端
仍无净 rescue，说明下一步必须改成候选级 verifier，而不是继续训练纯 global gate。正式
NO-GO，未运行 official validation 或 locked internal。详见
`20260725-v0.9.91-chemical-retype.md`。

v0.9.92 将 global gate 改为候选级 exact verifier，直接读取被修改 group、当前/目标键型和
action margin。三个 4.5k seed 中 seed20260725 在 train-dev 达到 `551→555`、双域各 +2、
4 rescue/0 break。锁定阈值 `-1.75` 后，official validation chemical `532→537`；一次性
locked internal `527→533`，ChemVision/full-v0 分别 `239→244`、`288→289`，presence
完全不变。正式 GO，默认前端切换到 `structure-v0992-carbon`。详见
`20260725-v0.9.92-chemical-retype-verifier.md`。

v0.9.93 扩大候选级 verifier 数据量，9k 阶段 train-dev chemical exact 达到 `551→558`，
但 official validation 仍为 `532→537`，未超过 v0.9.92，因此未运行 locked internal，默认
前端不变。新增 paired style audit：124 个完整四渲染分子中，presence mixed outcome 为
67.7%，chemical 为 56.5%；layout chemical gap 为 43.0pp，multiple-bond offset 为 13.8pp，
字体家族为 6.3pp。扩量同风格 verifier 无法修复上游风格敏感，实验 NO-GO。详见
`20260725-v0.9.93-style-sensitivity-audit.md`。

v0.9.94 在 250 个 molecule、1,000 张四风格图片上训练 paired AtomPair consistency，并等量
回放 ChemVision/full-v0。全 context 微调把 mixed outcome `86→83`，但 official ChemVision
presence `348→340`；随后改为可折叠的 129 参数零初始化 adapter，只在四张均 representable
时对齐并归一化 count，最终 style exact `348→349`，仍伴随 ChemVision `348→347`。21 档
adapter scale 中无安全增益。style validation 的 representable 仅 389/498，stress 仅 47/124，
证明 context 层无法修复风格导致的 proposal 缺失。实验 NO-GO，详见
`20260725-v0.9.94-atom-pair-style-consistency.md`。

v0.9.95 将审计和训练上移到 Bond proposal。style validation 的 Bond full coverage 仅
78.71%，且它到 AtomPair representable 只再损失 3 张，确认 proposal 是第一瓶颈。完整 Bond
head 微调因改变 `object_features` 使 deployed exact 崩溃；改用 37,391 参数零初始化
heatmap/endpoint residual 后，style Bond coverage/representable 均提高 0.40pp，official 两域
整图覆盖也各提高 0.40pp。locked internal 的 proposal recall 仍双域微升，但 full-v0
deployed/full exact `251/210→244/208`，因此组件 GO、部署 NO-GO，默认前端不变。详见
`20260725-v0.9.95-bond-proposal-style-residual.md`。

v0.9.96 冻结 v0.9.95 residual，同时用新旧两套 proposal 重训 178,565 参数 set/incidence。
三 epoch 将 style full exact `172→175`、新路径 official pooled `385→387`，但原路径
`376→373`；teacher 权重从 0.1 提到 10 后，原路径 pooled 虽保持 376，仍由 ChemVision +1
和 full-v0 -1 相抵。确定性 weight-space scale `0.25/0.5/0.75/1.0` 的最坏逐域回退为
`-1/-1/-2/-2`，无安全非零点。协议 GO、checkpoint NO-GO，未运行 locked internal，默认前端
不变。详见 `20260726-v0.9.96-dual-proposal-set-incidence.md`。

v0.9.97 冻结全部共享模块，只训练 2,306 参数 proposal-delta adapter；每个新 Bond proposal
按自身 dense `(slot,cell)` 回查原 score/endpoints，不依赖 rank 或 GT。epoch 1 将 style
Bond/Full `208/172→210/173`，official ChemVision `211/170→212/171`、full-v0
`247/215→248/215`，首次通过双域门控；但 deterministic locked internal 中 ChemVision 不变，
full-v0 `258/221→257/220`。架构/official GO、部署 NO-GO，默认前端不变。详见
`20260726-v0.9.97-proposal-delta-set-adapter.md`。

v0.9.98 在训练 safety verifier 前审计 adapter action 可学习性。official 996 只有 1 rescue、
0 break；严格排除 v0.9.97 replay 后的 train-only 8,000 只有 4 rescue、15 break，净变化 -11。
4 个 rescue 与 break 都表现为新增 1 根键，residual、margin、score/endpoint delta 范围高度
重叠，不足以训练可泛化 verifier。本轮不训练模型，action-verifier NO-GO；v0.9.95-v0.9.98
residual 补丁支线停止，下一阶段回到更大规模的 Bond/set/incidence 联合训练。详见
`20260728-v0.9.98-proposal-delta-action-reachability.md`。

v0.9.99 固定下一阶段数据协议：100k style train、8k fresh dev、4k sealed test，要求与旧数据
molecule/InChIKey/scaffold 零重叠并提前固定 commitment。同时完成三域各 2 张的联合工程
smoke：997,305 参数 Bond head 与 178,565 参数 set/incidence 同时更新，Bond/set loss 和整图
surrogate 均有效，标准 checkpoint 可重载。fresh dev 尚未生成，本轮不作能力结论、不运行旧
locked、不切换前端。详见 `20260728-v0.9.99-joint-training-preparation.md`。

v0.10.0 首次使用完整 8k fresh-dev：源模型 Atom/Bond proposal recall 为 99.39%/98.19%，
Full object exact 为 20.95%。当前学习率下联合训练 10,002 张后，同一 fresh probe 的 Bond
exact 从 32.9% 崩到 2.8%，Full exact 从 26.3% 崩到 2.7%，但 Bond proposal recall 仅下降
0.26pp；旧域也同步回退。说明 proposal 仍在，但 Bond 表征移动使 set 决策边界失配。本轮
提前停止并正式 NO-GO，下一轮改为冻结 Bond 的 set-first 适配。详见
`20260729-v0.10.0-fresh-joint-10k-no-go.md`。

v0.10.1 冻结 vision、Atom 和 Bond，只训练 set/incidence 10k。完整 fresh 8k 的 Bond exact
从 28.7125% 提高到 28.9000%，Full exact 从 20.9500% 提高到 21.1875%，旧域最差 Full
exact 提高 0.60pp。它完全避免了 v0.10.0 的崩溃，但未通过预设 `+2pp` Bond 门槛：稳定性
GO、能力 NO-GO。下一步从该 checkpoint 以十分之一 Bond 学习率做小步联合。详见
`20260729-v0.10.1-set-first-style10k.md`。

v0.10.2 从 v0.10.1 热启动，将 Bond 学习率降为 `1e-7` 后联合训练 10k。同一 fresh probe 的
proposal recall 95.33%→95.35%，但 Bond exact 32.9%→23.1%、Full exact 26.0%→18.3%；
旧域 Full exact 也降到 24.10%。纯学习率缩放正式 NO-GO，后续不继续搜索更小数值，改做
冻结 teacher 的 Bond 输出 anchoring/reachability 审计。详见
`20260729-v0.10.2-slow-bond-joint-10k.md`。

v0.10.3 不再微调 Dense Bond 或追逐其 latent 表征。冻结 vision、Atom、旧 Bond 和 set 后，
对 set 最终保留的所有 AtomObject 无序对读取尺度归一化 `9x32` 特征 ribbon，直接判断
`None/四种化学键`。不输入绝对长度和位置，也不使用最近邻、键长或空白中点过滤。6 图 smoke
已验证确定性训练、反传和 checkpoint 重载；正式 10k 在独立 fresh evaluation 1k 将 Bond
exact `30.6%→45.3%`，edge recall `95.15%→95.61%`，通过 `+5pp/-1pp` 门控。架构 GO；
chemical Bond exact 为 18.3%，推理编译接入前默认前端不切换。详见
`20260729-v0.10.3-direct-atom-pair-bond.md`。

v0.10.4 保持 v0.10.3 checkpoint、presence 和 AtomPair skeleton 完全不变，只在预测图中对
完整闭合且每条边均有 aromatic 支撑的 5/6 元环执行整环 chemical type 提升。阈值仅用 fresh
calibration 500 选择；独立 fresh evaluation 1k 的 chemical Bond exact `18.3%→28.2%`、
Full chemical exact `18.2%→28.1%`、边级键型 accuracy `93.48%→94.95%`。360/1000 张发生
720 次键型修改，presence/skeleton 指标逐位不变。编译器 GO，默认前端仍不切换。详见
`20260729-v0.10.4-direct-aromatic-ring.md`。

v0.10.5 先审计出旧 Atom role head 在 fresh 1k 的 TEXT recall 仅 `1/10,607=0.0094%`，整图
role exact 为 0；GD906 的 N/N/B/F 坐标均在 1.5px 内匹配，但全部被判为 IMPLICIT。新角色门
只读 anchor 周围 12/20/32px 原始像素，冻结 v0.10.4 结构与 OCR。通用三域 10k 在独立 fresh
1k 将 role exact `0%→44.7%`、role accuracy `72.92%→99.62%`、TEXT precision/recall 提至
`99.47%/99.11%`，正式 GO。
详见 `20260729-v0.10.5-anchor-atom-role.md`。

v0.10.6 在 fresh calibration 500 上否决现成 v0.8.8 CTC 的直接复用：固定半径和预测 extent
候选中最佳原子状态 accuracy 仅 56.3%、TEXT raw accuracy 9.3%、整图 exact 0.2%。当前从
v0.8.8 初始化，在 v0.10.5 预测 TEXT 与预测 extent 1.5× crop 上用 GT 可见字符串微调；但
2.5k/5k 均塌缩为全 blank，TEXT accuracy 0%，提前停止于 6k。独立 fresh eval 未读取，正式
NO-GO。详见 `20260729-v0.10.6-deployed-atom-ocr.md`。

v0.10.7 不继承 v0.10.6 blank-collapse 权重，从 v0.8.8 重新初始化；训练只保留 v0.10.5
预测 TEXT 中已匹配且 GT 可见字符串非空的 crop，blank 与误报不进入 CTC loss。角色门、结构
与 crop 协议冻结。7.5k 最佳 TEXT raw 仅 10.11%、state accuracy 56.45%、exact 0.2%；10k
轻微回退。v0.10.8 已证明这些数值来自 AtomObject/atom_id 标签错配，且 Pixel Encoder 实际
未解冻；本轮 checkpoint 作废，不能用于能力判断。详见
`20260729-v0.10.7-true-text-ctc.md`。

v0.10.8 修复 ChemScene AtomObject 身份协议，并从干净 v0.8.8 重算得到当前 crop TEXT
69.34%、Oracle bbox TEXT 85.23%。冻结结构、角色门和 CTC，只训练 274,148 参数的
16/32/48px anchor-local bbox 回归头。三域 10k 在 fresh 1k 将 mean IoU `0.570→0.839`、
TEXT `68.87%→82.93%`、state `91.43%→95.24%`、exact `6.6%→18.9%`。bbox 组件 GO，
最终 20% exact 组合门仍差 1.1pp。详见
`20260729-v0.10.8-text-bbox-protocol-fix.md`。

v0.10.9 从干净 v0.8.8 初始化，固定 v0.10.5 role 与 v0.10.8 learned bbox，按正确
ChemScene 顺序只训练真实 TEXT crop，并显式解除 Pixel Encoder runtime freeze。三域 10k
在独立 fresh 1k 将 TEXT `82.93%→93.13%`、state `95.24%→98.00%`、整图 exact
`18.9%→33.3%`；6/6 Pixel Encoder tensor 确认发生变化，最终组合门正式 GO。详见
`20260729-v0.10.9-correct-label-full-ctc.md`。

v0.10.10 无新训练，将 v0.10.3 Direct AtomPair、v0.10.4 芳香环、v0.10.5 role、v0.10.8
bbox 和 v0.10.9 CTC 接入同一推理链；文字写回后重新裁剪 bond port。GDG906 的 B/B/N/N/F
全部正确识别并生成有效 SDF，但结构仍为 85/107 对真值 84/104，不宣称图 exact。前端新增并
推荐 `structure-v01010-direct-aromatic-ocr`，运行中动态生效。详见
`20260730-v0.10.10-unified-ocr-inference.md`。

v0.10.11 定位并处理“同一视觉原子产生多个 AtomObject 且复制邻接”的集合错误。零训练
Oracle 在 fresh 1k 将 Skeleton `45.2%→53.0%`；固定图规则达到 51.7% 且 0 break。27,045
参数 learned affinity 三域 10k 后达到 50.1%、49 rescue/0 skeleton break，但出现 3 次合法
近邻误合并和 1 次 Atom break，机制 GO、默认部署 NO-GO。详见
`20260730-v0.10.11-duplicate-atom-affinity.md`。

v0.10.12 将训练扩至最多 50k，并用 4k route-select / 3,974 route-confirm 取代 500/1k
小校准。40k winner 在 confirm 将 Atom exact `51.21%→55.28%`、Skeleton
`46.53%→50.15%`，162/144 rescue、Atom/Skeleton break 均为 0；但 identity precision
99.33% 未达到预注册 99.9%，因此默认部署仍 NO-GO。选模同时修复为安全资格优先并永久保存
每个 step。详见 `20260730-v0.10.12-safe-duplicate-affinity-40k.md`。

v0.10.13 用方向性 `DELETE(node)` action 替代 same-GT identity。全新 route-select 8k 的
single-delete oracle 将 Atom exact `50.43%→61.40%`、Skeleton `45.95%→55.04%`，两桶
均零 break；但 63,374 参数 learned head 训练 50k 后，没有 checkpoint 同时达到零 harmful、
Skeleton `+2.5pp` 和 winner recall 25%。因此在 route-select 直接 NO-GO，42k confirm
保持 `000` 且未读取。详见 `20260730-v0.10.13-safe-deletion-action-50k.md`。

v0.10.14 将 safety 与 utility ranking 解耦，对 route-select 8k 运行 molecule-grouped
五折 OOF：三折训练、一折校准、一折评估，每折三个 seed。保守 ensemble 的 Atom/Skeleton
分别提高 `+2.15pp/+1.84pp`，但 407 次执行仍有 3 次 harmful，winner recall 仅 17.03%，
正式 NO-GO。取证证明 420D pooled feature 丢失逐邻边 chemical type 对应关系；confirm 与
sealed test 均未读取。详见 `20260730-v0.10.14-unsafe-certificate-oof.md`。

v0.10.15 不再训练 safety probability，改用 2px、非空等邻接、逐邻边 chemical type 一致的
structural-twin 规则，再由冻结 ranking 每图最多选一次。未见 route-confirm 42k 实际执行
4,605 次，0 harmful、0 coverage loss，Atom/Skeleton exact 分别 `+5.99pp/+5.13pp`，单侧
95% safe precision 下界 99.93497%，结构门正式 GO。完整 OCR/SDF/chemical exact 未回归，
默认前端仍为 v0.10.10。详见 `20260730-v0.10.15-structural-twin-certificate.md`。

v0.10.16 将完整候选接入 Direct AtomPair 输出之后、OCR/芳香环/SDF 编译之前，并同步过滤
removed query 的全部入射 pair。固定 20 action + 20 control 的端到端 A/B 中，执行协议 40/40
正确，SDF valid `18→24`、非立体 exact `9→12`、chemical exact `5→6`，所有回退为 0，
control OCR 与图零变化。工程 GO；该 pilot 来自开发集，前端 profile 暂保持非推荐。

2026-07-31 回到 v0.9.41 语义 BondObject 主线验证文字笔画污染：只删除两个语义端点都位于
同一 Text Object bbox 内部的 Bond 候选。固定 USPTO 前 100 仅 1 张触发、删除 2 个未进入
最终图的低层候选，Skeleton/chemical exact 均 `0 rescue / 0 break`。说明通用无语义点线
确有文字混淆问题，但它已不是 v0.9.41 的主要残差；该消融不扩测、不占用新版本号。详见
`20260731-v0.9.41-text-interior-bond-veto-pilot.md`。

v0.10.17 回到 v0.9.41 语义 BondObject utility router，只将协议兼容的 v0.8.8 CTC 替换为
v0.10.9 正确标签开放 CTC。两个内部 held-out 域 matched OCR 分别提高 `+9.51pp/+8.02pp`。
USPTO clean 1k 的 Skeleton 保持 72.7%，SDF valid `70.4%→78.5%`、chemical exact
`16.0%→21.5%`；但分别有 17/20 个 paired break，因此能力 GO、部署保持非推荐 shadow。
详见 `20260731-v0.10.17-bondobject-ocr-swap.md`。

v0.10.18 使用内部 4,000 张双 OCR teacher 训练 50→64→32→1 utility router。validation
break-0 阈值在首次 internal test 出现 2 个 break，未接公开集；固定更保守的 `0.95` 后，
fresh 2k confirm 为 10 rescue/0 break。USPTO clean 1k 只 route 12 次，Atom-typed
`293→301`、labeled graph `237→244`、chemical exact `160→162`，所有 paired break 为 0。
详见 `20260731-v0.10.18-conservative-ocr-router.md`。

v0.10.19 在同一双 OCR router 中加入主/专家各 48 维视觉描述及其差值，输入从 50 维增至
194 维。validation break-0 rescue 从 81 增至 394；但固定 `0.95` 在未消费 fresh_dev
下一段 2,000 张中 route 194 次，出现 185 rescue/3 break，均为 `O/OH` 显式氢混淆。
正式 NO-GO，未运行 USPTO、未接前端。详见
`20260731-v0.10.19-visual-ocr-utility-router.md`。

v0.10.20 补齐两 OCR 对两候选的 `2x2` CTC 交叉打分，输入为 198 维。独立第三段 fresh
2k 在固定 `0.95` 下 route 364 次、348 rescue、5 break；最高错误概率 `0.994`，错误包含
`O/Cl`、`O/OH`、`N/NH`。因此停止堆局部 router 特征，下一步改为图级双候选反事实。
详见 `20260731-v0.10.20-cross-score-ocr-utility-router.md`。

v0.10.21 不训练 router，只在主分支 SDF 无效且专家分支有效时采用专家。统一推理重新运行
USPTO clean 1k：触发 242 次、采用 98 次，SDF valid `704→802`、nonstereo exact
`203→232`、chemical exact `160→187`，所有指标 0 break。作为非推荐 shadow 接入前端。
随后完整运行 MR-TADF v1 1,100 张：valid 65.91%、Skeleton exact 48.09%、Chemical exact
24.00%；fallback 只增加 48 个 valid，完整 exact 无增益。MR-TADF 改为主应用 benchmark，
USPTO 降为公开 OOD 护栏。

2026-08-02 G3 P0-C 在 GT Atom-slot oracle 下统一 Bond group loss 与 pointer loss 的
confidence-aware Hungarian identity。固定 overfit32 训练 2,400 steps 后，count 与 Atom
set 均为 32/32，top-K/matched overlap 从 P0-B 的 54.77% 提高到 85.15%；但 query-set、
topology、type、structural graph exact 全部仍为 0/32，正式 NO-GO。说明 confidence 标量
能缓解但不能定义稳定 Bond 对象身份；下一步先做 GT Bond-slot identity 因果隔离。详见
`20260801-g3-p0-overfit32.md`。

同日 G3 P0-D 用 GT endpoint 只做 no-grad permutation，将 predicted visual proposals
重排到 canonical Bond prefix，未向 decoder 写入 GT 坐标或化学属性。router 始终覆盖
32/32 图、1508/1508 条 Bond，最终 count 和 Atom set 均为 32/32，但 topology/type/graph
只有 2/32，正式 NO-GO。final pointer 双端点同时正确仅 58.75%；1508/1508 只是无阈值
Hungarian 的协议覆盖，不代表 proposal 质量。现有 artifact 尚不能区分 route churn、几何
误差和 incidence/loss 冲突，下一步先重放六个 checkpoint 做 identity/geometry/micro 审计，
再决定是否切到参数中固定的 learned Bond slots。详见 `20260801-g3-p0-overfit32.md`。

P0-D 六 checkpoint 只读重放随后确认 step 400→2400 candidate churn 为 91.05%，final
router endpoint median/p90 为 12.81px/79.01px，4/8/16px 覆盖仅
14.66%/38.26%/55.17%；pointer 双端 56.96%，独立 type 85.68%，pointer 正确条件 type
94.30%。直接证据支持动态 proposal identity 与几何是主断点，键型不是首要瓶颈；下一轮
进入固定 learned Bond slots。详见 `20260801-g3-p0-overfit32.md`。

P0-E/P0-F 随后证明固定 learned Bond slot 的 canonical prefix 没有跨图视觉含义，而逐 step
Hungarian set matching 的 first-to-final query identity churn 达到 99.14%。P0-G 改用全部
无序 GT AtomPair 作为 Bond identity，不做距离或近邻裁剪，fixed-final 的 GT/deploy
skeleton 与 typed graph 均达到 32/32，pair identity churn 为 0%，正式通过容量 Gate。
这只证明完美 Atom set 条件下 relation 可学；下一轮 P0-H 从同初始化、同预算做 Atom-only
overfit，先把 count、candidate recall、selector identity 和 localization 四层拆开。详见
`20260801-g3-p0-overfit32.md` 与 `20260802-g3-p0h-atom-only-overfit32.md`。
