# ChemPic2SDF

ChemPic2SDF 是面向化学结构图片识别的视觉到结构项目。v0.8 开始将图片建模为
Junction、TextObject 和 BondObject 组成的 ChemScene，再通过确定性编译导出 SDF。

当前完成的研究 tag：`v0.10.15`。当前稳定推理基线为 `v0.10.10`；开发版本
`v0.10.16`。v0.10.15 的 structural-twin 组件已通过 42k confirm 结构门，v0.10.16
正在执行统一 OCR/SDF/chemical exact 回归，因此默认推理暂不替换。

当前研究实验为 `v0.10.16`。从 v0.9.86 开始，版本门控重新以 ChemVision/full-v0 普通结构的
双域 validation 和 locked internal test 为主；MR-TADF 长键只保留为专项诊断，不再主导模型方向。
此前的 locked internal100 继续作为历史回归护栏。MR-TADF 仍然只测不训，不参与阈值拟合或
hard replay。
v0.9.51-v0.9.52 的保守芳香一致性将 internal test100 的
SDF valid 从 56% 提到 82%、非立体 exact 从 30% 提到 36%。Oracle 阶梯表明正确 Bond
chemical type 只能将 exact 提到 40%，而 Atom/Bond presence 与 incidence 同时正确可到
51%。v0.9.58 首次将 Atom presence、Bond presence 和 endpoint incidence 放进同一个稀疏
MILP 联合求解，locked internal100 达到 `83% SDF valid / 41% 非立体 exact / 36% chemical
exact`，相对独立基线新增 5 个完整正确结构、回退 0。下一步学习适配联合解码的势函数，稳定
默认尚未替换。v0.9.59 进一步测得匹配键 chemical type Oracle 可将联合 exact 提到 46%，
说明键语义约有 5pp 上限。v0.9.61 证明 proposal 级非线性头在 train100 可达到 70/76 的
可表达上限；v0.9.62 用全 9k 的 3,249 张可表达难图 replay，将 locked internal 非立体/
chemical exact 提到 46%/41%。v0.9.63 加 checkpoint 级 invalid fallback 后，最终达到
`85% SDF valid / 46% 非立体 exact / 41% chemical exact`，完整正确新增 5、回退 0。
v0.9.64 使用同一 v0.9.63 解码协议完成 1,100 张 MR-TADF 主结构测试：有效 SDF
79.45%，Oracle 原子化学状态下 SDF 图完全一致 55.55%；标准、长、极端长键分桶 exact
分别为 68.49%、55.37%、16.23%。该数字不是完整 OCR 端到端准确率。
v0.9.65 确认极端长键的 Bond proposal 完整覆盖只有 32.02%，并接入已有多尺度 Bond
checkpoint 作为 invalid-only fallback。旧版合法结果全部保留，MR-TADF 有效 SDF 提到
88.55%，Oracle 原子化学状态下 SDF 图完全一致提到 59.27%；标准、长、极端长键 exact
分别达到 71.80%、58.76%、21.49%。
v0.9.66 进一步确认当前 AtomObject checkpoint 只在固定 100 张图片上训练。未被 Atom
head 使用的 ChemVision train 实际有 8,974 张，其中包含 976 个 `12-16px` 近邻原子
实例；当前 checkpoint 在该桶 recall 只有 74.69%。因此 GDG906/GDG1005 的顶点合并主要
是 AtomObject 训练量与联合目标问题，不需要数据人员重新生成训练集。
v0.9.67 将 AtomObject 扩到完整 train 后，MR-TADF Atom proposal 整图完整率从 93.73%
提高到 97.73%，12-16px recall 从 89.67% 提高到 97.28%，确认数据扩量有效。但直接替换
旧 Atom head 会使单尺度 SDF 图 exact 从 55.55% 降到 48.64%。v0.9.68-v0.9.70 又依次
否决了 selector 重对齐、128+16 双头候选融合和 feature-stable head-only 训练。问题已经从
“Atom 数据不够”推进为“分阶段训练造成 Atom geometry/score 与 incidence 联合分布失配”。
稳定结构主线仍为 v0.9.65，下一阶段改为 Atom/Bond/incidence 同批联合训练。
v0.9.71 在 full-v0 的 40k 图片上完成同批联合训练，双域 validation 的 Atom deployed exact
由 50.30% 提到 65.56%，incidence exact 由 52.31% 提到 67.57%，但 Bond deployed exact
降至 44.18%；30k 的 full-object exact 最佳为 37.25%，40k 回退。v0.9.72 冻结其余模块
校准 Bond residual 后仅提高到 37.65%。MR-TADF 外部 exact 为 54.00%，低于 v0.9.65 的
59.27%，因此两版均不替换稳定结构主线。结论是扩量有效但不充分，下一瓶颈是 Bond presence
稳定特征与跨布局泛化，不再继续盲目增加同分布数据。
v0.9.84 增加长度无关的全图线段到 AtomPair rescue。它不设最大键长，不把双键/三键拆成
四个或六个顶点，也不直接决定键型；线段只纠正已有 AtomPair group 的存在性。MR-TADF
极端长键 proposal 整图覆盖从 32.02% 提到 94.30%。在 v0.9.83 稳定栈 invalid fallback 下，
1,100 张外部测试的 SDF valid 保持 91.09%，Oracle 原子化学 exact 从 56.55% 提到 57.55%；
标准、长、极端长键 exact 分别为 67.63%、64.41%、21.49%。该结果证明长键前端方向成立，
但仍是结构实验，文字原子在上传推理中继续按 C 占位。
v0.9.85 进一步允许没有旧 AtomPair group 的 line-only 候选成为一等 Bond proposal。presence
和键型头只使用沿线视觉、两端 AtomObject 以及尺度不变的局部几何，不读取绝对键长。模型仅用
ChemVision/full-v0 选择 checkpoint 和阈值；MR-TADF 1,100 只作冻结外部验收。稳定 invalid
fallback 后，SDF valid 从 91.09% 提到 91.36%，exact 从 57.55% 提到 57.64%；极端长键
exact 从 21.49% 提到 22.37%，标准键回退 1 张。该结果是边际 GO，说明 line-only proposal
能够补回旧 group 之外的键，但当前小样本 head 尚未形成显著提升。
v0.9.86 回到普通结构主线，冻结 v0.9.83 AtomPair 局部分类器，只训练整图上下文 residual。
每条候选键同时读取两端原子的邻接候选和全图集合摘要，训练加入集合基数与逐原子邻接度监督，
不使用硬价态、绝对键长或分子类型规则。确定性双域 validation 的 presence+chemical exact
由 519/996 提到 524/996；冻结 locked internal test 由 518/996 提到 521/996，两个域均无回退。
默认 v0.9.86 profile 不启用 v0.9.84/85 长键 rescue，文字原子仍按 C 占位。
v0.9.87 修复图上下文“相对排序正确但真键 logit 仍低于部署阈值”的问题。训练增加真键绝对
正 margin，并仅在 validation 上将 residual scale 固定为 0.8。locked internal test 的
presence exact 从 v0.9.86 的 715/996 提到 717/996，presence+chemical exact 从 521/996
提到 523/996；ChemVision/full-v0 两域均无回退。默认 profile 继续关闭长键 rescue。
v0.9.88 将候选键建成关系图，显式加入第三顶点穿越、两跳共同邻居、直接墨迹、重叠解释率和
独占墨迹等 18 个特征，只允许否决 v0.9.87 已判为正的键。三种训练目标中最好结果仅将双域
validation presence exact 从 710/996 提到 711/996，chemical exact 保持 527/996；假键减少
55 条但新增 43 条漏键。高精度共线两跳模式为 8/9，但整图仅在 full-v0 增加 1 张 presence，
chemical 不变。该版 NO-GO，未运行 locked internal test，默认前端继续使用 v0.9.87。
v0.9.89 将逐边 veto 改成整图 one-move action：每张图联合比较 KEEP、top-4 DELETE、top-16
ADD 和 64 个 SWAP，不枚举任意边集合。17,932 张双域训练后，validation presence/chemical
exact 从 `710/527` 提到 `720/532`；locked internal test 从 `717/523` 提到 `730/527`，两个域
均无 chemical 回退。默认前端已切换到 v0.9.89，仍不启用长键 rescue，文字原子继续按 C 占位。
v0.9.90 冻结 v0.9.89，再训练第二阶段 residual move。warm-start 全量 checkpoint 在 validation
将 presence/chemical exact 提到 `726/535`；locked internal test 提到 `734/530`，但 full-v0
chemical exact `288→287` 回退 1 张。该版证明二次整图编辑可以继续减少漏键，但 presence-only
动作尚未约束键型效用，因此组件 GO、默认部署 NO-GO；前端继续使用 v0.9.89。
v0.9.91 冻结 v0.9.89 的 presence，只允许对已存在的一条键执行 RETYPE。train-only 4,500
审计中 typed exact 为 2,466，一次 bounded RETYPE-only 可达 2,823，16 条候选只漏 1 个。
分离 gate/action、KEEP+RETYPE 联合 softmax 和 global safety gate 均未把固定 train-dev chemical
exact 从 `551/996` 提高；累计 9,000 张后 action top-1 从 32.50% 提到 43.75%，但实际最佳
仍是 rescue/break `0/0`。该版 NO-GO，未运行 official validation 或 locked internal，默认
前端继续使用 v0.9.89；下一版改为读取具体候选边与目标键型的候选级 verifier。
v0.9.92 对每个具体 RETYPE 候选直接预测“执行后整图是否 exact”，并用最强错误动作作
hard negative。固定 seed 和阈值后，official validation chemical exact `532→537`，locked
internal `527→533`；internal ChemVision/full-v0 分别 `239→244`、`288→289`，presence
完全不变，`rescue=6, break=0`。该版 GO，默认前端切换到 `structure-v0992-carbon`；OCR
仍未接入，文字原子继续按 C 占位。

版本管理规则：每个产生正式 checkpoint 的模型版本都必须提交可复现代码和中文训练记录，并创建
与模型版本名完全一致的 annotated Git tag。tag 注释记录 checkpoint 路径、SHA256、训练样本数和
部署结论；模型权重、训练数据和生成图片不提交到 Git。

## 当前基线

- 完整模型约 5,149 万参数；无自回归 decoder 的部署路径约 1,518 万参数。
- GD906 可恢复为 `C80H44BFN2`，InChIKey 为
  `HJVIWKWTJWDQQZ-UHFFFAOYSA-N`。
- 500 张 scaffold 独立 CPU 盲测：完整拓扑 20.6%，化学完全一致 13.2%。
- v0.7.1 MR-TADF Oracle 表明当时第一瓶颈是 OCR：仅替换文字真值时，化学完全
  一致率由 33.82% 提升到 75.82%；顶点与文字同时使用真值时达到 94.91%。
- 完整分子字符替换 OCR 的平衡版 3000-step 实验，在未使用任何 MR-TADF 训练
  样本的前提下，将 1100 张外部测试的有效 SDF 从 71.91% 提升到 83.64%，
  化学完全一致率从 33.82% 提升到 44.82%。该 checkpoint 尚未打正式版本 tag。
- pixel OCR 直接从原图裁取字符区域，不再使用 stride-8 几何特征。候选模型在
  MR-TADF 上达到 52.55% 化学完全一致率和 75.55% 有效 SDF；原图修正版为
  49.91% / 81.36%。两者均未打正式版本 tag。
- v0.7.5 使用 `12/20/32px` 三尺度共享 Pixel OCR，并只对 B 全量、N 5% 采样
  文字 bbox 内锚点。训练数据池为 ChemVision pilot 的 9,000 张图片，最终
  checkpoint 位于 500 step、batch 4，实际约 2,000 次图片采样；在完整
  1,100 张 MR-TADF 上达到 89.73% 有效 SDF 和 67.45% 化学完全一致；正确、
  错误和无效 SDF 均已保存在 benchmark 输出中。
- v0.7.5 的 358 个错误样本中，255 个主要是顶点检测或定位问题，142 个含 OCR
  错误，122 个含多余键，其中 64 个是明确的交叉错连。下一阶段优先改进顶点
  召回和定位稳定性，OCR 作为并行分支持续维护。
- v0.7.6 使用同一 ChemVision pilot 的 9,000 张图片，只训练 92,548 个 vertex
  head 参数，其余模型权重逐位不变。MR-TADF 有效 SDF 达到 91.91%，化学完全
  一致达到 77.55%，三个键长难度桶全部提升；当前剩余 191 个顶点主错误。
- v0.7.9 使用 9,000 张图片训练全部原子点对的 relation/pair 参数；v0.7.10 沿用
  同一 checkpoint，以连续边概率恢复断图。USPTO clean 1k 有效 SDF 达到
  192/1000，MR-TADF 有效 SDF 达到 92.09%，化学完全一致保持 79.73%。
- v0.7.11 的保守结构 MoE 已复核为 NO-GO：USPTO/MR-TADF 的结果来自完全回退
  v0.7.10，MolGrapher routed detected skeleton exact 仅 1.3%。代码保留为实验
  基础设施，不作为模型能力提升。
- v0.7.12 在固定 100 张干净 MolGrapher 图片上验证集合式顶点：head-only 和解冻
  stride-4 block 的 vertex exact 分别为 87% 和 90%；Atom Query 负责计数、dense
  peak 负责一对一坐标吸附后，Count exact 达到 100%，4px vertex exact 达到 95%。
  接入旧 relation 后 skeleton exact 最高 23%。
- v0.7.13 在同一固定 100 图上，用 Hungarian 映射给检测顶点分配关系标签，并保留
  全部真边、挖掘最高分假边。只训练 relation head 达到 54% skeleton exact；同时训练
  146,308 个 pair/relation 参数后达到 89%，通过 85% 过拟合门槛。该结果不是外部
  泛化指标；扩量前先审计 MolGrapher 的低概率原子序号增强和硬错误，并重新训练 9k
  顶点分支。原子序号像素启发式只做人工复核，不直接删除训练样本。
- v0.7.14 完成 MolGrapher 9k/1k 顶点扩量。Atom Query 45k 的 Count exact 最高
  17.9%；dense head 约 44k 后在 6px 达到 89.06% F1、20.7% Count exact 和 5.3%
  vertex exact。继续追加 40k 后最高 vertex exact 为 6.3%，已进入平台。全局计数头
  和 dense 峰值计数分类/回归均未超过固定阈值，9k relation 因顶点未过门槛而暂缓。
- v0.7.15 只解冻最后一个 stride-4 视觉 block，将 MolGrapher validation 1k 的
  6px vertex exact 从 6.3% 提高到 9.5%。固定宽 NMS 可在同源数据达到 23.1%，并在
  USPTO 提高原子数一致和有效预测相似度，但会因绘图尺度变化增加断图，因此不设为
  默认。下一步改为对 dense proposals 做一对一学习式筛选。
- v0.7.16 新增 56,673 参数的一对一 proposal keep head。它冻结 dense 模型，用
  Hungarian 标签把重复峰训练成困难负样本；6px vertex exact 达到 22.0%。在未参与
  调参的 USPTO clean 1k 上，有效 SDF 从同权重基线的 63 提高到 107，但仍低于稳定
  v0.7.10 的 192，因此只作为 shadow 组件保留。
- v0.7.17 在稳定 v0.7.10 特征上重训尺度增强 keep head。USPTO clean 1k 有效
  SDF `192→226`、原子数一致 `8.7%→10.1%`，但完全一致 `22→19`，断图明显增加。
  该组件继续保持显式开关，不替换默认稳定推理；cardinality 和固定最小保留比例均
  已判定 NO-GO。
- v0.7.18 对最多 192 个高召回候选的全部点对采样原图线带，用 edge supervision
  产生 top-k incident support 反向救回节点。MolGrapher 1k 的 6px vertex exact 为
  7.5%，edge F1 为 78.35%；USPTO clean 1k 有效 SDF `226→241`、完全一致
  `19→29`。组件和 shadow 推理 GO，但断图仍高于 v0.7.10，暂不静默替换稳定默认。
- v0.7.19 尝试仅在断图 repair 时用高置信 graph edge 提升补桥排序。USPTO clean 1k
  只改变 2 个样本，完全一致仍为 29，并把一个断图变成价态失败，判定 NO-GO。
  接口默认关闭；下一步改为训练期 relation fusion，不继续在公开测试集上降低阈值。
- v0.7.20 增加两层全局 proposal transformer 和 node-edge 联合梯度。held-out 字体
  vertex exact `32.93%→36.75%`，USPTO clean 1k 有效 SDF/完全一致 `241/29→258/30`；
  但统一 threshold `0.90` 在大型 MR-TADF 上过删顶点，完全一致降至 49.00%。全局图
  组件 GO，固定阈值部署 NO-GO，稳定默认仍为 v0.7.10。
- v0.7.21 回到集合式 Atom Query 主线：逐层 Hungarian 监督，并以低学习率解冻
  stem/stride-4 视觉块，让 query 直接预测 object/no-object 与坐标。固定 100 图上，
  单次局部 refinement 在 75% 形成平台；增加零初始化的第二次可微局部 refinement
  后，纯 query 的 6px vertex exact 达到 97%，Count exact 为 100%。全程不使用
  dense snap，也不使用 MR-TADF/USPTO 调参。扩到 scaffold 独立的 MolGrapher 9k/1k
  后，6px F1 从历史 20.40% 提高到 39.95%，但 Count exact 只有 20.2%，整图 exact
  仍为 0；Poisson-binomial 计数损失进入最后两层 query decoder 后，Count exact
  提高到 23.9%、F1 提高到 40.14%。方向有效但尚未产生整图 exact，下一步是显式
  全局 cardinality token，而不是继续扩大 decoder 解冻范围。
- v0.7.22 增加显式全局 cardinality token，依次关注 128 个 atom queries 与整图
  视觉 memory。固定 100 图可达到 Count exact 100%，但 MolGrapher scaffold 独立
  1k 上，token-only 分类、联合 token 分类和 Poisson 集合计数分别只有 18.1%、
  21.5% 和 22.7%，
  未超过 v0.7.21 的 23.9%。更关键的是 Oracle 直接提供真实原子数后，6px F1 仍仅
  40.22%、整图 exact 仍为 0，证明主瓶颈不是计数头，而是 query 候选覆盖、排序和
  坐标质量。cardinality token 单独方向 NO-GO，稳定默认仍为 v0.7.10。
- v0.7.23 修复 Hungarian 匹配中的自强化：正样本身份不再使用 presence 奖励，只按
  最终坐标做一对一匹配；计数 presence、最终坐标感知 selection 和 ranking presence
  三者独立。MolGrapher scaffold 独立 1k 上，部署型 set-count 6px F1 从 40.14%
  提高到 54.79%，known-count selection 为 55.10%，coordinate Oracle 为 65.39%；
  Count exact 为 19.6%。顶点分配与排序方向 GO，计数和整图 exact 仍未过门槛，稳定
  默认继续使用 v0.7.10。
- v0.7.24 复核发现 v0.7.23 的冻结 ranking prior 仍有正贡献，不直接删除；随后将
  MolGrapher 式 stride-4 dense 顶点 head 打包进 Atom Query checkpoint，与 query、
  relation 共享同一视觉主干，只增加 92,678 个参数。MolGrapher scaffold 独立 1k
  的部署型 6px F1 从 54.79% 提高到 76.61%，coordinate Oracle 从 65.39% 提高到
  93.62%；跨字体 internal 500 保持 76.77%。顶点精定位组件 GO，但 Count exact 和
  relation 仍使 skeleton exact 为 0，稳定默认继续使用 v0.7.10。
- v0.7.25 固定 v0.7.24 的 set-count + shared dense-snap 顶点，只训练 146,308 个
  pair/relation 参数。MolGrapher scaffold validation 1k 的 Edge F1 从 52.56%
  提高到 62.70%，假边从 11,065 降到 4,408；未参与训练的跨字体 internal 500
  Edge F1 从 53.99% 提高到 58.28%。relation 组件判定 GO，但 vertex exact 与
  Count exact 仍使 skeleton exact 为 0，因此不替换稳定 v0.7.10。
- v0.7.26 不再让全部 query 抢占 dense 峰，而是保持 dense 候选坐标不动，聚合附近
  Atom Query 上下文后做集合筛选。MolGrapher validation 1k 的 6px 顶点 F1
  `76.61%→90.02%`、Count exact `20.0%→27.5%`、vertex exact `0→6.1%`；跨字体
  internal 500 顶点 F1 达到 92.12%。接入 v0.7.25 relation 后，MolGrapher 和跨字体
  skeleton exact 首次达到 5.7% 和 4.2%。组件 GO，稳定默认仍为 v0.7.10。
- v0.7.30 将 dense 候选扩到 256，MolGrapher scaffold 独立 1k 的 Candidate
  Oracle exact 达到 88.7%。候选中心单点视觉将 strict vertex exact 提至 3.5%，
  3x3 对齐 stroke feature patch 进一步达到 4.4%、F1 84.72%。候选覆盖与坐标对齐
  视觉方向 GO，但距离稳定部署仍远；下一步直接优化最强假候选与最弱真候选的极值
  排序，稳定默认继续使用 v0.7.10。
- v0.7.36 对文字/隐式双分支端点的全部无向点对读取二维宽线带，一对端点只输出
  一个 Bond Object，不使用最近邻或键长预删除。真实端点 9k/498 验证达到 99.16%
  Edge F1 和 78.31% skeleton exact；接入预测端点后，匹配端点条件 Edge recall
  仍为 98.70%，但 9.84% vertex exact 将 strict skeleton exact 限制为 7.83%。
- v0.7.37 统一使用 Render Layer 作为 OCR/顶点角色真值，修复 710 个视觉与语义
  标签冲突；候选等价组 REAL/NULL 监督配合同角色 6px 去重后，9k/498 在不提供
  Count 时角色严格 vertex exact 达到 56.02%。接入 v0.7.36 Bond Object 后，
  strict skeleton exact 从 7.83% 提高到 41.37%，匹配端点条件 Edge recall 保持
  98.60%。`bond=0.80` 仅为 shadow 候选，稳定默认仍为 v0.7.10。
- v0.7.38 按通用验证集实测的亚像素端点误差微调 Bond Object，不加入长键专属规则。
  498 张 strict skeleton exact 从 41.37% 提高到 51.00%，匹配端点条件 Edge recall
  提高到 99.56%，`>96px` 长键条件 recall 从 92.84% 提高到 96.08%；真实端点
  skeleton exact 无回退。组件进入 shadow，稳定默认仍为 v0.7.10。
- v0.7.39 扫描 Text/Implicit 独立阈值并训练候选绝对边界。role vertex exact 提高到
  59.64%，strict skeleton exact 提高到 53.61%，但未达到 62%/57% 正式门槛，
  且大分子退化，因此默认 NO-GO；当前 shadow 仍为 v0.7.38。
- v0.7.40 对 256 个候选的全部 32,640 点对做完整邻接消息传递，固定 100 图容量
  达到 89%，但 9k 端到端 role exact/skeleton 只有 57.83%/52.81%；用更强的
  Bond Object 替换 edge evidence 反而回退，说明边证据强度不是唯一瓶颈。
- v0.7.41 进一步联合监督并逐层修正 edge logits，端到端 role exact/skeleton 提高到
  59.44%/54.62%，仍未过正式门槛。它减少漏点但显著增加位于真实键线内部的假点；
  下一步转向端点射线终止/穿越证据，不继续单独加 edge loss 或训练步数。
- v0.7.42 在候选端点内外采样短射线，3px 线内假点从 162 降到 107，但漏点从
  411 增到 472，role exact/skeleton 为 59.24%/53.82%；机制有效但默认 NO-GO。
- v0.7.43 加入对向强边和墨线穿越的节点摘要，找回部分漏点，但最终只追平
  59.64%/54.62%。下一步使用可学习二维端点 patch，不继续叠加手工统计量。
- v0.8.0 完成 ChemScene v0.1 对象协议和确定性 SDF 编译。validation 500 张中
  498 张无损往返、2 张同坐标隔离；首个 3,000 张训练分片中 2,983 张无损往返，
  8 张源描绘冲突、9 张同坐标隔离，未知失败为 0。下一步训练 Bond Query，不再从
  大量像素点候选开始重建结构。
- v0.8.1 否决跨图归零的自由 Bond Query，改用空间中心、无向角度和长度表达整根
  直线对象。100 图 SDF exact 达到 98%；9k/498 object accuracy 达到 67.55%，
  校准 SDF exact 为 2.61%。方向成立但当前不可部署，下一步做多尺度中心检测和
  Junction Assembler。
- v0.8.2 加入零初始化 stride-8/16 上下文，长键和双键显著改善，object accuracy
  达到 70.66%，SDF exact 达到 4.42%。v0.8.3 的纯二维线带分类器将局部 object
  accuracy 提高到 71.22%，但 SDF exact 回退到 2.61%，因此不替换多尺度类型头；
  v0.8.4 的 learned Junction-Bond Assembler 将局部 pair exact 提高到 98.18%，但
  端到端 exact 为 19.08%，仍低于 nearest 的 19.48%。v0.8.5 图级 type-only
  修正首次净增 1 张完整正确结构，exact 达到 19.68%，但仍只作为实验分支。
- v0.8.29 的纯全局 Bond Group Query 在 holdout 上只有 1% cell exact，证明它会记忆
  训练组而缺少局部视觉证据。v0.8.30 的 endpoint patch 将 holdout 提到 16%；
  v0.8.31 的 stride-4 特征线带没有继续改善泛化。
- v0.8.32 改为读取原图 `25×64` 定向墨线带，只训练 243,812 个局部参数，将 holdout
  cell exact 提到 18%、collision bond recall 提到 43.69%，并保持 98.79% protected
  retention。18 个正确 cell 全部来自 occupancy 2，occupancy 3 仍为 0/6。全部 refined
  candidates 的 collision recall 为 43.69%，旧 presence 的 known-count top-k 只保留
  9.71%，因此下一步冻结 geometry、单独训练 selector；稳定 v0.7.10 不变。
- v0.8.33 冻结 v0.8.32 的 vision、dense detector 和全部 geometry，只训练 108,289
  个 presence selector 参数。capacity 4k 的 representable group recall 为 99.43%、
  exactly-one 为 98.89%，通过容量门控。正式 safe-select 将 collision known-count recall
  从 9.71% 提高到 25.73%，正确 cell 从 2/100 提高到 6/100；ordinary geometry recall
  提高 0.32pp，image exact 仅回退 0.20pp，冻结权重逐位不变，正式判定排序方向 GO。
  本轮没有训练可部署 count 或键型，也未接入稳定推理；下一步独立训练 representable
  count/解码，再运行完整 ChemScene、SDF 有效率和化学一致性评测。

当前架构与结果见 [ChemVector 中文主文档](docs/model/CHEMVECTOR.md)，每次训练的
完整命令、指标和决策见 [独立训练档案](docs/model/training/README.md)。

## 目录

```text
chemvector/       模型、训练、推理和评测 Python 包
experiments/      自动实验 Campaign 配置与有限假设队列
Data/sdf/         ChemMol/ChemSVG 协议与数据构建工具
Data/images/      手工外部测试图片
Data/MR-TADF_SDF/ MR-TADF 外部测试集及渲染白名单
docs/             项目文档索引与模型说明
tests/            ChemVector 与数据协议测试
tools/            项目维护和目录整理工具
outputs/          checkpoint、评测和单图预测产物
paper/            本地论文资料，不进入 Git
third_party/      上游参考代码与版本说明
```

`outputs/` 已按用途分为：

- `outputs/checkpoints/`
- `outputs/evaluations/`
- `outputs/predictions/`
- `outputs/diagnostics/`

历史路径映射见 `outputs/MANIFEST.csv`。新实验不得再直接写入 `outputs/` 根目录。

## 常用命令

运行测试：

```bash
.venv-train/bin/python -m unittest discover -v
```

CPU 推理：

```bash
.venv-train/bin/python -m chemvector.infer_parallel_v6 \
  --checkpoint outputs/checkpoints/chemvector-v5-vertex-10k/last.pt \
  --image Data/images/GD906.png \
  --output-dir outputs/predictions/chemvector-v6-parallel-cpu-gd906 \
  --device cpu --cpu-threads 16
```

MR-TADF 渲染前审计：

```bash
.venv-train/bin/python -m Data.sdf.audit_mr_tadf
```

构建并评测 MR-TADF v1 benchmark：

```bash
.venv-train/bin/python -m Data.sdf.build_mr_tadf_benchmark \
  --output-dir Data/benchmarks/mr_tadf_v1

.venv-train/bin/python -m chemvector.benchmark_mr_tadf \
  --checkpoint outputs/checkpoints/chemvector-v5-vertex-10k/last.pt \
  --benchmark-manifest Data/benchmarks/mr_tadf_v1/manifest.csv \
  --output-dir outputs/evaluations/mr-tadf-v1-v071 \
  --device cuda
```

查看所有 benchmark 的真值/预测图片和 SDF：

```bash
.venv-train/bin/python frontend/server.py
```

浏览器打开 `http://127.0.0.1:8765`。页面会自动扫描普通评测和 Oracle 评测结果。
v0.7.6 结果已支持按 OCR、漏/多原子、漏/多键和交叉错连筛选。

v0.7.11 实验性 runtime 由 base checkpoint 和 packed bundle 组成：

```text
base:   outputs/checkpoints/chemvector-v079-relation-all-pairs-9k/last.pt
packed: outputs/checkpoints/chemvector-v0711-conservative-moe-packed/last.pt
```

packed bundle 为 983 KB，只保存 28 个变化的结构头 tensor 和 gate。该组合已判定
NO-GO，仅保留用于复现实验，不是当前正式推理候选。

生成完整分子 OCR 字符替换预览：

```bash
.venv-train/bin/python -m chemvector.replacement_ocr \
  --samples 16 \
  --output-dir outputs/diagnostics/replacement-ocr-preview
```

替换标签全部按普通字符串训练，覆盖 118 个元素符号、带氢/电荷/同位素写法、
`A/X/R1` 等占位符和 `Ph/Me/tBu/Boc` 等化学简写。该阶段不做化学理解。

完整文档入口见 [docs/README.md](docs/README.md)。

Benchmark 前端会显示每次运行的模型版本、checkpoint 和推理协议，并支持上传单张
结构图生成预测 SDF。启动方式和公网安全边界见 [前端说明](frontend/README.md)。
