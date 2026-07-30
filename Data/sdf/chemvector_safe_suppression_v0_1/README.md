# ChemVector Safe Suppression v0.1 数据交付协议

## 用途

本目录为 ChemVector `v0.10.13` 的 safe-suppression action 提供全新、未见过的
route-select 和 route-confirm 数据。该版本不再学习“两条 proposal 是否匹配同一个 GT
原子”，而是学习一个方向明确的动作：删除某个冗余 AtomObject 后，是否完整保留原子与键
覆盖。

现有 `fresh_dev` 的 route-select 和 route-confirm 已被 `v0.10.12` 消费，不能在
`v0.10.13` 中重新用于选模、调阈值或证明泛化。旧 `sealed_test` 继续保持 `000`，本版本
不得读取。

## 一、交付规模

每个分子生成 `original / rdkit / coordgen / stress` 四张 768x768 PNG：

| split | 分子数 | 图片数 | 用途与权限 |
|---|---:|---:|---|
| `route_select` | 2,000 | 8,000 | 可见，只用于 checkpoint 和 action threshold 选择 |
| `route_confirm` | 10,500 | 42,000 | 初始权限 `000`，只运行一次冻结 winner |
| 合计 | 12,500 | 50,000 | molecule 和 scaffold 跨 split 隔离 |

42,000 张 confirm 图按 `v0.10.12` 的动作密度预计产生约 3,100 次实际删除。最终至少需要
2,995 次删除且零有害事件，才能让单侧 95% Clopper-Pearson safe-action precision 下界
达到 99.9%。若实际动作不足 2,995 次，只能判为证据不足，不能通过部署门。

## 二、分子来源和隔离

从尚未使用的 ChEMBL 37 记录中确定性选择，不得从当前训练包或 benchmark 重采样。新数据
必须在 molecule ID、完整 InChIKey 和 Bemis-Murcko scaffold 三个层面同时与以下集合零
重叠：

- 所有历史 train、validation、internal test 和 fresh-dev；
- 旧 fresh 的 sealed 集合；该项由数据人员内部比对，只公开重叠数 `0`，不得暴露 ID；
- USPTO、MR-TADF、MolGrapher 及其他公开或内部 benchmark；
- `route_select` 与 `route_confirm` 彼此。

仅保留模型当前支持的单组分有机结构。每个分子必须通过 RDKit sanitize、规范化 SDF 往返
和 InChIKey 复核。不得引入盐、混合物、聚合物、反应图或金属配位结构。

重原子数按 `8-20 / 21-40 / 41-60 / 61-80` 四桶分层，并记录环数、稠环、桥环、手性中心
及 B/N/S/Se 等元素分布。不得针对 MR-TADF 骨架、长键或特定错误样本采样。

## 三、通用分布和近邻安全桶

- 75% 分子保持 ChEMBL 合格池的自然随机分布；
- 25% 分子属于通用 `close_distinct` 桶：四种渲染中至少一种存在两个不同 GT 原子，其像素
  anchor 距离 `<=16px`；
- `close_distinct` 只按渲染后的几何统计选择，不修改分子、坐标或图片；
- 两个 split 内均保持 `general / close_distinct` 比例，并分别报告验收指标；
- 四种 layout 数量严格完整，不添加模糊、JPEG、透明背景或文档噪声。

字体 family 在 `route_select` 和 `route_confirm` 之间隔离，各自至少包含 4 个未用于旧训练
的 family。黑白图不少于 60%，其余可沿用有限元素色和灰度；完整记录字体、字号、线宽、
多键间距、旋转和颜色参数。

## 四、样本协议

每张图片至少包含以下 WebDataset 成员：

```text
sample.png
sample.chemmol.json
sample.render.json
sample.layers.json
sample.sdf
sample.meta.json
```

必须保留：

- 稳定 `atom_id`、元素、形式电荷和像素 anchor；
- `TEXT / IMPLICIT` role、文字 bbox 和可见字符串；
- 稳定 `bond_id`、两端 atom ID、display type 和 chemical type；
- molecule ID、InChIKey、scaffold、variant 和全部 style 参数；
- 每个成员的 SHA256。

数据人员不需要生成“删除节点后的图片”，也不运行任何模型。反事实 action 标签由模型侧在
数据 commitment 固定后，用冻结的 `v0.10.10` 组件离线生成。

## 五、Commitment 和权限

模型人员第一次读取 `route_select` 图片前，必须固定：

- `samples.csv`、molecule 集合、字体分组、schema 和所有 tar 的 SHA256；
- 两个 split 各自的 molecule ID、InChIKey 和 scaffold set SHA256；
- source SDF SHA256、生成器 Git commit、RDKit 版本和随机 seed；
- general/close_distinct、重原子数、layout、字体、线宽、多键间距和颜色精确计数；
- 与所有旧数据及两个 split 间的三类 overlap 均为 `0`；
- 每个分子四图完整、tar 成员可读、PNG 解码、化学往返、前景 IoU 和重复生成 hash 验收。

`route_confirm` 的 manifest、tar、图片和 SDF 必须全部置为 `000`。根 commitment 仅公开
总样本数、集合 hash、tar hash 和 aggregate validation；不得生成 confirm preview。

只有模型侧写出以下冻结信息后，才允许临时开放 confirm 并运行一次：

```text
candidate checkpoint path and SHA256
selected step and action threshold
code commit
data commitment SHA256
action-cache commitment SHA256
route-select metrics
fixed inference arguments
```

## 六、模型侧反事实 Action Cache

该部分由模型人员执行，不属于数据人员生成任务。冻结 `v0.10.10` 产生 proposal 后，每张图
只建立一个全局 `KEEP`，并为具有 `<=16px` 近邻证据的每个节点建立唯一删除动作：

```text
KEEP
DELETE_NODE_<proposal_id>
```

`DELETE_LEFT / DELETE_RIGHT` 只可作为 pair 交换时的序列化表示，进入训练前必须按实际被删
node 去重。同一节点由多个近邻 pair 提议时，应汇聚全部 nearby witnesses，不能重复生成动作
或重复增加 loss 权重。

每张图推理最多执行一次删除，避免两个单独安全的动作组合后共同删掉最后一个正确节点。
删除节点时同时删除其入射预测边，不重新运行视觉编码器或 repair。

删除前后均使用带 dummy 和禁止边的 float64 稳定 Hungarian：先最大化 6px 内的 GT 匹配
数，再最小化总距离。不能先做全距离 Hungarian 再过滤 6px。还需与更宽、拓扑感知的匹配
口径交叉复核；两种口径对动作安全性结论不一致或存在无法消解的最优匹配歧义时，标签记为
`IGNORE`，不能强标 safe。每个动作记录：

- 匹配到的 GT atom-ID 集合变化；
- 可表达 GT bond-ID 集合变化；
- 已部署 Direct AtomPair 覆盖的 GT bond-ID 集合变化；
- 键型正确的 typed GT bond-ID 集合变化；
- Atom FP/FN、Bond FP/FN、Atom exact 和 Skeleton exact 变化；
- OCR、chemical type、芳香编译和 SDF valid 变化。

`safe=true` 至少同时满足：原有 GT atom-ID、expressible/covered/typed GT bond-ID 集合均
不丢失，Atom/Skeleton/SDF-valid 不从真变假，invalid/self/duplicate/component error 不
增加。集合必须按稳定 ID 比较，不能只比较数量。安全动作再按以下顺序形成 set-valued 最优
标签，并列动作都视为正标签：

```text
Skeleton exact
Atom exact
-(Atom FP + Atom FN)
-(Bond FP + Bond FN)
最少编辑
```

Action cache 必须单独生成 commitment，包含原始数据 commitment、所有基础 checkpoint
SHA256、代码 commit、匹配协议、固定阈值、16px 候选半径、各类动作数和 cache tar SHA256。
GT 和 utility 字段只能进入离线 label builder，forward batch 必须显式剔除并由测试验证。

模型输出必须区分保留端与删除端，不能继续用对称 pair 分数后按 raw proposal score 决定
survivor。建议使用 `P(safe)` 和 ordinal utility 双头：utility 依次区分 safe-neutral、减少
FP/invalid、Atom exact rescue、Skeleton/typed/SDF rescue。训练目标是每图一个 KEEP 与全部
唯一 DELETE action 的全局 set/listwise 竞争，不能对每个 pair 独立做三分类。safe-neutral
只监督 safety，不得作为“应删除”正类；只有相对 KEEP 有严格收益的 safe rescue 才能进入
部署候选。推理先通过预注册安全阈值，再要求 utility 严格优于 KEEP。删除后必须重建
surviving pair index、邻接和 degree，去除 self/duplicate edge，并重新运行 ring/aromatic
compiler 与 RDKit sanitize。

## 七、预注册训练和门控

- 训练仍用现有三域 `4:1:1`，最多 50,000 张图片曝光；新 confirm 不进入训练；
- 训练前先在训练/route-select 上运行 best-single-delete oracle；若 Atom 和 Skeleton 上限无法
  同时达到预注册 `+3.0pp`，立即停止本路线，不得用堆训练量掩盖不可达目标；
- 固定保存 `10k / 20k / 30k / 40k / 50k`，unsafe checkpoint 不得覆盖 safe winner；
- route-select 排序第一关键字是 safety eligible，其后依次为 Skeleton exact、Atom exact、
  safe-action recall、更保守阈值和更早 checkpoint；
- 冻结唯一 winner 后写 `candidate_manifest.json`；confirm 禁止运行 runner-up、重选阈值或
  同版本重训。

Route-select 资格：

- lost GT atom/bond coverage、protected distinct deletion、Atom break、Skeleton break 均为 0；
- Skeleton exact 相对冻结基线至少 `+2.5pp`；
- safe-action recall 至少 25%。

Route-confirm 最终 GO 门：

- 全局 winner 实际执行的删除至少 2,995 次；不得用全部过阈候选数代替；
- harmful deletion、lost GT atom IDs、lost expressible/covered/typed GT bond IDs、
  protected deletion、Atom/Skeleton/SDF-valid break、结构错误增加全部为 0；
- 单侧 95% safe-action precision 下界至少 99.9%；
- Skeleton exact 和 Atom exact 均至少 `+3.0pp`；
- SDF valid 与 chemical exact 不低于冻结 `v0.10.10` 基线；
- `general` 与 `close_distinct` 两桶分别通过零伤害门。

任一条件失败即 `v0.10.13 NO-GO`，不得在读取 confirm 后降低门槛。后续修改必须升级为
`v0.10.14`，本次 confirm 永久降级为已消费 dev。无论 GO/NO-GO，正式训练完成后都必须先
提交代码和中文训练记录，再创建不可移动 annotated tag `v0.10.13`；NO-GO 不切换默认前端。

## 八、模型侧必测

在正式训练前至少增加以下测试：

- pair 交换时 `DELETE_LEFT / DELETE_RIGHT` 互换，全局 KEEP 不变；
- 同一个 removed node 的多个 witness 合并为一个动作；
- unmatched 冗余节点的有收益删除是 safe rescue；
- 两个真实近邻的两个删除方向均为 unsafe；
- 一侧独占关键 Bond 时只允许删除另一侧；
- 三节点重复簇仍然每图最多执行一个动作；
- safe-neutral 不会击败 KEEP；
- label cache 重复生成逐字节一致；
- forward 输入不含 GT ID、action utility 或其他标签泄漏字段；
- 删除后芳香编译、价态检查和 SDF sanitize 确实重新运行。

## 九、完成标志

目录至少包含：

```text
README.md
summary.json
validation.json
commitment.json
molecules.csv
samples.csv
schemas/
route_select/shards/
route_select/previews/index.html
route_confirm/shards/
```

`validation.json` 完成状态必须满足：

```json
{
  "status": "passed",
  "molecules": 12500,
  "route_select_images": 8000,
  "route_confirm_images": 42000,
  "old_dataset_molecule_overlap": 0,
  "old_dataset_inchikey_overlap": 0,
  "old_dataset_scaffold_overlap": 0,
  "cross_split_overlap": 0,
  "route_confirm_mode": "000"
}
```
