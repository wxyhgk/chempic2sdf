# v1 P0 Edge-centric Oracle10k

> 日期：2026-08-14
>
> 状态：NO-GO；对象协议通过，原始数据覆盖门差 3 张
>
> 审计：`chemvector-v1p0-edge-centric-oracle10k`

## 目标

停止 P19 relation router 支线，正式切换到 edge-centric ChemScene v1 主线。首轮不训练模型，
先冻结旧端到端 benchmark，并证明目标对象协议可以无损表达现有 10k 分层数据。Oracle 不通过
时只修协议，不允许用神经网络掩盖标签或编译错误。

## 旧路线基线

固定已完成的 MR-TADF v1 1,100 张 v0.10.21 结果，不重新选择模型或阈值：

- SDF valid：725/1,100；
- Skeleton exact：529/1,100（48.09%）；
- Atom-typed exact：311/1,100；
- Bond-typed exact：361/1,100；
- Chemical exact：264/1,100（24.00%）。

summary、object decomposition、records 与 benchmark manifest 必须逐项匹配已冻结 SHA256。

## v1 对象定义

- `AtomObject`：一个化学原子，包含 chemical anchor、可选 TextObject 和全部 incident ports；
- `BondObject`：一根化学键，固定拥有两个 port、display type 和 chemical type；
- 双键、三键和芳香键各自仍是一个 BondObject，内部墨线不生成额外 Atom；
- 隐式原子由 Bond endpoint 等价类产生，显式文字原子通过文字区域边界 port 接入；
- 不使用键长上限、最近邻候选或全 AtomPair 枚举。

当前协议复用已经过局部验证的 ChemScene v0.2；v1 指视觉模型和训练目标的新代际，不重复定义
不兼容的 SDF/JSON 格式。

## Oracle10k

全量读取 `chemvision_layers_v0_1/pilot_10k` 的 5 个 tar、10,000 张图，仅读取
ChemMol/Render/Layers JSON，不生成或复制 PNG。逐样本验证：

1. chemical atom、layer atom、AtomObject 一一对应；
2. chemical bond、layer bond、BondObject 一一对应；
3. 每个 BondObject 恰有两个分别归属端点 AtomObject 的 port；
4. visible text 与 TEXT AtomObject 数量一致；
5. SINGLE/DOUBLE/TRIPLE/AROMATIC 的对象计数与 layer 一致；
6. ChemScene parse、SDF 生成、canonical isomeric SMILES/InChIKey 往返通过。

重叠原子坐标和 depiction conflict 只能确定性跳过并计数，不生成 PNG，也不进入后续训练。

## GO/NO-GO

- 输入严格为 10,000 张，Oracle 覆盖率至少 99.5%；
- 除预定义 collision/conflict 外零失败；
- 所有 accepted 样本六项不变量和化学往返 100% 通过；
- 四布局、四 stress operation、双键和三键均有覆盖；
- MR-TADF 基线文件和 1,100 条 records 完整。

GO 后下一周期直接构建 Text/Bond/Endpoint 语义张量和小型分层视觉头；NO-GO 则修协议，禁止
训练模型。本轮不读取 USPTO、sealed test 或新的 MR-TADF 输出，不创建模型 checkpoint。

## 实际结果

预注册提交为 `5869008`。全量运行 10,000 张耗时 82.21 秒，结果为：

| 指标 | 结果 |
|---|---:|
| 输入样本 | 10,000 |
| 完整通过 | 9,947 |
| Oracle 覆盖率 | 99.47% |
| 坐标碰撞 skip | 28 |
| depiction conflict skip | 25 |
| 未知失败 | 0 |
| AtomObject | 328,945 |
| BondObject | 356,702 |
| incident ports | 713,404 |
| DOUBLE / TRIPLE / AROMATIC BondObject | 55,198 / 729 / 70,420 |

所有 9,947 个 accepted 样本均通过对象数量、端口所有权、键类型和 SDF/InChIKey 往返；
四种布局、四种 stress operation、双键和三键均有覆盖，也没有生成 PNG。唯一失败门为覆盖率：
`99.47% < 99.50%`，即只差 3 张，但不得事后下调阈值。因此 P0 结论严格记为 NO-GO。

结果文件：

- `outputs/audits/chemvector-v1p0-edge-centric-oracle10k/result.json`，SHA256
  `2061c375971eac10cee49018d1fffedf18d66e57a41e361833b09139ceeaf143`；
- `outputs/audits/chemvector-v1p0-edge-centric-oracle10k/skip_diagnostics.json`，SHA256
  `86268c54faf905d5e359befab45da0ab25da51a73b00441335d6c1eefb241ceb`。

## 失败归因

53 张 skip 来自 38 个分子，不是未知的 ChemScene 解析失败：

- 33/53 是 stress 布局；其中 crossed-bonds 19、near-nonbonded 11、long-bond 2、
  crowded-subgraph 1；
- 坐标碰撞表示两个化学原子在 q1023 画布上落到同一坐标，图片无法提供两个独立端点；
- depiction conflict 表示变形后的视觉几何与源 E/Z 或手性标签冲突，尤其集中于
  crossed-bonds；
- 38 个分子中，26 个只坏一个布局、9 个坏两个布局、3 个坏三个布局。

这些图不应被模型学习，也不能靠直接使用源结构绕过视觉矛盾。下一周期建立分子级安全清单：
一个布局异常就让该分子的四个布局全部退出。预计保留 `2,462 × 4 = 9,848` 张；不补图、不
重新渲染，也不把原 P0 改判为 GO。安全清单 100% 往返通过后，才允许进入 v1 视觉容量实验。
