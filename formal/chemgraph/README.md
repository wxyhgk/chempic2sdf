# ChemGraph 形式化工程

这个目录用 Lean 4 检查“图片到化学图”路线中最基础的数学前提。它不是模型实现，也不以通过几个定理来替代真实 benchmark。

本轮机器验收见 `validation.json`。当前状态是 `passed_finite_abstract_foundation`，其中 `training_authorized=false`。

## 当前结论

第一版先回答唯一性问题。把不同语义对象的墨迹直接做像素并集后，中心四价原子的十字和两根互不相连键的交叉可以得到完全相同的观测，但两者化学图不同。因此：

```text
存在 s1 != s2，使 Render(s1) = Render(s2) 且 Graph(s1) != Graph(s2)
=> 不存在对所有场景都正确的确定性 Decode
```

这个结论与参数量、训练图片数量和优化器无关。遇到这种样本只能：

1. 从渲染中保留更多可观察证据；
2. 用明确先验缩小允许场景集合；
3. 输出多候选或“不可判定”；
4. 从 strict exact 测试集剔除，并保留冲突审计记录。

`Recovery.lean` 还验证了一个很小的 Oracle 结果：对于只含“十字交叉或中心 junction”的有限语法，如果外部已经给出每个语义 Bond 的墨迹分组，查表解码器能够区分两者。这里的 `BondGroupingOracle` 已包含关键语义分组，不是模型可直接读取的图片观测，也不是恢复成功的证明。

相反，`owned_observation_not_factor_through_opaque` 证明：在当前不透明四臂模型中，不存在一个只读 `Ink` 的确定性函数，能够同时为 crossing 和 junction 生成正确分组。`grouping_oracle_not_sufficient_for_general_graph` 又证明，仅有分组仍会忘掉键级，不能恢复一般 ChemicalGraph。Two-Port 路线若要成立，必须从真实最终 PNG 中找到这个四臂抽象尚未表达的额外可观察证据，并显式预测 order、可见端口和文字实例；否则必须登记为歧义。

## 文件说明

- `ChemGraph/Scene.lean`：化学图、语义 Bond 和十字墨迹的有限模型。
- `ChemGraph/Render.lean`：丢弃所有权的普通渲染，以及保留 Bond 分组的观测规格。
- `ChemGraph/Identifiability.lean`：可辨识性、先验消歧、左逆充分条件和碰撞不可能定理。
- `ChemGraph/Stability.lean`：异图最小间隔、噪声半径与唯一解码定理。
- `ChemGraph/Grammar.lean`：精确线段几何、三层 grammar 信号与分类定理。
- `ChemGraph/Counterexamples.lean`：crossing 与 junction 的机器检查反例。
- `ChemGraph/Multiplicity.lean`：双键与两根平行单键的墨迹分组反例。
- `ChemGraph/Recovery.lean`：受限十字语法上的恢复定理。
- `ChemGraph/FiniteAudit.lean`：最小可执行碰撞分类器和 strict/ambiguity 回归测试。

## 工具链与构建

工程固定 Lean `v4.33.0`，版本记录在 `lean-toolchain`，不跟随浮动版本。官方 Linux 工具链压缩包大小为 `574882764` 字节，官方 SHA256 为 `4b3fb03c29a1e0a253fb1d11f9bae3725f19a0dc6fc09b3ea16d2c9df3349e2c`。

```bash
source "$HOME/.elan/env"
cd /home/zhenghe/GK/Project/chempic2sdf/formal/chemgraph
lake build
```

## 数学门

在训练 Two-Port Ink Field 前，形式化和可执行验证必须依次通过：

1. 定义最终栅格输入上的等价关系，而不只比较矢量源文件。
2. 自动枚举空图、文字、单/双/三键、共线链、fan、环、交叉和 junction。
3. 找出所有“同一最终输入、非同构化学图”的碰撞并移入 ambiguity 集。
4. 在 strict 集上证明 Oracle 中间表示到 production decoder 为左逆。
5. 对端口交换、Bond 顺序、平移、旋转和缩放验证不变性或等变性。
6. 由未参与实现的 reviewer 使用未知 seed 连续审批两轮。

## 稳定恢复条件

仅仅没有发现完全相同的输入还不够。`Stability.lean` 把不同图的渲染当作观测空间中的码字。令异图之间的最小距离为 `Δ`，实际输入扰动上界为 `ε`；只有满足：

```text
2 * ε < Δ
```

时，Lean 定理 `robust_ball_has_unique_graph` 才保证同一个噪声球内的全部合法候选属于同一化学图。后续 production tensor 审计必须估计每个样本族的 `Δ`，并用小于 `Δ/2` 的拒识半径；否则只能输出多解或不确定，不能把“hash 没撞”误称为稳定。

当前 Lean 模型只证明有限十字抽象中的逻辑事实。`graphEquivalentBool` 和 `GraphEquivalent` 已忽略 Bond 列表排列，但 Atom 编号目前仍由有限场景的空间角色规范化；一般图同构尚未形式化。当前还只检查精确碰撞，没有定义观测度量、异图最小间隔或拒识半径，所以不能把有限扰动未碰撞误称为数学稳定。真实抗锯齿 PNG、文字遮挡、短键、双键 lane、虚线和任意分子的恢复也尚未证明。因此当前状态仍是 `FORMALIZATION_IN_PROGRESS`，不能据此开始大规模训练。

## 参考原则

- 逆问题必须先检查解的唯一性和稳定性；只有“存在预测函数”不够。
- 若渲染映射在语义图上不是单射，增加训练数据不能恢复已经被观测丢掉的信息。
- 图上的 crossing 和 vertex 是不同的组合结构；像素相交本身不能定义 Atom。
- Lean 官方文档：<https://lean-lang.org/documentation/>
- Arridge 等关于数据驱动逆问题的综述：<https://doi.org/10.1017/S0962492919000059>
- 关于多笔画图像中歧义区分析的工作：<https://doi.org/10.1016/j.eswa.2016.08.017>
