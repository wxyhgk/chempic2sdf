# Benchmark 前端

该前端直接读取 `outputs/evaluations/` 下的评测记录，左右对照真值 SDF 和模型
预测 SDF。所有结构图都从对应 SDF 渲染，不使用 SMILES 重建预测结果。

启动：

```bash
.venv-train/bin/python frontend/server.py
```

服务默认监听 `0.0.0.0:8765`。本机浏览器打开 `http://127.0.0.1:8765`，公网访问
使用 `http://<服务器公网 IP>:8765`。需要确保云安全组和主机防火墙允许 TCP
`8765`；该服务没有身份认证，而且上传会触发 GPU/CPU 推理，只应在可信网络或
外部反向代理完成鉴权和限流后开放。

页面支持：

- 切换不同 benchmark 运行；
- 明确显示运行对应的模型版本、checkpoint 和推理协议；
- 选择或直接从剪贴板粘贴 PNG/JPEG/WebP/TIFF 图片并生成预测 SDF；
- 按完全一致、原子/键数一致、数量不同和无效 SDF 筛选；
- 按键长难度和分子 ID 搜索；
- 按 OCR、漏/多原子、漏/多键、交叉错连和无效原因筛选；
- 左右查看真值图和预测图；
- 左右查看或下载完整 SDF；
- 查看原子数、键数、分子式、指纹相似度和最大公共子结构覆盖率；
- 查看失败原因、原始 ChemSVG 和修复后 ChemSVG。

## 上传测试

页面右上角“上传测试”只使用服务端预定义的推理 profile，浏览器不能提交任意
checkpoint 路径或命令。当前默认 profile 是 `structure-v0992-carbon`。它按顺序执行
v0.9.83 AtomPair group、v0.9.87 图上下文、v0.9.89 整图 one-move 和 v0.9.92 候选级
Chemical RETYPE verifier。verifier 只允许对一条已存在键改型，不增加或删除连接。它默认关闭
v0.9.84/85 长键 rescue，而且 OCR 尚未接入，所有文字原子仍按 C 占位。因此这里的
“推荐”仅表示当前结构识别主线，不表示已经具备完整元素、基团或分子式识别能力。

历史稳定 profile `stable-v0710` 和实验性 `v0.8.3`、
`v0.8.5`、`v0.8.6`、`v0.8.8`、`v0.8.9`、`v0.8.16` 和 `v0.8.22` ChemScene hybrid
可手动选择。
新增 `v0.9.67 结构实验` 上传 profile，接入全 9k AtomObject 与当前
Bond/Incidence 路线。该 profile 尚未接开放 OCR，所有文字原子明确按 C 占位，只用于观察
顶点、键和连接关系，不能把输出元素或分子式视为完整化学识别结果。由于完整 SDF 图回归
未通过，它不是默认推荐模型；该实验完成时的稳定 profile 仍为 v0.7.10。
`v0.9.72 联合扩量结构实验` 使用 full-v0 的 30k 联合训练与 5k Bond presence 校准。
它同样把文字原子按 C 占位，且 MR-TADF 外部结构 exact 低于 v0.9.65，因此只用于对照，
不标记为推荐模型。
`v0.9.85 一等 Line-Only AtomPair 结构实验` 允许全图线段在旧 group 缺失时创建新的
Bond proposal。它不使用绝对键长，仍把双键/三键作为两个语义端点，并继续把文字原子按 C
占位。该 profile 用于观察长键和漏键修复，未标记为推荐模型。
`v0.9.86 AtomPair 整图上下文结构实验` 回到普通结构主线，同时建模一条候选键、两端邻接
候选和整图集合。默认不启用 v0.9.84/85 长键 rescue；OCR 仍未接入，因此文字原子继续按 C
占位。该 profile 已通过双域 locked internal test，但仍不是完整 OCR 推荐模型。
`v0.9.87 AtomPair 正 Margin 图上下文` 在 v0.9.86 上增加真键绝对正 margin，并使用 validation
固定的 residual scale 0.8。它继续默认关闭长键 rescue，在 locked internal test 的两个普通
结构域均有净提升；OCR 未接入，仍按 C 占位且不标记为完整模型推荐项。
`v0.9.89 AtomPair 集合级 One-Move` 在 v0.9.87 结果上联合比较整图编辑动作。双域
validation 的 presence/chemical exact 从 `710/527` 提高到 `720/532`，固定 checkpoint 和
margin 后，locked internal test 从 `717/523` 提高到 `730/527`。两个域的 presence 都有净
提升，因此它成为当前默认结构 profile。

`structure-v0103-direct-atom-pair-carbon` 是 v0.10.3 的非默认 shadow。它冻结最终
AtomObject 后枚举全部无序 AtomPair，直接判断无键或四种化学键，不使用最近邻和键长过滤。
独立 fresh 1k 的 Bond exact 为 45.3%，但文字原子仍按 C 占位，且尚未完成公开集和 MR-TADF
逐图回归，因此只供上传诊断，不替换 `structure-v0992-carbon`。

`structure-v0104-direct-aromatic-carbon` 在 v0.10.3 相同 checkpoint 和骨架上增加完整环语义：
只允许满足锁定整环证据的 5/6 元环整体变为 aromatic，不逐键修补。fresh 1k chemical Bond
exact 从 18.3% 提高到 28.2%，presence 和 skeleton 不变。它同样尚未接 OCR，保持非默认。
`v0.9.92 候选级 Chemical RETYPE` 冻结 v0.9.89 的 presence，只对具体“某条边改成目标
键型”的动作训练整图 exact verifier。validation chemical exact `532→537`，locked internal
test `527→533`，internal 两域分别 `239→244` 和 `288→289`，presence 完全不变。该 profile
替换 v0.9.89 成为当前默认结构 profile；OCR 仍未接入，文字原子继续按 C 占位。

运行时模型注册表位于 `frontend/inference_profiles.json`。服务端每次读取 profile API、
浏览器每次打开上传面板时都会重新加载该文件，因此新增或更新 profile 后不需要重启
前端服务，也不需要手动刷新网页。只有修改 `frontend/server.py` 本身时才需要重启。
`v0.9.28 Object Shadow` 是历史完整对象 shadow：固定八个 checkpoint 及 SHA，
直接输出 ChemScene、有效或 diagnostic SDF 和对应渲染图。它用于真实图片试用和公开 benchmark，
现在保留为对照，不再是默认推荐 profile。
固定 USPTO clean 100 的结果为 SDF valid 77%、非立体结构完全一致 21%、化学完全一致 19%。
无效预测会用非 sanitize 模式生成诊断图，并继续明确标记为无效。
存在 `object_decomposition.json` 的新 benchmark 会额外显示纯连接骨架、原子类型+骨架、
键级+骨架和完整非立体图。v0.9.29 的 USPTO clean 1k 结果分别为 72.4%、29.2%、50.5% 和 21.2%。
页面同时显示拓扑正确条件下的单原子类型和单键级 micro accuracy，避免把整分子 exact 当成
单对象准确率；当前 1k 分别为 96.45% 和 89.04%。
v0.8.9 将原子 anchor 与文字 bbox 中心解耦，并用保守双 crop 防止域外简单字符回退；
真实门控图仍有 NH 后缀幻觉，因此不推荐替换稳定模型。
v0.8.16 从每个完整 Bond Object 的两个语义端点聚合 Junction，OCR 只能附着标签，
不能再把双键副线或字符笔画变成结构顶点；交叉和近邻区域的假 Bond 仍未解决。
v0.8.22 在 top-128 候选上使用 keep/drop selector，再用方向对齐 endpoint patch 聚合
Junction。无效价态和 selector 空输出都会保留 diagnostic SDF，不再只显示异常文本。

命令行直接运行当前 Object Shadow：

```bash
.venv-train/bin/python -m chemvector.infer_shadow_v0928 \
  --image Data/images/GD906.png \
  --output-dir outputs/manual/gd906-v0928
```

每次上传会独立保存在：

```text
outputs/uploads/<时间戳-ID>/
```

目录中包含上传原图、768×768 预处理图、预测 SDF、SDF 渲染图、原始/修复后
ChemSVG 或 ChemScene JSON、模型版本和耗时指标。上传大小上限为 20 MB，图片边长
上限为 12000px。新版还会显示芳香降级、自环和重复边诊断。
上传入口保留 PNG/WebP 的透明层，再由统一预处理按白底合成，确保与命令行读取
同一图片时结果一致。

默认使用 CUDA 推理。测试 CPU 部署时使用：

```bash
.venv-train/bin/python frontend/server.py --inference-device cpu
```

已有 benchmark 如果还没有图错误字段，可先回填：

```bash
.venv-train/bin/python -m chemvector.benchmark_diagnostics \
  outputs/evaluations/<运行目录>
```

诊断器使用原子像素坐标和 ChemSVG 图元直接比较，不使用 SMILES 判定局部错误。

Oracle 评测目录与普通评测目录使用相同协议，因此会自动出现在“评测运行”列表中。
Oracle 模式只在评测时替换指定真值，不参与模型训练。

预测 SDF 必须由评测器直接保存在评测目录的 `sdf/` 中。历史运行如果没有保存
预测 SDF，页面会明确显示“未保存预测 SDF”，不会通过 SMILES 补造结果。
