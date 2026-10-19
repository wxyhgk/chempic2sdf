# 2026-09-14：冻结自研参照，继续有限探索

## 冻结决定

冻结的是一份可复核的研究基线，不是停止项目，也不是将当前架构认定为最终方案。

- 研究参照：`structure-v01016-structural-twin-ocr`，原 200 图非立体正确 59/200。
- 部署默认保持：`structure-v01010-direct-aromatic-ocr`，同批 55/200。
- 两者仍保留，不更换默认，不训练、不调阈值、不继续覆盖原 checkpoint。
- 已反复使用的 50 个分子、200 图只是开发对照；不能称为独立泛化或上线确认。

快照目标：`outputs/baselines/chemvector-v01016-reference-20260914/`。
保留测量绑定的代码、全部 10 个依赖 checkpoint、两个 profile 的配置与实测输出、
严格 ID 参考及真值控制证据。原 200 PNG 也在测量工件中保留。使用独立副本，不用硬链接。

工作区已有未提交改动，不能只靠 Git HEAD 代表此次基线。快照记录 HEAD 作为上下文，
以实际文件 SHA 为代码身份；不会清理、提交、回滚用户改动，也不会创建正式发布 tag。

两个原始数据 shard 只保存路径与 SHA，不再复制；Python 包版本记录在 manifest 中，
虚拟环境、系统 CUDA 驱动不打包。因此这是代码/权重/测量快照，不是独立可启动容器。
哈希校验不等于数字签名或操作系统禁止写入，未来实验不得原地修改快照。

```bash
.venv-train/bin/python tools/freeze_chemvector_baseline.py create --output-dir outputs/baselines/chemvector-v01016-reference-20260914
.venv-train/bin/python tools/freeze_chemvector_baseline.py verify --output-dir outputs/baselines/chemvector-v01016-reference-20260914
```

`create` 拒绝既有目录。`verify` 只检查副本，不要求当前工作区继续保持原样；
重新运行完整实验仍需匹配的数据 shard 和兼容环境。

## 下一步探索范围

暂不继续增加自研网络、修补规则或候选筛选训练。先寻找可靠的完整预训练 OCSR 参照，
目标是选择基座，而不是继续在 MoE、循环、原子优先、键优先之间猜测。

已复核的仓库现状：

- `benchmark_public_ocsr.py` 实际运行 ChemVector，不是第三方完整模型排行榜。
- `audit_molgrapher.py` 审计合成数据/关键点；许多 MolGrapher 字样的 checkpoint 是
  ChemVector 的风格适配结果，不是原版 MolGrapher 预训练推理成绩。
- 历史调研文档曾记录本地 MolGrapher 源码复核，但当前 `third_party/` 只见
  StarVector；不能把旧文档里的“本地可用”当作现在仍可运行。
- 当前 `.venv-train` 未安装 molscribe、molgrapher、DECIMER，也未安装 timm。
- 已检查的推理入口、依赖清单、评测 summary/report 与文献记录中，没有找到
  同一 200 图上完整预训练 OCSR 系统的可复核成绩；这是本次检查范围内的缺口，
  不声称穷尽了机器的全部历史文件或环境。

下一项只做一个完整系统的官方预训练推理对照，优先核验 MolScribe 的官方权重、
许可证与接口可用性。先用隔离依赖，原 `.venv-train` 不安装或降级第三方依赖。
选择它只是建立参照，不是宣称其性能优于自研或其架构应照搬。

对照必须保留第三方原始输出，记录预处理、源码/权重版本；统一评测完整化学图，
失败不剔除。若只有 SMILES，转换 SDF 时记录坐标是否重新生成，不能冒充原图定位能力。
原 200 图先用于工程/开发比较；是否扩大到未见评估要另行固定协议，不能据本批宣称上线。

只有完整系统对照结果出来，才决定继续哪个基座、哪些错误值得自研。
本轮冻结不授权自动开展大规模训练、评测封存数据或修改线上默认。
