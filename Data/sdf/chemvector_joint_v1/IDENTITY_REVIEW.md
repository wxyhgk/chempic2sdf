# 只返回候选排除结果的数据侧核验

## 2026-09-15 扩量批次：新的单次授权与结果

用户明确授权“本批一次固定候选核验”。本次范围为 1,963 个分子 / 11,130 张唯一兼容候选图；原 287 候选的授权和凭据不被扩展或复用。新授权记录见 `docs/model/training/releases/20260915-expansion-identity-authorization.json`。

请求 SHA256 为 `4eb4ec527147672bb6eaa3a770d4bf5394a99927032fc9d5a0d65771724eb1c7`，候选文件 SHA256 为 `c0412d15c4d0651a0d316510b03030d5da7a603129db1990ef84b5f57dc6b1fe`。沿用下文相同的、哈希未改变的隔离实现，仅执行一次，无重试。

结果位于 `outputs/audits/chemvector-full-12k-private-identity-20260915/`：`checked_conservative`，全部评估身份覆盖、未解析 0，候选完整键 / 连接层 / 保守分组命中为 **0 / 0 / 1**。唯一命中的候选 `curr_2dfc9519ada8b515` 整体排除六张图；不是完全相同分子的命中。保留 **1,962 个分子 / 11,124 张候选图**，仍是 quarantine，没有训练许可。

数据侧只读身份清单，模型侧仅验证受限 receipt 并用自身候选清单统计排除数量。封存目录仍为 `000`，没有图片、SDF、tar 或 work 副本读取，没有权限/划分/模型变更。现存两条源分组冲突继续按并集保守排除，不修改标签；评估两部分旧连接层/骨架交集仍为 1 / 3。本次单次授权已使用，不自动再次调用隐藏核验服务。公开排除与来源用途审查仍待后续完成。

以下保留 2026-09-14 的流程和历史结果，数值仅适用于旧批次。

## 2026-09-14 原批次授权

2026-09-14，项目负责人在“允许建立独立数据侧排重流程，只返回重合候选与统计，不向模型侧开放隐藏测试内容”的询问后回复“开始”。本次据此执行**固定请求**的身份排重，不授权训练、解封模型侧访问、修改评估标签/划分或进行任意自适应候选查询。

原 README 的模型侧 sealed 访问门禁不变。本次唯一增量是独立数据侧进程能够只读处理身份清单，输出严格受限的统计/候选排除项。

## 固定输入与隔离边界

请求 SHA256：`b5b07ac382b6724342231bb79a012eb56311a07de317d09505cb7177b37e29d8`。

候选身份 SHA256：`ec0b92bc31fdc7a72b1262f516beb355d0ae009c85b61422a4f2fd79084bcfea`，287 个候选、2,513 张图。

评估 commitment：`13a456afb2c475a73fc4dca82df5fe62d4a31277d6b584637d9cd92736024512`。

数据侧读取的私有文件仅为 fresh-dev 和 sealed-test 各自的 `samples.csv`、`molecules.csv`。不会读取图片、SDF、tar、逐图推理结果或 work 中的副本。

执行器：`chemvector/training/curriculum/private_review.py`；数据侧 worker：本目录的 `identity_review_worker.py`。

1. 模型侧父进程只读取候选请求与公开代码；需要显式 `--authorized-identity-only`。这个参数记录操作意图，不是认证服务或权限凭证。
2. 已授权的本地 custodian 使用 sudo 打开固定清单的只读句柄，使用 `O_NOATIME`，不改原目录/文件权限。随后降回调用用户，再建立隔离进程。
3. bubblewrap 隔离 user/PID/network/IPC/UTS 命名空间并清空环境、移除全部 capabilities。身份清单通过匿名只读文件进入沙箱；运行环境只读。没有整个项目、home、模型权重或原始评估目录挂载。
4. worker 在读取私有内容前关闭普通 stdout/stderr；包括 RDKit/InChI 的原生日志。唯一结果管道只能返回预定 receipt。
5. custodian 和模型侧父进程分别验证字段白名单、计数、候选 ID 白名单以及请求/代码绑定。未知字段、私有 ID、异常原文和不一致计数一律变为固定失败码，不保存原始日志。
6. 最长数据侧 wall timeout 为 180 秒，CPU 上限 120 秒、地址空间 2 GiB；无 GPU。原始 `sealed_test` 目录仍为 `000`。

这是用户授权的本地进程隔离，**不是独立机构审计、远程认证或第三方签名证明**。不得把该工具当作允许无限查询隐藏集合的服务。

## 检查和保守排除规则

- 验证 commitment 自哈希、RDKit 版本和身份计算实现哈希。
- 两份 samples.csv 必须匹配已承诺的完整 SHA256 和字节数。
- 必须覆盖 8,000 / 4,000 张图、2,000 / 1,000 个分子，每个分子的四种变体完整，样本与分子行身份一致。
- molecule ID、InChIKey、legacy scaffold 三个集合必须匹配 commitment；逐分子由 canonical SMILES 重算完整 InChIKey 并核对。
- 按请求中的 `identities()` 重算连接层和忽略立体/同位素的 core/scaffold。
- 对已承诺的 legacy scaffold 也应用相同的去立体/同位素分组规则。与重算分组不一致时，**两者都加入排除集合**，不选择、修复或覆盖真值。候选只要命中任一分组即排除。
- 任一身份或旧骨架无法解析，记 unresolved，不能宣称完整负向排重。

`checked` 表示无覆盖缺口；`checked_conservative` 表示已覆盖，但保留源分组冲突并按并集排除；`incomplete` / `failed` 不能关闭身份覆盖缺口。任何状态的 `training_authorized` 均为 false，也都不是可交给训练 launcher 的 admission。

## 本次结果

最终凭据：`outputs/audits/chemvector-private-identity-review-20260914-conservative/receipt.json`。

| 项目 | 结果 |
| --- | ---: |
| 候选分子 / 图 | 287 / 2,513 |
| 评估分子 / 图 | 3,000 / 12,000 |
| 未解析身份 | 0 |
| 候选完整键 / 连接层 / 保守分组命中 | 0 / 0 / 0 |
| 旧骨架差异经既定去立体/同位素规则归一 | 33 条 |
| 仍不一致、保留两种分组参与排除 | 2 条 |
| fresh-dev 与 sealed 完整键 / 连接层 / 保守分组交集 | 0 / 1 / 3 |

最终状态为 **checked_conservative**。两条旧骨架冲突没有被“修复”，只完成了本次候选的保守排除检查。

新口径下评估两部分并非连接层/core/scaffold 完全独立。此结论不等于旧完整 InChIKey 协议出现相同结构泄漏，也没有改写旧 validation 的 passed 状态；未来按新规则报告验证/测试独立性前，必须由数据侧另行处理。

早期三份失败凭据及一份 incomplete 凭据保留在同名前缀的输出目录中，不能与最终凭据混用。最终执行记录绑定 runner、worker、请求和 receipt 的 SHA256；候选请求、评估原件与旧缓存均未重写。

## 验证与剩余工作

新增隔离核验测试及原请求/准入/发布回归共 **74 passed，9.30 秒，CPU**。包含真实 bubblewrap 对合成 `000` 权限清单的运行、秘密内容不会进入返回值/落盘、身份重算及源证据绑定、保留冲突分组共同排除、无法解析仍不完整、原文件权限不变。最终又核对了 receipt/执行代码/候选请求哈希，原 sealed 目录仍为 `000`。

该凭据仅关闭**本次候选对既定隐藏评估清单的保守身份排除**这一项。公开评测 13 条未解析记录尚未处理；完整分子 development validation 的数据接入及新口径下的评估分组独立性尚未完成。候选继续 quarantine，没有 optimizer、GPU 训练、识别准确率结果或前端变更。
