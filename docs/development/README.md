# 开发记录

按实际工作日期归档，时区统一为 `Asia/Shanghai`。CSV 保存简短台账，Markdown
解释背景、证据、验证和下一步。已有实验报告保留原位置，通过链接引用。

## 入口

- [当前状态](STATUS.md)：最近核验的默认配置、已完成工作与待核验问题。
- [推进计划](ROADMAP.md)：跨天任务的当前状态、优先级与验收条件。
- [2026-10-19](2026/10/19/README.md)：建立 GitHub 仓库、整理首页与开发记录。
- [技术决策](decisions/README.md)：长期有效的选择、依据与重新考虑条件。
- [每日说明模板](templates/daily.md)、[决策模板](templates/decision.md)。

## 每日目录

```text
YYYY/MM/DD/
  README.md         当天目标、结果、验证、问题与下一步
  changes.csv       当天代码、数据、配置或文档改动
  tasks.csv         当天任务进展
  experiments.csv   当天实际执行的实验索引；没有实验就不创建
```

每日说明从 [daily.md](templates/daily.md) 复制，CSV 从 `templates/` 中对应的
表头文件复制。仅创建实际需要的表，不填虚构的实验或占位数据行。

## 表结构

| 文件 | 字段 |
| --- | --- |
| `changes.csv` | id、module、summary、commit、record |
| `tasks.csv` | id、module、priority、status、summary、record |
| `experiments.csv` | id、module、status、conclusion、summary、report |

`module` 使用 atoms、bonds、ocr、data、integration、engineering 或 docs。
任务优先级为 P0/P1/P2；任务状态为 planned/in_progress/blocked/done/cancelled。
实验状态为 running/completed/failed/interrupted；实验结论为 pending/GO/NO-GO/inconclusive。
执行完成与结果通过分开记录：completed + NO-GO 是有效结果，不代表运行失败。

## 维护约定

1. 每个 CSV 最多 50 条数据行、当前模板最多 6 列，字段用短句。
   同一天超出上限时按专题拆成 `experiments-presence.csv` 等；专题内仍超限则加 `-02` 分片。
2. ID 首次创建后保持不变：任务 `T-YYYYMMDD-01`、改动 `C-YYYYMMDD-01`、
   实验 `E-YYYYMMDD-01`、决策 `D-YYYYMMDD-01`。同一类型跨专题及分片统一编号。
   一次真实运行也可直接沿用既有唯一 run ID，不另造一个不相干的身份。
3. 跨天任务沿用原 ID，每天仅记录当天进展；同一天每个任务只保留一行。
   ROADMAP 维护当前状态，历史日记录保留当时状态，禁止跨天复制实验并冒充新运行。
4. `record` 和 `report` 使用相对于所在 CSV 的仓库文件路径，支持 `#锚点`；
   commit 使用实际提交 SHA，尚未提交可留空，提交后可在后续记录中补充关联。
5. 参数、数据身份、评测范围、指标、预算及实际用量写进详细报告，沿用
   [实验模板](../model/training/TEMPLATE.md)。台账只写简短结论与链接。
6. 计划、实现、验证和默认配置更新分别记述。未运行写“未运行”，未核验写“待核验”；
   文件名中的日期、遗留文档的“当前”和运行进程名均不单独构成完成证据。
7. 历史事实需纠正时，在原说明中追加日期、原因和新证据链接；保留原结论及其评测条件。
   不修改旧指标来适配新结论。
8. CSV 使用 UTF-8、标准逗号分隔；含逗号或双引号的字段按 CSV 规则引用，避免多行单元格。
   公开记录只引用可公开信息，凭据、受限样本与封存内容不写入文档。

每轮有实质进展后更新当日说明与相关台账；项目现状或下一步发生变化时，同步更新
STATUS 和 ROADMAP。README 首页只提供入口，不追加日常流水账。
