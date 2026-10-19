# 当前状态

核验日期：2026-10-19（Asia/Shanghai）。此页记录核验范围，不代表所有历史实验已复核。

## 项目与模型

| 项目 | 最近核验状态 | 依据 |
| --- | --- | --- |
| GitHub | `wxyhgk/chempic2sdf` 已公开，main 已上传项目快照及首页整理 | [当日记录](2026/10/19/README.md) |
| 默认推理配置 | `structure-v01010-direct-aromatic-ocr`，注册表 recommended=true | [注册表](../../frontend/inference_profiles.json) |
| 冻结研究参照 | `structure-v01016-structural-twin-ocr`，非默认 | [冻结说明](../model/baselines/20260914-frozen-reference-and-exploration.md) |
| 模型方向 | 自研分子图路线，文字和几何分别提供证据 | [职责说明](../model/20260910-graph-first-contract.md) |
| 公开交付 | 源码、配置、测试和记录；无完整训练数据及 checkpoint | [项目首页](../../README.md) |

这里的“默认”指配置文件设置，不表示核验了线上部署或端到端服务健康。

## 本轮完成

- 整理首页，明确路线差异、设计强项、实际限制与安装使用条件。
- 首页列出的两组组件测试合计 6 passed，前端 CLI 帮助命令通过；这不是模型准确率评测。
- 建立按天拆分的小型 CSV 台账、每日说明和技术决策模板。

## 待核验

本轮只整理文档，未启动训练、未修改权重或推理配置。后续研究任务开始前，需根据原始报告
核实最近一次训练／探针的完成状态、数据身份、预算与采用结论，并检查是否有未结束进程。
旧实验文件名的日期不用于推断本日运行情况，也不在此补写未经核实的最新准确率。

下一步见 [推进计划](ROADMAP.md)。
