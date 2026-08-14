# v1 P2-D2-E3B Style Endpoint Slot 容量审计

> 日期：2026-08-14
>
> 状态：已完成，路由到 slots=4
>
> 审计：`chemvector-v1p2d2e3b-style-endpoint-slot-capacity`

## 目标

E3 固定三 slot endpoint 模型在训练超过 38,400 张后，遇到同一 stride-4 cell 内 4 个不同
visible ports 并中止。本轮对 E3A 的 99,336 张安全样本全量统计 target cell occupancy，决定
固定 slots=4 是否足够，不能通过删除合法拥挤结构继续训练。

## 固定协议

- safe manifest SHA256：
  `ee41d57275d8c48ebe9461b69eeef5bbb90abaa10de9319b746f8274024686e6`；
- 对每个 ChemScene v0.2 BondObject 的两个 visible ports 去重；
- 按 P2-B 完全相同的 `192x192` 网格映射统计每格唯一 port 数；
- 分 fit/calibration/holdout、variant 和 stress operation 报告；
- 不训练模型，不生成 PNG，不读取 fresh_dev 或 sealed_test。

## 路由规则

- 最大 occupancy 不超过 3：`current_slots_sufficient`，说明 E3 实现有误；
- 最大 occupancy 等于 4：`increase_slots_to_4`，下一模型显式扩为四 slot；
- 最大 occupancy 超过 4：`dynamic_cell_set_required`，不能继续固定 slot 架构。

无论超容量样本多少，都不允许把它们从 safe manifest 删除。

## 执行命令

```bash
.venv-train/bin/python -m \
  chemvector.evaluation.audits.bonds.audit_v1p2d2e3b_style_endpoint_slot_capacity
```

## 实际结果

- 99,336 张全部完成统计，共 3,659,071 个 unique visible ports；
- 97,170 张最大 occupancy=1，2,071 张为 2，92 张为 3，3 张为 4；
- 最大 occupancy 严格为 4，没有 5+；
- 超三 slot：fit 2 张、holdout 1 张、calibration 0 张；
- 三张分别为两个 `crossed_bonds` stress 和一个 rdkit 布局；
- 不生成 PNG，fresh_dev/sealed test 读取数均为 0。

按预注册规则判定 `increase_slots_to_4`。下一模型保留全部三张合法样本，将 endpoint 输出从
3 slot 扩为 4；共享 backbone 与前三个 slot 从 P2-B checkpoint 初始化，新增第四 slot 随机
初始化。不能继续使用 slots=3，也没有证据支持改成动态 cell set。

## 产物

- audit：`outputs/audits/chemvector-v1p2d2e3b-style-endpoint-slot-capacity/audit.json`；
- 超容量样本：同目录 `over_capacity.jsonl`。

SHA256：

- `audit.json`：`90f3c5511382ddce2751d211dfdb335aa4caf0e40a073297707524be6dd3ab1c`；
- `over_capacity.jsonl`：`cb584aed68326875f7a37285c35217325bb210d7fe860795225d47fe246f1da9`。
