# MR-TADF 外部测试集

本目录中的 SDF 原文件由数据提供方放入，原文件不修改、不覆盖。该数据集当前只
用于模型能力检验，不加入训练集。

## 渲染前硬门禁

任何 PNG 生成流程都必须读取 `audit/approved.csv`，禁止直接遍历本目录中的
`*.sdf`。下列结构不会进入图片生成：

- 无法读取、空结构或 RDKit sanitize 失败；
- 盐、混合物等多组分结构；
- 缺少唯一二维构象，或任一原子的 Z 坐标不为零；
- 与现有 18 万训练分子存在结构或 Murcko scaffold 重叠；
- InChIKey 重复的后续副本。

MR-TADF 中的超长键不是异常，不按键长过滤。审计文件会记录最大键长和相对中位
键长的比例，供后续按难度分层评测。

重新执行审计：

```bash
.venv-train/bin/python -m Data.sdf.audit_mr_tadf
```

当前审计结果：

- 原始候选：1,120 个；
- 唯一且可用于渲染：1,100 个；
- 拒绝：20 个；
- 原始 SDF 目录中的 PNG：0 个；
- 与现有训练集的结构和 scaffold 重叠：0。

详细名单见 `audit/approved.csv`、`audit/rejected.csv` 和
`audit/summary.json`。`.ipynb_checkpoints/` 等隐藏子目录永远不参与审计和渲染。

正式 PNG 不写入本目录，而是由白名单构建到
`Data/benchmarks/mr_tadf_v1/`。构建和评测协议见
[`docs/benchmarks/MR_TADF_V1.md`](../../docs/benchmarks/MR_TADF_V1.md)。
