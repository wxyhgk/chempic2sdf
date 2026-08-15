# Sealed

当前目录必须保持为空，除本说明外不得创建任何 ID、GT 或 mapping。

实际执行时仅数据保管员可写入 `selection.json`、GT SDF、render annotation、per-bond layer 和
`causal_plan.json`。建议文件由独立保管账号持有、目录 `0700`、文件 `0400`，并通过 ACL 或隔离挂载使
模型运行员完全不可见。不要对仓库其他现有目录批量执行 `chmod`。

独立评估员只能在 predictions tree 冻结后获得只读访问。sealed 内容永远不得进入训练目录、模型容器、
日志、前端上传区或 proposer stdin。
