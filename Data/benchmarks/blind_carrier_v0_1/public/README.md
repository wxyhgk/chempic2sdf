# Public

这里只能放不泄露样本 ID、原始路径、SDF、annotation、GT 或秘密 seed 的公开 commitment。

预期文件包括 `selection_commitment.json`、`proposer_freeze.json` 和 `inbox_commitment.json`。
所有 JSON 必须 canonical、closed、包含完整源码/数据 tree framing 说明和
`training_authorized=false`。任何 proposer freeze 必须早于匿名 PNG 的发布。

本目录不能出现逐样本 source SHA 到 molecule ID 的映射。
