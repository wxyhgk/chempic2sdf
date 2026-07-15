# 来源记录

每个公开数据集一个 JSON 文件。第一阶段只创建 `uspto_30k.json`。

推荐结构：

```json
{
  "schema_version": "public-ocsr-source-0.1",
  "dataset_id": "ds4sd/USPTO-30K",
  "source_url": "https://huggingface.co/datasets/ds4sd/USPTO-30K",
  "revision": "必须填写具体 commit SHA",
  "downloaded_at": "ISO-8601 时间",
  "download_tool": "实际工具与版本",
  "license_from_dataset_card": "原样记录",
  "official_subsets": {},
  "field_mapping": {},
  "files": [
    {
      "path": "raw/uspto_30k/...",
      "bytes": 0,
      "sha256": "..."
    }
  ]
}
```

许可证字段只能抄录 dataset card，不得根据其他项目的许可证自行推断。
