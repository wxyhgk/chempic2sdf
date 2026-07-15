# pilot_10k 交付区

数据人员先在这里交付 10,000 张图片，不要直接启动全量生成。

完成后本目录应包含：

```text
shards/
previews/index.html
samples.csv
molecules.csv
fonts.csv
rejected.csv
summary.json
validation.json
```

开始前请完整阅读上一级 [`README.md`](../README.md)。只有
`validation.json.status == "passed"` 且模型人员完成 Oracle semantic layer
实验后，才讨论 `full/`。
