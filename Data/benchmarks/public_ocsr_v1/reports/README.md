# 报告与验收

必须交付：

```text
summary.json
validation.json
rejected.csv
duplicates.csv
previews/index.html
```

## summary.json 最低内容

- dataset ID 和固定 revision；
- 三个子集的官方数量、ready、rejected、pending 数量；
- 图片格式和尺寸分布；
- 原子数、键数、片段数分布；
- 电荷、同位素、立体化学、缩写样本数量；
- SDF V2000/V3000 数量；
- 结构往返成功率；
- 与训练数据和其他 benchmark 的重叠统计；
- 总磁盘占用和准备耗时。

## rejected.csv 字段

```text
dataset,subset,source_record_id,image_file,source_structure_file,stage,error
```

没有失败也必须保留表头。

## duplicates.csv 字段

```text
overlap_type,left_dataset,left_subset,left_id,right_dataset,right_subset,right_id,structure_key
```

`structure_key` 只用于审计，不作为正式评测标签。

## previews

随机抽取至少 500 个样本，页面左右显示：

- 官方原始图片；
- ground-truth SDF 渲染图；
- sample ID、子集、原子数、键数和状态；
- 原图和 SDF 下载链接。

缩写样本还要显示官方 source structure 文本，便于人工检查信息是否丢失。
