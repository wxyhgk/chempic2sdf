# 清单文件字段

所有 CSV 使用 UTF-8、逗号分隔、第一行为表头。字符串中含逗号时必须使用
标准 CSV 引号，不允许手工拼接。

## samples.csv

```text
sample_key,molecule_id,split,variant,source_sample_key,source_shard,output_shard,font_family_id,font_size,font_bucket,bond_line_width,rotation_degrees,stress_operation,combined_sha256,text_sha256,bond_sha256,semantic_sha256,layers_sha256
```

`font_bucket` 只能是 `small`、`normal`、`large`、`extreme`。

## molecules.csv

```text
molecule_id,source,source_structure,scaffold_id,split,image_count,visible_atom_count,bond_count
```

## fonts.csv

```text
font_family_id,font_family,font_style,font_file,font_file_sha256,license,license_url,split
```

同一字体家族的所有 style 必须使用同一个 split。

## rejected.csv

```text
source_sample_key,molecule_id,split,variant,stage,error_code,error_message
```

推荐固定的 `stage`：`select`、`layout`、`svg_render`、`svg_parse`、
`rasterize`、`label_build`、`roundtrip`、`validation`。

错误信息不得只保存 Python traceback；必须同时提供稳定的 `error_code`，方便
汇总失败类型。
