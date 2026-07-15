# Manifest 字段协议

三个子集 CSV 固定使用以下字段和顺序：

```text
sample_id,
dataset,
subset,
official_split,
source_record_id,
image_file,
sdf_file,
source_structure_file,
source_structure_format,
source_revision,
image_sha256,
sdf_sha256,
source_structure_sha256,
width,
height,
formula,
atoms,
bonds,
fragments,
has_abbreviation,
has_charge,
has_isotope,
has_stereo,
status,
notes
```

实际 CSV 表头写成一行，不要保留上面的换行。

## 字段说明

- `sample_id`：项目内稳定 ID，建议 `<subset>_<官方ID>`。
- `dataset`：固定为 `uspto_30k`。
- `subset`：`uspto_10k`、`uspto_10k_l` 或 `uspto_10k_abb`。
- `official_split`：保持官方定义；如果官方没有 split，写 `benchmark`。
- `source_record_id`：官方数据中的原始 ID，不得重新编号替代。
- `image_file`：相对 `public_ocsr_v1/` 的图片路径。
- `sdf_file`：ready 时填写 ground-truth SDF；否则留空。
- `source_structure_file`：raw 官方结构文件或可审计导出文件的相对路径。
- `source_structure_format`：例如 `mol_v2000`、`mol_v3000`、`sdf`。
- `source_revision`：与 `sources/uspto_30k.json` 一致的 commit SHA。
- 三个 SHA256：小写 64 位十六进制。
- `formula/atoms/bonds/fragments`：从准备后的 SDF 读取；不可读时留空。
- `has_abbreviation`：官方结构是否含 alias、S-group、superatom 或非元素标签。
- `has_charge/has_isotope/has_stereo`：按结构真值统计。
- `status`：只允许 `ready`、`rejected`、`pending_superatom_protocol`、
  `blocked_missing_official_structure_file`。
- `notes`：简短说明，正常 ready 样本留空。

正式评测只读取 `status=ready`，但汇总必须以官方总记录数为分母报告准备覆盖率。
