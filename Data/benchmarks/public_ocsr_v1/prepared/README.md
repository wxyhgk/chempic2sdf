# Prepared 评测区

本目录保存评测器直接读取的图片和 ground-truth SDF。图片必须与 raw 官方图片
逐字节一致；只有结构容器允许在无损往返验证后转换为单分子 SDF。

统一命名：

```text
prepared/<subset>/images/<sample_id>.<原扩展名>
prepared/<subset>/ground_truth/<sample_id>.sdf
```

`uspto_10k_abb/source_structures/` 额外保存与每个 prepared SDF 对应的官方结构
文件，便于检查 alias、S-group 和 superatom 信息是否丢失。

当前第一版随机公开基线位于 `uspto_clean_1k/`。它由
`../build_uspto_clean_1k.py` 使用固定 seed 从 clean 子集构建，详细协议见
`../../../../docs/benchmarks/USPTO_CLEAN_1K.md`。
