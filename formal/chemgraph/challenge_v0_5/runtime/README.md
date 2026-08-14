# P1 冻结运行时

正式 raster pilot 使用只读 Apptainer SIF，不读取宿主 Conda 的 Python/stdlib/sort：

```text
file   python-3.13.5-slim-bookworm.sif
sha256 f6c91d4743df17faa4c50a66f6cd9ae00cb5c9e9d265edfcab7759016b79dc60
python Python 3.13.5
zlib   1.2.13
sort   GNU coreutils 9.1
```

该 42 MB SIF 是外部冻结 artifact，不进入 Git；`p1_source_freeze_v0_5.json` 必须绑定它的绝对路径和
SHA256。正式只读安装路径为：

```text
/opt/chemgraph/runtime/f6c91d4743df17faa4c50a66f6cd9ae00cb5c9e9d265edfcab7759016b79dc60.sif
```

其父目录和文件均为 `root:root` 且不可写。原始 OCI 来源为 `python:3.13.5-slim-bookworm`，本机因 Docker Hub 直连超时，通过
`docker.1ms.run/library/python:3.13.5-slim-bookworm` 拉取并转换为 SIF。
