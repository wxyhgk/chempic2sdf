# 给数据人员的执行任务

请在以下目录完成 ChemVision Layers v0.1 第一批数据：

```text
/home/zhenghe/GK/Project/chempic2sdf/Data/sdf/chemvision_layers_v0_1/pilot_10k
```

执行前必须完整阅读同级上方的 [`README.md`](README.md)，并使用：

```text
Data/sdf/webdataset_full_v0
```

作为唯一的分子、split、scaffold 和 ChemMol 来源。

第一批交付固定为 2,500 个分子、10,000 张图片。每张图必须从同一次语义
SVG 渲染中生成 combined、text、bond 和 semantic 四层，不能从 PNG 反推
文字和键线。第一批只做清晰黑白图和字号/字体变化，不添加模糊、透明背景、
扫描噪声或 JPEG 压缩。

完成标志：

```text
pilot_10k/validation.json
```

存在并且 `status` 为 `passed`。如果有任何失败样本，必须写入
`pilot_10k/rejected.csv`，不要静默替换，也不要启动 `full/`。
