# 生成产物目录

`outputs/` 只保存模型训练和评测生成的产物，不保存源码或原始数据。

- `checkpoints/`：包含 `last.pt` 的训练 checkpoint 与训练配置。
- `evaluations/`：批量验证、盲测、外部测试和指标记录。
- `predictions/`：GD906、test1 等单张图片的推理结果。
- `diagnostics/`：smoke test、兼容性检查、OCR 教师输出和预览。
- `experiments/`：Campaign 状态、日志、门控结果、GPU 遥测和自动报告。
- `diagnostics/loose-files/`：早期实验遗留的单个兼容性检查文件。
- `MANIFEST.csv`：整理前后的完整路径映射。

新实验必须直接写入对应分类目录，不再向 `outputs/` 根目录增加实验文件夹。

重新整理顶层遗留目录：

```bash
.venv-train/bin/python tools/organize_project.py --apply
```

命令默认只预览迁移，加 `--apply` 才会执行。整理只移动生成产物，不删除 checkpoint
或评测结果。
