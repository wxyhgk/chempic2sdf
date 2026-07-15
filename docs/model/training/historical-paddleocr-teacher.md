# H01：历史 PaddleOCR 教师方案

> 状态：停止，仅用于复现

## 原方案

PaddleOCR 3.7.0 曾在开发阶段作为补充教师，为真实论文图片生成文字伪标签：

```text
PP-OCRv5_mobile_det
en_PP-OCRv5_mobile_rec
```

通用文字检测器不适合直接寻找嵌在化学键线中的单字符。实际流程必须由几何分支
或 Render JSON 提供锚点，再识别锚点周围的小图。

输入示例：

```json
{
  "anchors": [
    {"id": "atom-1", "x": 262, "y": 257},
    {"id": "atom-2", "x": 390, "y": 257}
  ]
}
```

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  .venv-ocr/bin/python -m chemvector.ocr_teacher \
  --image Data/images/GD906.png \
  --anchors-json outputs/diagnostics/ocr-teacher/gd906-anchors.json \
  --output outputs/diagnostics/ocr-teacher/gd906.json
```

输出保留原始文字、置信度和锚点坐标，不能擅自删除 `+`、`-`、数字或括号。

## 停止原因

- 通用 OCR 的文字先验与化学结构图不同，可能把键线误当字符笔画。
- 主数据中不存在的 `R1/Ph/Me/A/X` 无法由教师凭空补出。
- 数据已有精确锚点和原始结构，可以直接生成真值文字，不需要伪标签。
- 教师错误会被学生模型继承，并掩盖几何和 OCR 的真实误差来源。

## 当前替代方案

使用完整分子字符替换、开放 UTF-8 byte decoder、Pixel OCR 和 ChemVision 多尺度
ROI。推理不依赖 PaddleOCR，也不在 OCR 阶段执行化学理解。
