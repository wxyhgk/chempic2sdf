# OCR 教师模型

ChemVector 的最终目标仍是单模型端到端推理。PaddleOCR 只在开发和数据准备阶段
作为补充教师模型，为真实论文图片和非标准缩写生成文字伪标签。现有合成分子图
已经有 ChemMol 元素真值和 Render 锚点，不应再调用 PaddleOCR 重复标注。

当前使用独立环境中的 PaddleOCR 3.7.0：

```text
PP-OCRv5_mobile_det
en_PP-OCRv5_mobile_rec
```

通用文字检测器不适合直接寻找嵌在化学键线中的单字符。实际流程由几何分支或
Render JSON 提供文字锚点，再对锚点周围的小图执行 PP-OCRv5 识别。

## 输入格式

```json
{
  "anchors": [
    {"id": "atom-1", "x": 262, "y": 257},
    {"id": "atom-2", "x": 390, "y": 257}
  ]
}
```

## 运行

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  .venv-ocr/bin/python -m chemvector.ocr_teacher \
  --image Data/GD906.png \
  --anchors-json outputs/ocr-teacher/gd906-anchors.json \
  --output outputs/ocr-teacher/gd906.json
```

输出保留 OCR 原始文字、置信度和锚点坐标。不要在教师输出阶段擅自删除 `+`、
`-`、数字或括号，因为它们可能是形式电荷、同位素、重复数或缩写的一部分。

正式模型将使用这些标签训练内部字符 decoder。推理时不依赖 PaddleOCR。

主数据中完全不存在的字符不能由教师模型凭空补出。`R1/Ph/Me/A/X` 等覆盖先由
`chemvector.synthetic_ocr` 的在线局部图课程提供，再使用真实论文 crop 缩小字体
和绘图风格差异。
