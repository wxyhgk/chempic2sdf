# 006：v0.6 无 decoder 并行 CPU 推理

> 状态：完成  
> 部署参数量：约 15.18M

## 动机

v0.5 的主要延迟来自自回归 decoder。GD906 视觉编码约 10ms，但 99 条边需要串行
生成约 992 个 token，decoder 耗时约 4.4 秒。v0.6 在部署路径删除 decoder、
token embedding 和序列投影，直接并行分类全部顶点对。

## CPU 推理

```bash
.venv-train/bin/python -m chemvector.infer_parallel_v6 \
  --checkpoint outputs/checkpoints/chemvector-v5-vertex-10k/last.pt \
  --image Data/images/GD906.png \
  --output-dir outputs/predictions/chemvector-v6-parallel-cpu-gd906 \
  --device cpu --cpu-threads 16
```

Ryzen 9 9950X 上 GD906 完整推理为 0.735 秒：顶点与 OCR 0.122 秒、3,570 个点对
分类 0.538 秒、墨线修复 0.075 秒。CUDA 并行版本约 0.389 秒。

## 500 张 scaffold 验证

- 平均推理：0.202 秒/张，比 v0.5 GPU 自回归快 6.6 倍。
- 完整拓扑：20.6%，v0.5 为 17.2%。
- 化学完全一致：13.2%，v0.5 为 12.4%。
- 顶点 precision/recall：99.01%/98.98%。
- 边 precision/recall：98.37%/98.19%。
- 断图失败从 120 降到 12。
- 价态冲突从 112 增加到 223。

结果位于：

```text
outputs/evaluations/chemvector-v6-parallel-cpu-validation-500/summary.json
```

## 决策

并行推理成为部署主线。后续需要导出只含前端的 checkpoint，并评估 ONNX、
OpenVINO 和 INT8；模型能力提升优先于当前速度优化。
