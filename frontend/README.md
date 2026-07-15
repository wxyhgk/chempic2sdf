# Benchmark 前端

该前端直接读取 `outputs/evaluations/` 下的评测记录，左右对照真值 SDF 和模型
预测 SDF。所有结构图都从对应 SDF 渲染，不使用 SMILES 重建预测结果。

启动：

```bash
.venv-train/bin/python frontend/server.py
```

服务默认监听 `0.0.0.0:8765`。本机浏览器打开 `http://127.0.0.1:8765`，公网访问
使用 `http://<服务器公网 IP>:8765`。需要确保云安全组和主机防火墙允许 TCP
`8765`；该服务没有身份认证，只应在可信网络或外部反向代理鉴权后开放。

页面支持：

- 切换不同 benchmark 运行；
- 按完全一致、原子/键数一致、数量不同和无效 SDF 筛选；
- 按键长难度和分子 ID 搜索；
- 按 OCR、漏/多原子、漏/多键、交叉错连和无效原因筛选；
- 左右查看真值图和预测图；
- 左右查看或下载完整 SDF；
- 查看原子数、键数、分子式、指纹相似度和最大公共子结构覆盖率；
- 查看失败原因、原始 ChemSVG 和修复后 ChemSVG。

已有 benchmark 如果还没有图错误字段，可先回填：

```bash
.venv-train/bin/python -m chemvector.benchmark_diagnostics \
  outputs/evaluations/<运行目录>
```

诊断器使用原子像素坐标和 ChemSVG 图元直接比较，不使用 SMILES 判定局部错误。

Oracle 评测目录与普通评测目录使用相同协议，因此会自动出现在“评测运行”列表中。
Oracle 模式只在评测时替换指定真值，不参与模型训练。

预测 SDF 必须由评测器直接保存在评测目录的 `sdf/` 中。历史运行如果没有保存
预测 SDF，页面会明确显示“未保存预测 SDF”，不会通过 SMILES 补造结果。
