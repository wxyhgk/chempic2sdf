# 纯合成 fixtures

这里不含分子、元素、SDF、GT、真实 benchmark 图片或样本 ID。

`build_fixtures.py` 用标准库确定性生成一张 768×768 灰度 PNG：白色背景上只有一条水平黑线。合法
prediction 将该线分成两个具名 segment、两个 single-bond lane 和三个 terminal，用于验证 RLE、前景
引用、端点、BondBundle 和 interpretation ownership 的结构合同。`synthetic_double.png`、
`valid_double_inbox.json` 与 `valid_double_prediction.json` 额外提供一个两条 support 完全不相交、
centerline separation 恰好位于冻结边界的合法双键正例。

重新生成：

```bash
cd /home/zhenghe/GK/Project/chempic2sdf
.venv-train/bin/python formal/chemgraph/audit/blind_carrier_v0_1/fixtures/build_fixtures.py
```
