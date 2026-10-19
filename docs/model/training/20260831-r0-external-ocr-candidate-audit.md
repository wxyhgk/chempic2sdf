# 2026-08-31 R0 外部 OCR 候选覆盖审计

> 状态：COMPLETE，natural fresh-dev `NO-GO`  
> 训练授权：NO  
> 在线替换授权：NO

## 问题

当前 v0.10.9 byte CTC 在 fresh evaluation 上的 TEXT raw 为 93.13%。历史
PP-OCRv5 mobile pilot 的四尺度 any-exact Oracle 只有 22.22%，但该 pilot 使用原子中心固定
方形 crop 并拉伸到 192x96，不能代表 v0.10.8 predicted bbox 的部署输入。

本轮只回答一个问题：在完全不训练、不路由、不替换的条件下，通用外部 recognizer 是否能
为 v0.10.9 greedy 与 prefix top-8 的去重并集增加足够多的正确原始字符串候选。

## 冻结协议

- 数据只使用 `chemvector_joint_v1/fresh_eval_12k/fresh_dev`；禁止读取 sealed test、
  MR-TADF 和 USPTO。
- 按 molecule_id 的固定 SHA256 顺序选 500 个 molecule，每个 molecule 保留四种 variant。
- crop 只使用冻结 v0.10.5 role gate 和 v0.10.8 predicted bbox。
- 所有 arm 读取同一份 192x48、等比 letterbox、4px context PNG cache。
- arm 固定为 v0.10.9 greedy、v0.10.9 prefix beam top-8、PP-OCRv5 mobile top-1、
  PP-OCRv5 server top-1。
- 比较只使用 raw UTF-8 exact membership；不同模型的 confidence 不可比较。
- 第一轮只报告 candidate union Oracle，不执行 AtomObject 修改或 SDF 替换。

## 门槛

natural fresh-dev 只把实际存在的 `simple_element / modifier` 作为充分性分组，两组任一少于
200 TextObject 时结果为 `INCONCLUSIVE`。`open_string` 继续描述性报告，但不阻断 natural
结论；它必须由 `Data/ocr/open_text_v0_1` 的 molecule-disjoint validation/internal_test
另行确认，不能用自然总体里不存在的样本凑门槛。组数充分时，同时满足以下条件才允许把
外部 OCR 保留为 natural fresh-dev 候选源：

1. 相对 v0.10.9 greedy 与 prefix top-8 的去重并集，overall candidate recall 增加至少
   2.0pp，或者 modifier/open-string 任一增加至少 5.0pp；
2. 外部模型新增正确候选覆盖至少 1.0% 的 matched TextObject。

未通过即停止通用 PaddleOCR 路线。通过只授权完成独立 Open Text challenge，再设计 R1
全图融合 confirm；不授权在线替换，也不代表 `Ph/Me/R1/tBu/Boc` 等开放字符串已经通过。

## 入口

```bash
.venv-train/bin/python -m chemvector.audit_external_ocr_candidates select ...
.venv-train/bin/python -m chemvector.audit_external_ocr_candidates prepare ...
.venv-train/bin/python -m chemvector.audit_external_ocr_candidates recognize --arm ctc_v0109_greedy ...
.venv-train/bin/python -m chemvector.audit_external_ocr_candidates recognize --arm ctc_v0109_top8 ...
.venv-ocr/bin/python -m chemvector.audit_external_ocr_candidates recognize --arm ppocrv5_mobile ...
.venv-ocr/bin/python -m chemvector.audit_external_ocr_candidates recognize --arm ppocrv5_server ...
.venv-train/bin/python -m chemvector.audit_external_ocr_candidates reveal-truth ...
.venv-train/bin/python -m chemvector.audit_external_ocr_candidates summarize ...
```

每个 recognizer 只读取 `crops.jsonl` 和 crop PNG。四个候选 bundle 冻结后，`reveal-truth`
才读取 GT 并生成 `truth.jsonl`；最终 `summarize` 生成逐对象 joined records 和能力报告。

## 正式结果

正式选择为 500 个 molecule、2,000 张图。predicted producer 生成 79,804 个 AtomObject，
其中 21,943 个进入 TEXT crop。GT reveal 得到 22,285 个可见文字对象，21,291 个匹配到
冻结 crop，TEXT detection recall 为 95.54%；另有 652 个 predicted negative crop。

| 候选源 | raw exact |
|---|---:|
| v0.10.9 greedy | 93.52% |
| incumbent greedy + top-8 | 99.00% |
| PP-OCRv5 mobile top-1 | 24.07% |
| PP-OCRv5 server top-1 | 32.69% |
| Paddle mobile + server | 40.44% |
| incumbent + Paddle | 99.19% |

Paddle 只在 21,291 个 matched TextObject 中新增 39 个正确候选，增量覆盖 0.183%，overall
增加 0.183pp；modifier 增加 0.144pp。三项都未达到 1% / 2pp / 5pp 门槛。同时 Paddle
union 在 652 个 negative crop 中有 540 个非空输出，候选数均值从 8.00 增到 9.22。

因此通用 PP-OCRv5 不保留为 natural fresh-dev 候选源，不进入 router、fusion 或在线替换。
当前更大的问题是 994 个 GT TextObject 没有进入冻结 crop；后续优先改进结构对象、语义墨迹、
TEXT role 与 bbox，而不是替换 recognizer。natural fresh-dev 没有 open string，本结论不外推到
`Ph/Me/R1/tBu/Boc`；若后续测试真正的化学专用预训练 OCR，应使用独立 Open Text challenge。

冻结 artifact：

- crop manifest SHA256：`1bfe3dcb36e8f034cf933e940b24c1dffaf74c639bc7de3af1d3272ea34a0a29`；
- truth manifest SHA256：`d063dd9c4bcf339c377799aab3b4647ee3fc9cd89e67d04e684bd183d7df87c7`；
- report SHA256：`73cb158d8c9e5264155b3afa22d5d7d993e46b253c051ae4e87d47bdf1455f18`。
