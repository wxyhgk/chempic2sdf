# Raster-neutral Evidence v0.3 预注册

日期：2026-08-15

状态：`preregistered_before_implementation`

本周期只验证下面这条有限命题，不恢复训练：

```text
完整 PNG bytes
  -> 隔离、确定性的中性视觉 Evidence
  -> 独立候选枚举
  -> Graph | Ambiguous | Reject
```

`training_authorized` 在本周期始终为 `false`。即使本周期全部通过，也只说明有限微型语法上的
可观察性和候选完整性边界成立，不代表真实化学图片已经可用。

## 1. 预注册问题

v0.2 已证明：相同 observation、不同 chemical graph 的场景不能被一个 sound 确定性解码器同时
正确解出。v0.3 不再尝试用全局 crossing 或 parallel prior 强行选一个解释，而是检验：

1. 是否能只读最终 PNG bytes，得到不含 Scene/Graph 真值的几何 Evidence；
2. 是否能从该 Evidence 完整列出有限语法允许的所有候选图；
3. 是否能对唯一候选输出 Graph，对异图多候选输出 Ambiguous，对域外输入输出 Reject；
4. 这一过程是否在未知几何变换和有限栅格扰动下稳定。

## 2. 冻结输入边界

extractor 必须作为隔离子进程运行：

```text
stdin:  一个完整 PNG 文件的原始 bytes
stdout: 一个 canonical JSON Evidence 文档
stderr: 诊断信息；正式运行必须为空
```

允许：Python 标准库、PNG chunk 解析、zlib 解压、确定性整数/有理数几何。

禁止：

- 导入仓库模块、Pillow、OpenCV、RDKit、Torch、NetworkX；
- 读取路径、目录、环境变量、网络、时钟、随机源或进程信息；
- 接收文件名、sample key、Scene ID、SDF、manifest、Render JSON 或 Graph 标签；
- 保存或查询 PNG SHA、像素摘要到 Scene/Graph 的表；
- 输出 Bond bundle ownership、semantic atom ID、crossing/junction 真值或 graph signature。

正式 runner 使用 `python -I -S`、清空后的环境和只含必要系统文件的工作目录。`strace` 审计必须
证明 extractor 在启动装载标准库以后没有打开输入/输出管道之外的数据文件，也没有网络调用。
源码静态审计和运行期系统调用审计必须同时通过。

## 3. Evidence v0.3 schema

Evidence 只描述可复算的墨迹几何：

```json
{
  "schema_version": "raster-neutral-evidence-v0.3",
  "image": {
    "width": 64,
    "height": 64,
    "color_type": "rgba8",
    "threshold_rule": "alpha_composite_white_luma_lt_128"
  },
  "foreground": {
    "row_runs": [[12, 8, 56]],
    "pixel_count": 147
  },
  "components": [
    {"id": 0, "bbox": [8, 11, 56, 13], "pixel_count": 147}
  ],
  "skeleton": {
    "pixels": [[8, 12], [9, 12]],
    "terminal_events": [[8, 12], [56, 12]],
    "branch_events": [],
    "intersection_regions": []
  },
  "lanes": [
    {"id": 0, "polyline": [[8, 12], [56, 12]], "support_pixels": 147}
  ],
  "unclassified_components": [],
  "text_mask": {"status": "not_inferred", "row_runs": []}
}
```

数组必须按公开的几何键排序，ID 只由排序后的数组位置决定。Evidence 可以保留完整前景 row-runs，
因为它们就是 observation 的无损几何编码；但不得只输出或消费其 digest。`text_mask=not_inferred`
表示本周期没有冒充解决 OCR/文字分割。出现文字式干扰时，允许保守 Reject，不允许借 GT mask 删除。

## 4. 候选枚举边界

candidate enumerator 与 extractor 分离，不能导入 extractor 源码。它可以读取：

- canonical Evidence JSON；
- 公开的有限 motif grammar；
- 本周期冻结的 renderer profile 参数范围。

它不能读取当前样本的 Scene、SDF、Graph、manifest 行或隐藏 seed。候选必须通过几何约束生成，或者
在运行时独立前向渲染整个有限 grammar 并与完整 foreground row-runs 比较；禁止 digest lookup。

候选集合按 graph equivalence 去重。对每个样本必须分别审计：

```text
soundness:    predicted_candidates <= oracle_graph_candidates
completeness: oracle_graph_candidates <= predicted_candidates
```

两者都必须是集合包含关系的逐样本 100%，不能用平均 F1、top-1 或“真值在候选中”替代。

## 5. 冻结有限域

### 5.1 Exact collision fixtures

- 十字 crossing 与中心 degree-4 junction：最终 PNG bytes 必须完全相同；
- 一根 double bond 与两根独立 parallel single bonds：最终 PNG bytes 必须完全相同。

每一对必须满足：

- PNG bytes 相同；
- extractor Evidence bytes 相同；
- oracle candidate graph set 含两个异图；
- production decision 为 `Ambiguous`，不能选择其中一个。

### 5.2 Strict motifs

只纳入相对于冻结 grammar 图唯一的 observation：

- 空白；
- 单根 single lane；
- 两段非共线 chain；
- degree-3 fan；
- 不含内部 crossing 的三角环和四边环；
- 互不相交且非平行对齐的两个 component。

如果独立 oracle 枚举发现某个预期 strict motif 实际包含异图候选，该样本自动转入 ambiguity，不允许
保留 strict 标签来迁就测试。

### 5.3 域外/干扰

- 非 PNG、CRC 错误、interlace、16-bit、未知 critical chunk；
- 文字式 `N`、`H`、`Cl` 笔画与 lane 相触或遮挡；
- 曲线、虚线、楔形、抗锯齿导致的 topology 不稳定；
- 不能被冻结 grammar 完整解释的墨迹。

这些样本必须 Reject 或在候选确实异图时 Ambiguous；不得输出未经认证的 Graph。

## 6. Hidden challenge

实现冻结以后，独立 red-team agent 才 reveal seed，并生成至少 2,000 个样本。参数至少覆盖：

- 未见坐标，且端点离边界至少 4 px；
- 旋转、水平/垂直反射；
- 整数平移和离散缩放；
- 1--5 px 线宽；
- RGB/RGBA/灰度 PNG；
- 最近邻和抗锯齿两条 raster lane；
- 与结构分离和相触的 text-like interference。

commitment 在实现冻结前只公开 `SHA256(seed || canonical_plan)`，不公开 seed。reveal 后 runner 必须
验证 commitment。实现作者不能修改 extractor、enumerator、阈值或 schema；任何修改都新开版本。

## 7. 稳定性审计

观测距离使用完整二值前景的 Hamming distance；对抗锯齿另报 8-bit luma L1，但不混入主定理。

对每个 renderer profile / motif bucket 计算：

```text
epsilon = 同图允许扰动到 canonical render 的最大距离
Delta   = 该桶中异图 render 的最小距离
```

只有 `2 * epsilon < Delta` 的样本桶可以进入 stable strict Graph 域。等号也不通过。无法完整枚举异图
对的桶只能标为 `uncertified`，不能用采样最小值冒充全局 Delta。

## 8. GO / NO-GO 门禁

全部条件是合取关系：

1. extractor 隔离静态审计通过；
2. extractor 运行期系统调用审计通过；
3. PNG bytes 到 Evidence 的 A/B 输出逐字节一致；
4. exact collision pair 的 PNG 与 Evidence 均逐字节相同；
5. collision candidate soundness/completeness 为 100%；
6. strict candidate soundness/completeness 为 100%；
7. strict decision Graph exact 为 100%；
8. ambiguous decision honesty 为 100%；
9. 域外输入错误 Graph 输出为 0；
10. 被标为 stable 的全部桶满足严格不等式 `2 epsilon < Delta`；
11. 独立 verifier 不导入 extractor、runner、Scene generator 或现有 canonicalizer；
12. 两个独立 reviewer 均批准，红队没有未解决的真值泄漏路径。

任一项失败，总判定为 `NO-GO`。禁止通过缩小分母、删除失败样本、修改 hidden seed、把 Reject 算作
strict correct，或只报告总体平均数来获得 PASS。

## 9. 本周期输出

- `formal/chemgraph/ChemGraph/RasterEvidence.lean`：接口级定理；
- `formal/chemgraph/audit/raster_neutral_extractor_v0_3.py`：隔离 extractor；
- `formal/chemgraph/audit/run_raster_evidence_v0_3.py`：正式 runner；
- `formal/chemgraph/audit/verify_raster_evidence_v0_3.py`：独立 verifier；
- `formal/chemgraph/audit/test_raster_evidence_v0_3.py`：正常和篡改测试；
- `formal/chemgraph/audit/results/raster_evidence_v0_3/{final_a,final_b}`；
- `docs/model/formal/20260815-raster-evidence-v0_3-results.md`；
- 更新 `formal/chemgraph/validation.json`，但保持 `training_authorized=false`。

## 10. 明确不声称的内容

- 不声称已恢复任意分子图；
- 不声称文字与键已经分离；
- 不声称 double/triple bond 可仅由平行墨迹无歧义恢复；
- 不声称 crossing 可全局解释为 disconnected；
- 不声称有限微型 grammar 的 PASS 能迁移到 USPTO 或 MR-TADF；
- 不因 Lean build 通过而声称 Python extractor 正确。
