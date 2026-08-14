# v1 P2-D2-E3A Style Endpoint 安全清单

> 日期：2026-08-14
>
> 状态：已预注册，等待执行
>
> 审计：`chemvector-v1p2d2e3a-style-endpoint-safe-manifest`

## 目标

P2-D2-E2 已证明 endpoint 模型需要重训。本轮在训练前对 `style_train_100k` 的 100,000 张
标签执行完整 ChemScene v0.2 编译预检，避免数据侧 chemical roundtrip 通过、但视觉坐标协议
仍不可表示的样本进入训练。

## 数据与规则

- 源 commitment：
  `2082372d253fa309715becfd5cba9e5cdb78eb8d84a5ece4ff1eb73bd67390eb`；
- 25,000 个分子、每分子四布局、100,000 张图；
- 逐条读取 ChemMol、Render、Layers JSON 并调用 `build_chemscene_v0_2`；
- 任一布局失败时整分子四布局排除，禁止只跳过失败图片；
- 不生成新的 PNG 或 SDF；
- fresh_dev、sealed_test、MR-TADF、USPTO 均不读取。

安全样本按 `SHA256(20260823:scaffold)` 排序：前 750 个 scaffold 为 calibration，后 750 个
scaffold 为 holdout，其余为 fit。molecule 与 scaffold 跨 split 重叠必须都为 0。

## GO/NO-GO

- 100,000 张源标签全部完成预检；
- 所有失败均有 sample key、异常类型和信息；
- 整分子排除后每个保留分子仍恰有四布局；
- fit/calibration/holdout 的 molecule 与 scaffold 重叠均为 0；
- 生成 PNG 数为 0。

通过只代表训练数据协议可用，不代表 endpoint 模型能力通过。

## 执行命令

```bash
.venv-train/bin/python -m \
  chemvector.evaluation.audits.bonds.build_v1p2d2e3a_style_endpoint_manifest
```

## 产物

- manifest：`outputs/audits/chemvector-v1p2d2e3a-style-endpoint-safe-manifest/manifest.jsonl`；
- audit：同目录 `audit.json`；
- 异常清单：同目录 `rejections.json`。
