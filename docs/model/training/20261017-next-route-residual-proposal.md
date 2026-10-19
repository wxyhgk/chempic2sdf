# r5 下一步：残余原子 proposal shadow

## 已确认的边界

固定 identity-aligned r5 的 160/80 诊断切片、production initial 和
baseline graph。production checkpoint、selector、bond head、OCR 以及 sealed
test 均不读写。

- 全部 vertex candidate 的 holdout atom coverage：`0.9857`
- baseline-owned 后的 residual atom reachability：`0.8686`
- residual candidate hard-collision：`0.0436`
- sidecar pair shadow 的 precision：`0.8663`
- 按错误的 all-bond 分母计算的 recall `0.3873` 不再作为 residual sidecar
  的唯一判据；修正后的 residual recall 为 `0.8148`，union recall 为
  `0.9273`，仍未达到预设 residual recall `0.85`。

因此瓶颈不是候选数量、top-k 容量或 bond type，而是 residual candidate
ownership。图遍历审计（radius 24px）得到 pair precision `0.9799`、bond
recall `0.6433`；其剩余问题是 raster line evidence，暂不进入 GPU 训练。

## 唯一下一步实验

先完成一个 residual-only vertex proposal shadow：

1. 冻结 production model、baseline candidates、bond/OCR；只训练新的小
   residual objectness/ownership head。
2. 正例只取距离 baseline-owned 区域大于 6px 的 GT atom；baseline-owned 和
   6--20px collision 区域作为 hard negative/protected zone。
3. 重点保留 visible O/N，以及一端 baseline、一端 residual 的 bond witness；
   不把 implicit-C 泛化样本重复当作主要数据源。
4. 固定 160 train / 80 holdout，最多 800 steps（单卡）；阈值只用 train
   校准一次，禁止 holdout sweep。
5. 仅生成 in-memory shadow artifact。production 接入门槛：residual
   precision >= 0.80、residual recall >= 0.85、union recall >= 0.90、
   complete-edge precision >= 0.80、dangling <= 0.05，且 baseline prefix
   byte-identical、candidate reuse=0、distinct owners 全部满足。

任一条件失败即 `NO-GO`，转为继续增加 residual-specific 样式/元素数据，
而不是再次训练 pair compatibility MLP。当前没有任何 sidecar 获得生产授权。

## Shadow 结果

按上述协议完成了一次 800-step isolated shadow。holdout residual target
175 个，预测 270 个，命中 165 个：

- residual recall：`0.9429`
- residual precision：`0.6111`（门槛 `0.85`，失败）
- hard collision：`0`
- combined bond coverage：`0.9818`
- one-residual bond coverage：`0.3727`（门槛 `0.50`，失败）
- visible O/N recall：`0.9892` / `0.9434`

因此这次实验说明 residual head 能找回 O/N，但产生了大量非目标候选，
且尚不能稳定提供可连接的 residual endpoint。sidecar、candidate artifact
和训练状态均保持隔离；production checkpoint 的 SHA 与 production state
均未变化。结论为 `NO-GO`，不再做阈值调节或直接 edge promotion。
