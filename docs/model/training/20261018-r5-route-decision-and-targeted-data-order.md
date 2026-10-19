# r5 路线决策与 targeted data 顺序

## 当前决策

现有归因显示，474-image holdout 的 3234 条 GT bond 中有 653 条（20.2%）在 raw line/candidate proposal 阶段不可达；node keep 与 edge threshold 对可达边没有额外损失。因此继续训练 node identity、edge scorer、ownership 或 graph router 不会增加 recall。

保留 direct line-node graph 作为中间表示和 CPU upper bound，但不生产化。冻结 ownership MLP、many-to-one、graph reranker、text-seam bridge、raw multiplicity type router；node-keep 和 frozen edge/type sidecar 只用于诊断基线。

## 先数据，后训练

先生成 6k 张 targeted synthetic shadow 数据：4k train、1k dev、1k fresh holdout，分子身份和字体/样式分层隔离。重点覆盖：

- cov13 base48、stroke16_48；
- 30–50px 短键与 30–60° 方向；
- DOUBLE/AROMATIC 平行线与端点 bundle；
- visible TEXT_SEAM→implicit-C junction。

每张图保留端点、segment、bundle、文字 seam 实例标签；OCR 仍冻结，sealed test 不参与。

## 执行顺序与门槛

1. 先做 target-free raw endpoint / double-bundle deterministic ceiling。新增边 precision `>=.99`，union precision `>=.965917`、recall `>.850031`，六 profile recall `>=.75`、sanitize `>=.90`，dangling `<=.05`。失败即停止。
2. 只有 ceiling 通过，才拟合 tiny endpoint proposal sidecar（CPU ≤500 steps）；200-step full holdout 无严格 graph 改善立即早停。
3. CPU fit 全部门通过后，最多一次 ≤800-step GPU shadow；不改 production。
4. bond-type router 后置。当前 CPU type fit accuracy `.923` 但 sanitize `.743`，暂不训练/GPU；必须先达到 type accuracy `.90`、predicted-type sanitize `.95`。

该顺序把有限 GPU 用在唯一已定位的瓶颈：raw endpoint reachability，而不是继续优化下游已经没有可学习空间的头。

## 6k targeted panel 结果

6k quarantine panel 已实际生成（accepted `5975`、rejected `25`，身份与 SDF/文件合同通过），但 fresh holdout 的严格 direct-LSD proposal replay 只有 edge P/R `.9916/.7893`，dangling `.2635`；六 profile recall 为 `.740-.810`。因此数据扩充本身没有达到 proposal gate，不能进入 sidecar 训练或 GPU shadow。后续必须先改 endpoint ownership/text-seam 连接协议，再重新做 deterministic ceiling。
