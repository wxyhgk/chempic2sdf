# 自动实验 Campaign

本目录保存可执行的实验规格，不保存训练产物。训练、评测、状态和报告统一写入
`outputs/experiments/<campaign_id>/`。

## 原则

- 一个 Campaign 对应一个明确目标或正式版本；
- 每个 trial 只验证一个主要假设；
- GO/NO-GO 门槛必须在运行前写入 YAML；
- Campaign 初始化后，配置、源码和输入 checkpoint 哈希必须保持不变；
- 自动系统只能生成 promotion candidate，不得替换稳定模型或打 tag；
- MR-TADF 不得出现在自动门控中。

## 命令

验证配置：

```bash
.venv-train/bin/python -m chemvector.experiment_loop validate \
  experiments/campaigns/v0737-vertex-set/campaign.yaml
```

运行或恢复：

```bash
.venv-train/bin/python -m chemvector.experiment_loop run \
  experiments/campaigns/v0737-vertex-set/campaign.yaml
```

查看状态：

```bash
.venv-train/bin/python -m chemvector.experiment_loop status \
  experiments/campaigns/v0737-vertex-set/campaign.yaml
```

`run` 会自动跳过已通过阶段。上次在某阶段中断且该阶段声明了 `resume_command` 时，
恢复运行会使用恢复命令；否则重跑该阶段。

当前 Campaign：

```text
v0737-vertex-set/        修正指标和错误分类基线
v0737-candidate-oracle/  双分支低阈值 Candidate Oracle
```
