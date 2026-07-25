# ChemVector 自动实验循环

## 目标

自动实验循环负责连续执行、恢复、门控、归档和生成下一任务，减少每轮训练后等待人工
输入。它不是无限自改代码系统，也不能自动发布模型。

```text
有限假设队列
    -> 配置、源码、数据和 checkpoint 哈希
    -> 单元测试与 smoke
    -> 100 图容量
    -> 短训练淘汰
    -> 9k 泛化
    -> 最佳候选复现
    -> 锁定测试集
    -> 端到端 SDF
    -> promotion candidate
    -> 人工 promote / reject / defer
```

## 状态与产物

每个 Campaign 的运行目录是：

```text
outputs/experiments/<campaign_id>/
  manifest.json       配置、源码、数据、checkpoint 和环境哈希
  state.json          原子更新的 Campaign 状态
  events.jsonl        状态变化事件
  decision.json       自动门控结论
  next_task.json      后续动作
  report.md           中文报告
  trials/<trial>/<stage>/
    stage.log
    heartbeat.json
    gpu_telemetry.jsonl
    result.json
```

重新执行同一命令时，已通过阶段会跳过。中断阶段优先使用 Campaign 中声明的
`resume_command`。Campaign 创建状态后，修改 YAML、源码或输入 checkpoint 会被拒绝，
防止一个 run id 对应多个不可复现实验。

## 门控规则

- 门槛在训练前写入 Campaign，运行中不得降低；
- 模块实验以对应 exact 为主指标，F1 只作回归护栏；
- 100 图只证明容量，不能直接成为正式 GO；
- 某 trial 失败后自动运行下一个有限候选；
- 第一个完整通过全部阶段的 trial 成为 promotion candidate；
- 自动系统不修改稳定 `v0.7.10`，不打 tag；
- MR-TADF 不参与任何门控。

## GPU 与恢复

GPU 阶段使用 `/tmp/chempic2sdf-gpu0.lock` 独占 5090。发现其他 Campaign 使用 GPU
时等待，不杀进程。运行期间定期保存 GPU 利用率、显存、温度和功耗。超时先发送
`SIGTERM`，30 秒后仍未退出才发送 `SIGKILL`。

当前 MVP 提供阶段级恢复。各 Trainer 后续仍需逐步补齐 Python、NumPy、Torch、CUDA
随机状态、epoch、shuffle 顺序和数据游标，才能实现 batch 级精确续训。

## 数据集治理

现有 validation、internal 500、USPTO clean 1k 和 MR-TADF 已经被多次查看，只能作为
开发回归或诊断集。正式 promotion 前需要新增 locked holdout：

- molecule、scaffold、font family 和全部 render variant 跨 split 隔离；
- 每个 Campaign 只允许最终一个候选读取一次；
- Campaign 结束前只返回聚合门控，不展示失败图片；
- 失败后不能根据 locked 样本继续修改同一假设族。

## v0.7.37 第一轮

第一份 Campaign 位于：

```text
experiments/campaigns/v0737-vertex-set/campaign.yaml
```

它先运行测试，再重新评测 v0.7.36 的 498 张预测端点基线。修正版评测会分别输出：

- 按 GT display type 分桶的边召回；
- 已恢复真边上的 display、chemical 和 direction accuracy；
- 完整 Bond Object recall；
- 未匹配顶点坐标、最近 GT 键、键型和距离；
- 双键/三键 12px 邻域伪顶点标志。

该 Campaign 只建立可信基线，不宣称新模型提升。下一 trial 才加入全局 Count 和集合
选择器。

首次运行结果：

```text
Campaign 状态                         promotion_candidate
Vertex exact                          9.84%
匹配端点条件 Edge recall             98.70%
已恢复真边 Display accuracy          77.54%
Bond Object recall                    53.62%
计数完全一致                         99 / 498
计数 -1                              89 / 498
计数 -2                              89 / 498
多余顶点                              592
靠近双键的多余顶点                    60（10.1%）
```

诊断表明少选点远多于多选点，双键伪点存在但不是第一主因。自动推荐的下一假设是
`global_count_and_set_selection`。

第二个 Campaign 使用多个隐式阈值和 128/256 候选上限，分别报告 Candidate Oracle、
known-count 和 deployed exact。`t=0.05、k=256` 的预注册门槛是 Candidate recall
不低于 98%、Candidate Oracle exact 不低于 80%。通过后进入集合选择器；失败则先做
MolGrapher 式 adaptive Gaussian 顶点监督。

首次 Candidate Oracle 评测完成后发现门控路径中的 `t0.05-k256` 含小数点，旧解析器
把它误当成多级字段，因此原 Campaign 的 `NO_GO` 属于基础设施失败，不是模型失败。
门控现已支持 JSON Pointer，并使用已生成的同一份 summary 重新判定，不重复运行 GPU。
