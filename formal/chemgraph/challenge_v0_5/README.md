# Observation Closure v0.5

本目录实现训练前的 P1 数学与可执行审计，不包含 Candidate 或训练代码。P0 权威合同是：

```text
prereg/p0_contract_v0_5.json
SHA256 55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65
```

主要边界：

- `enumerate_domain_v0_5.py`：主 mixed-radix enumerator；
- `enumerate_domain_reference_v0_5.py`：不导入主实现的独立 reference；
- `canonical_raster_v0_5.py`：strict PNG decoder 与 tight-crop/recenter `P`；
- `domain_scene_v0_5.py`：把冻结 tuple 绑定到 v0.4 motif/rasterizer；
- `run_raster_pilot_v0_5.py`：360,885 tuple worker，输出待资源审计 artifact；
- `supervise_raster_pilot_v0_5.py`：专用 cgroup/tmpfs 资源硬门和最终裁决；
- `schemas/`：P1 closed schemas。

冻结测试命令使用 package-aware `unittest`，不使用 pytest 的目录自动 root 推断：

```bash
cd /home/zhenghe/GK/Project/chempic2sdf
python -m unittest discover -s formal/chemgraph/challenge_v0_5 -p 'test_*.py' -v
cd formal/chemgraph
lake build
```

正式 census/pilot 必须在 P1 source commit 和 `p1_source_freeze_v0_5.json` 生成后重新运行。任何 freeze 前
运行只算 diagnostic，不能写入 P1 结果。Raster worker 只能输出
`completed_pending_resource_gate`；只有 supervisor 能写最终 `passed` 或 `no_go`。

当前 `training_authorized=false`。

正式 supervisor 使用专用系统身份 `chemgraph-pilot`（`61184:61184`、home `/nonexistent`、
shell `/usr/sbin/nologin`）运行 worker，并要求启动前该 UID 没有任何进程。
worker 与 supervisor 分别受 78 GiB 和 1 GiB sibling cgroup 硬限制；二者上限之和为冻结的
79 GiB。迁移前 `VmSize` 必须不超过剩余 1 GiB，并与两个 cgroup peak 相加，作为整个流程
聚合 RSS 的保守上界；发布后还会重新检查两组 OOM 事件。
