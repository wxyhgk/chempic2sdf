# Owner Capacity Confirmation v0.5

本目录预注册 fixed-R0 的容量确认，不授权训练。

当前阶段只完成：

```text
source builder -> frozen synthetic-support L/H observations
independent source measurements -> complete 12-axis plan
conditional certificate checker -> conditional/unproven NO-GO axes
```

`build_axis_plan.py`、`source_builder.py` 和 `certificate_checker.py` 均不 import 或调用 R0 candidate constructor。
八份 observation 各自绑定由 parent-support union 确定性生成的真实 PNG bytes 和 `capacity_png.py` SHA；但
`geometry_renderer_binding=false`，所以这些 fixture 只验证资源容量，不证明 geometry、support 与真实化学图片一致。

冻结裁决：

```text
reachable end-to-end:     max_carriers, max_total_rle_runs, max_pixel_variables
reachable rejection-only: max_intersections
conditional/unproven NO-GO: remaining 8 axes
aggregate even if pass:   PARTIAL/NO_GO_GENERIC
```

运行预注册自检：

```bash
.venv-train/bin/python -m unittest \
  formal.chemgraph.audit.owner_capacity_confirmation_v0_5.test_preregistration -v
```

当前 Python checker 只验证冻结依赖 SHA、镜像 Lean 算术并计算 3,353,654-byte 手工 schema envelope；它没有证明
任意 Python source 都满足 Lean 前提，也没有完成 universal byte-bound proof。因此剩余八轴仍为 conditional/unproven
NO-GO。

`EXPLORATORY_EXCLUSION.md` 记录并排除了一次发生在 plan commit 前的未冻结 subagent constructor 运行。正式容量确认
在本目录形成 Git trust anchor 前不得开始。
