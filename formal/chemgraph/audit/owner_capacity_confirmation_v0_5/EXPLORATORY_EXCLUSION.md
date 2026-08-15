# 探索运行排除记录

日期：2026-08-15

```text
exclusion_id=owner-capacity-v0.5-exploratory-oracle-001
status=excluded_exploratory
actor=/root/v02_oracle_audit
event_order=before_current_witness_generation
reported_time=2026-08-15 23:00:33 +08:00
exact_execution_time=unknown
head=4972e0853150e6b9de72be4cd4490c554468c39c
cwd=/home/zhenghe/GK/Project/chempic2sdf
python=.venv-train/bin/python (Python 3.13.5)
constructor_sha256=5631ba0d2ca24ab324f77880ecc0ce9668bb7bc85de78e828325b2c04fb3b4cf
exploratory_source_persisted=false
exploratory_source_sha256=null
exploratory_sha_comparison_performed=false
structurally_byte_distinct=true
```

在本目录的 `axis_plan.json` 提交前，Oracle subagent 未遵守“禁止运行 constructor”的任务约束，使用其自行设计、
未冻结、未写入本目录的内存 fixture 调用了 R0 constructor，并报告了四轴的预期成功/拒绝结果。

该 agent 报告共执行两条 shell invocation、十次 fixture constructor call：第一次为四轴 L/H 八次，第二次重新执行
intersection L/H 两次。探索 observation 使用伪字段 `png_sha256=00...00`、`renderer_sha256=11...11`，没有
PNG/support/renderer binding；source 从未序列化或计算 SHA，禁止事后重建摘要伪造 provenance。

当时已知依赖闭包：

```text
common.py             26b7ff36222e4b5239322052f03025fe4db7d98492f10425b82f489606ca68d8
build_observation.py  5736e410270e06b5728f6285187156fd1549abd77ce495b57cccec97b1d9a748
fragment_root.py      69aa32043b87cc241d3fac83279e8e708f06c36c7c0771c8f623d97daad85106
support_partition.py  c36eef270ebee85a842e445e0048b214dec7382544766b7b29a30062d0240e36
```

报告的探索结果如下；这些数字只用于审计污染范围：

```text
carriers L/H:      carrier 8/9, intersection 2, fragment 10/11,
                   child RLE 510/511, variables 2, raw 1
                   L candidate_domain; H CAPACITY_EXCEEDED:max_carriers
intersections L/H: intersection 8/9, fragment 22/23 或第二次 H=24,
                   child RLE 1114/1165 或 1665, variables 8/9
                   L R0_REQUIRES_TWO_INTERSECTIONS;
                   H CAPACITY_EXCEEDED:max_intersections
RLE L/H:           child RLE 65536/65537, variables 2, raw 1
                   L candidate_domain; H CAPACITY_EXCEEDED:max_total_rle_runs
variables L/H:     variables 4096/4097, edges 15999/16002, raw 1
                   L candidate_domain (reported 2,081,633 bytes);
                   H CAPACITY_EXCEEDED:max_pixel_variables
```

该运行没有 write-deny sandbox、side-effect/process/syscall census；Python import 可能写入 `__pycache__`，不能声明零
副作用。

该运行被明确排除在正式确认之外，原因如下：

- 没有预提交 source bytes、source SHA、builder SHA 或完整 dependency bundle；
- 没有 write-deny sandbox、side-effect census 或独立 generic semantic verifier；
- Oracle 报告的计数与本目录 witness 明显不同，例如其 carrier L child RLE 为 510，而当前 witness 为 55；其 pixel L
  edge 为 15999，而当前 witness 为 15806；
- 当前八份 witness 是在该 agent 运行之后由不同 builder 生成，并具有下列 allowlist SHA：

```text
max_carriers L         79cedc830035f27e12639398259cd0a414199f52d5fcbf151c771617bbb9d776
max_carriers H         398fe48caa4c7ff64a3efa6cdbfb87f8467d72b5a94e207a3fb19315a15bc861
max_intersections L    fbff170d0021f281b1e7f94b6a83d38abd441f224fd953f71c095c5369683c9b
max_intersections H    80a24131a4d3715739762d4af6d55ba7aa99a5f8b11577a61438c30cde963a32
max_pixel_variables L  118dea6a4e9f010b85181ef1c0480bdda3444c841414ad9fd91a4244a9e8d2f9
max_pixel_variables H  a7ec8c9c381b191911655997593f5a323a64182e73e36e1b91652b5b39382a7a
max_total_rle_runs L   010a194d13ffdb4dca1aa36186b2eb523a76c1e60e885b503d573f0dcfe6fd2a
max_total_rle_runs H   c1392d46915f66cdb8cc9248a30de9669cf38e292201bd04edadd169667fe00f
```

- 字节不同由结构计数证明，但不存在可比较的探索 SHA；
- Oracle 输出不得进入逐轴 verdict、GO 分母、replay count、通过率、阈值/fixture 选择、failure priority 或 benchmark
  叙述；
- 信息流已经到达 parent，因此当前计划不宣称 blind，只能依赖 committed-byte isolation 和新执行器；
- 正式执行器只能接受上述 committed allowlist，任何其他 source 一律标为 exploratory；
- 最终 verdict artifact 必须绑定本 exclusion record SHA，并证明未引用上述探索输出。

这次污染说明多 agent 互审本身也必须受预注册约束。正式确认只能在本目录 plan commit 之后，由新的隔离执行器读取
已提交 witness bytes 运行；任何未绑定运行继续只记为 exploratory。
