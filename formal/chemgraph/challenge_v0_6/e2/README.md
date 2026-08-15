# E2 生产像素绑定

E2 当前状态是：

```text
functional_pass_resource_audit_pending
training_authorized=false
```

三套独立实现已经对 8 个冻结跨 tuple witness 完成功能验证；第一版权威资源 supervisor 没有通过
红队审计，因此本目录不宣称 authoritative E2 GO，也没有向 `/opt/chemgraph/results` 发布 v0.6
正式结果。

## 功能结果

冻结组合是：

```text
caps      butt, square
width_px  2, 4, 6, 8
总计      2 x 4 = 8 对
```

每一对由 S8 tuple 与 C4 tuple 组成。左右 raw Gray 的绝对位置不同，规范化前字节必须不同；冻结
translation preprocess 后，canonical Gray 和 canonical PNG 必须逐字节相同。选定 interpretation
的规范图必须不等价。

开发运行结果：

| 项目 | 结果 |
| --- | ---: |
| witness pair | 8 / 8 |
| parent tuple membership | 16 / 16 |
| padded clipping certificate | 16 / 16 |
| raw Gray full-byte distinct | 8 / 8 |
| canonical Gray full-byte equal | 8 / 8 |
| canonical PNG full-byte equal | 8 / 8 |
| Graph inequivalent | 8 / 8 |
| sidecar full-byte replay | 64 / 64 |
| sidecar bytes | 35,677,876 |

Producer、verifier A、verifier B 都使用真实固定 4-worker 调度，每个 worker 处理 2 对：

- Producer：Python，复用冻结生产栅格栈；
- Verifier A：独立 Python stdlib 实现，不导入 producer 或 v0.4/v0.5 栅格实现；
- Verifier B：独立 C11/GCC/zlib 实现，不调用两套 Python 实现。

测试结果：

```text
producer     5 项通过
verifier A   8 项通过
verifier B   正向 1 项、负向篡改 12 项通过
verifier B   ASan/UBSan 通过
```

## 冻结与哈希

```text
P0 contract
e326fc744903b998793746f299232c4225ce3b8648e1abdda7b4e065f06a9402

E2 source commit
5970f316a4cfd30a25d3e89776ab8a4c50742b65

E2 source archive
3385833f32802983b5e84885ecf4ccaaa757f48341a993f576e0bbca8d97e21f

E2 source freeze record
994ae0de25ae47835c8b2bb97e3579555c45d8b79fd41a3a7c195eb9a0c798dc

Producer manifest
1a961a0e82b6b699530981cfd3fc3ab4c31455d5f539702f42acab48773803a1

Artifact tree
982f77bce0d5a24ffc6f691543dec6b7611e5f253efdf42592b7e0174138ab08

Producer validation
07c2a68faf3afd5f9d5b4b4e89d38e2cde456a82eff967046ec7c0e9af9145f5

Verifier A validation
55c812081898ad75d12149774a9ca576f797bd97f29a4f74730cbfcc9ef8fdf4

Verifier B validation
fb28ddc867f0f55e01feac07ba920bc45d8e2079c17cbdcdff9d5f9af93d3457
```

机器可读汇总见 `development_receipt.json`。它明确标记 `authoritative=false`。

## 目录

```text
producer/     冻结生产实现、测试和本地开发结果
verifier_a/   Python 独立验证器、测试和开发 validation
verifier_b/   C 独立验证器、构建脚本、负向测试和开发 validation
```

开发结果目录和 validation 没有进入 E2 source commit。Source freeze 只绑定 19 个实现/测试文件，
避免先看结果再修改源码。

## 开发重放

在仓库根目录运行：

```bash
python -m unittest \
  formal.chemgraph.challenge_v0_6.e2.producer.test_producer_v0_6 -v

python -m unittest discover \
  -s formal/chemgraph/challenge_v0_6/e2/verifier_a -p 'test_*.py' -v

bash formal/chemgraph/challenge_v0_6/e2/verifier_b/test_negative.sh
```

Producer 的正式 CLI、A 的 CLI 和 B 的 `run.sh` 都要求冻结 P0/parent contract，拒绝哈希漂移。

## 权威资源审计为何未签发

第一版 supervisor 虽然实现了 cgroup、tmpfs、4 核 affinity 和 network namespace，但红队证明其
隔离边界仍不够严格，主要包括：

1. cgroup 没有覆盖从 archive 到 copy/fsync/publication 的完整流水线；
2. root stage 在 drop capability 前可以用 `setns` 返回宿主网络；
3. CPU 需要 cgroup cpuset 硬门，不能只靠轮询 affinity；
4. writable mount、`TMPDIR` 和 stage descendant 必须闭合，避免逃离专用临时空间或产生 TOCTOU；
5. Git、Python、GCC、zlib 和完整 archive 仍需更强的字节级绑定；
6. 权限变更后必须再次 fsync，wall 必须由 cgroup 外部 launcher 从 exec 前执行硬 deadline。

在这些门全部修复、再次独立审查并从冻结 supervisor 运行之前，E2 只能报告功能证据，不能报告
权威资源 GO。

## 结论边界

E2 功能结果证明：冻结父域中至少存在 8 个跨 tuple 的生产 canonical-raster 异图碰撞。它不证明
碰撞率，不证明现实 OCSR 分布中的发生率，也不证明神经网络泛化。

这进一步确认了 v0.6 的核心设计：系统必须支持 `Graph / Ambiguous / Reject`，不能继续把任意单图
强制压成一个 SDF 并称为普适精确逆映射。
