# Hidden Generator v0.4 P2 Commitment 记录

日期：2026-08-15

P0：`55a459499a5f72775e2796c7bd0c0ff012eaf244`

P1：`b015d5d61f9dcc8f4a6c8581d3ff1e2392827557`

两名独立 custodian 在各自 agent 上下文中生成 32-byte share。raw share 未写入仓库、文件系统或给主
agent 的消息；本阶段只公开 domain-separated SHA256 commitment：

```text
C1 = ca577bebe7ef3b9bbca271988d3db8fe995ee65cb8ae9feaff96c3b6e5ee322d
C2 = 3d60542ccf325012e898c13fb6cb689469d8a793f76e8fa60a1a0482c1a66af5
```

Commitment 文件 SHA256：

```text
custodian_1.json
576508af9b788057b6112bf1dd9470ad29af22a524b7d72bf2fac9bc54d4c368

custodian_2.json
ec2173ac9c351f22dccf5147674f4df6bdec7fc49848057c01c17b0095c2c665
```

P2 当前只能验证：commitment 文件格式、C1/C2 不同、P0/P1 绑定、raw share 未出现在交付文件，以及
custodian 声明仍持有 share。由于 share 尚未 reveal，主 agent 和 Lean 都不能验证 C1/C2 是否确实由
声明 share 计算；该绑定必须在 prediction commitment 之后 reveal 时复算。这里明确记录
`commitment_binding_verified=false`，防止把 procedural commitment 冒充密码学证明。

P2 通过后只允许开始 P3 的 extractor/EvidenceCore/candidate 实现。禁止生成正式 hidden pack；禁止
请求 share；禁止训练。`training_authorized=false`。
