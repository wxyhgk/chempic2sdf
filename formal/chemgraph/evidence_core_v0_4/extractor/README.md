# EvidenceCore v0.4 Extractor

该目录只实现 `PNG stdin -> bounded EvidenceCore stdout`。它不读取 Scene、SDF、generator、sealed GT、
family、sample ID 或 GraphSet。

当前实现使用 Zhang-Suen centerline + Hough line，再用墨迹支撑谷判断重复 centerline，避免把真正的
双键/三键 lane 合并。junction 端点只有在延伸路径存在墨迹支撑时才 snap，terminal 只沿自身方向扩展到
可见墨迹末端；小型孤立 component 不参与全局 Hough，但仍进入 residual 统计。

完整 raster 只存在于进程内存，输出禁止 foreground RLE、pixel/skeleton list、digest 或 opaque payload。
成功和失败输出分别进入 closed `ok/reject` schema 分支，双方 stdout 都是 canonical JSON。

```bash
python extract_evidence_core_v0_4.py < input.png > evidence.json
```

公开 smoke 的 primitive geometry 已达到 48/48，单像素稳定性和独立红队子门通过；但下游 Candidate
因 C4/S8 协议冲突只有 44/48，所以 P3 整体仍是 NO-GO。它不是模型，也未获 hidden 或训练授权。
