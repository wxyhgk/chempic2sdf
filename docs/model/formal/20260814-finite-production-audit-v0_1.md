# 单一样式 production tensor 全量审计

日期：2026-08-14

## 结论

固定提交 `dae58c7` 完整枚举 3×3 点阵上的 66,712 个 0–4 边几何核心，并额外加入
crossing/junction、double/parallel-singles 四个 Lean fixture。所有图片都经过当前部署默认的
`prepare_input_image(768, 548)` 和 `image_tensor`，按最终 CHW float32 tensor 建等价类。

结果：

| 指标 | 数值 |
|---|---:|
| 核心场景 | 66,712 |
| fixture | 4 |
| 唯一 observation | 53,206 |
| 同输入异图 observation 类 | 3,518 |
| 涉及场景 | 17,028 / 66,716（25.5231%） |
| 仅由 production 预处理新增的碰撞类 | 0 |
| 单个 observation 最多不同图签名 | 9 |

两轮全量运行的确定性产物逐字节一致：

```text
summary.json
56c2d7af660711fb5e54cf2767d8a7e205401b53c14978080e2c4c6686d65508

validation.json
7c7c15014bb5ba0cf50d76f8e162f19dbf66abeeb4239be44177e81e8ef6f4b4

collision_certificates.json
fe0df24ed8266d380beba97caf0893709f5076d77b1605b6f8f73e5787920077

independent_review.json
d4e027fd619cbd3d2dbc780c83ac854bf2a817db9afb70f78cdaeeaffe2ec9a4
```

同一冻结提交的 A/B 两轮分别耗时 458.50 秒和 462.21 秒。耗时不参与承诺哈希。

## 额外审查

- Lean 4 `v4.33.0` clean build：11/11。
- Python 单元测试：6/6。
- 两组 Lean fixture 均为 raw raster 相同、production tensor 相同、图签名不同。
- 提交独立 verifier 后，对全部碰撞证书的 12,294 对不同签名 witness 运行 NetworkX 精确
  图同构检查，错误差异为 0；reviewer 脚本 SHA 为
  `0f9ef32937fbd70d9e7c52c42695a6370f7e0917eab53f08756bf56d97fe9cd6`。
- Atom ID、Bond 顺序和无向端点交换不改变规范图签名。
- production digest 函数公开输入只有图片和 `image_is_prepared` 标志。

## 如何解释

这些碰撞主要来自共线长边与分段边、重叠线段、内部 crossing 与真实 junction，以及在已有
墨迹上增加语义 Bond 后像素并集不变。所有 3,518 类在 raw raster 上就已经相同；768 resize
和归一化没有额外制造 exact collision。

25.5231% 不是现实化学图片的预期错误率，也不是当前模型准确率上限。这个有限语法均匀枚举
任意 0–4 边组合，故意高频覆盖非典型的共线、重叠和穿越。它证明的是：只要训练生成器允许
这些场景，单图片单图标签就会产生确定的互相矛盾监督，增加模型参数或样本重复次数无法解决。

红队额外几何诊断表明，该比例对 grammar 非常敏感：排除键穿过活跃 Atom、正长度重叠和
不可见 180° 二价点后，允许 crossing 时只有 8 个碰撞类仍保留不同图；再禁止无标记 crossing
时只剩 double/parallel fixture。该数字尚未由提交内的正式 grammar classifier 复现，只作为
下一周期的预注册目标，不属于本轮承诺结果。

## 当前决定

状态是 `passed_full_audit`，但 `training_authorized=false`。训练数据必须按最终 observation
等价类分为 `graph_strict`、`prior_resolved` 和 `intrinsically_ambiguous`；第三类不能继续作为
普通单答案监督。

下一门是补齐 Text、双/三键、visible port、文字 attachment 和多归属像素的 Oracle field，
在此之前先冻结 `simple_line_angle_valid`、`crossing_allowed`、`adversarial_invalid` 三层
grammar，并输出全部场景的 `scene_id -> observation -> grammar layer -> class` manifest。
随后再运行字体、颜色、线宽、lane gap、模糊和缩放 shell，计算异图最小间隔与拒识半径。
最后才做 Oracle field 到 production decoder 的 100% roundtrip，仍不直接开始大规模训练。
