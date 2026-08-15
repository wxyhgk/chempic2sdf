# E2 独立验证器 B

本目录是 E2 的第二套独立实现。验证器使用 C11、GCC 和 zlib，不导入、不调用 producer 或 verifier A，也不把 manifest 中的结论当作事实。

它独立完成以下工作：

- 用冻结混合进制公式复算 16 个 tuple ID，并核对 8 组 `cap x width` 笛卡尔积。
- 从固定 motif、Q16 变换和 `binary_1x` 采样规则重新生成 butt/square 线帽栅格。
- 在 1024 x 1024 画布重画并核对嵌入区、外围白色区域和前景包围盒。
- 对 768 x 768 Gray8 做紧包围盒裁切和平移规范化。
- 使用 zlib level 9 独立编码 canonical Gray8 PNG。
- 对 raw、padded、canonical Gray 和 PNG sidecar 做完整字节比较；SHA256 只作为附加承诺检查。
- 独立构造 `s8` 与 `c4` 的选定图语义，核对规范图字节并证明二者不等价。
- 严格启动 4 个子进程，每个进程固定验证 2 对；任一 worker 异常都会使整体验证失败。

构建和运行：

```bash
./build.sh
./run.sh ../producer/result/manifest.json validation.json ../producer/result
```

篡改测试覆盖 tuple/轴值、pair 缺失/新增/重复、cap/width、sidecar 哈希、图签名、截断 JSON、伪造相同哈希但实际字节不同，以及 canonical PNG 尾随字节：

```bash
./test_negative.sh
```

任何依赖缺失、输入格式错误、资源门失败或单项不一致都会非零退出并写出 `NO_GO` 报告。报告始终保持 `training_authorized=false`，也不宣称完整域碰撞率或训练泛化结论。
