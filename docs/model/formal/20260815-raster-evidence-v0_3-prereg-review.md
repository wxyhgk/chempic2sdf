# Raster Evidence v0.3 工程威胁模型与预注册审查

日期：2026-08-15

状态：`PREREGISTRATION_REVIEW`。本文只定义隔离协议与可执行门，不实现 extractor，不修改主预
注册文件，不授权训练。

## 1. 审查结论

v0.2 已经证明：从有限 Scene `PhysicalMember` 重建 lane，再验证 `field -> graph` 左逆，不等于
实现了 `PNG -> Evidence`。v0.3 的首要问题不是提高几何指标，而是证明 extractor 的输出确实
只由交给它的 PNG bytes 决定。

建议把系统拆成三个互不信任的可执行单元：

```text
Host input builder          Extractor sandbox       Candidate sandbox
canonical PNG bytes  ---->  neutral Evidence  ----> complete CandidateSet
       |                            |                         |
       +-------------------- Host-only verifier <------------+
                                    |
                              GT / SceneClass_D
```

- Extractor 不得看到 Scene、SDF、manifest、样本 ID、输入 hash 或 GT。
- Candidate enumerator 只看到 neutral Evidence，不看到 PNG、Scene 或 GT。
- Verifier 是唯一允许同时读取冻结 GT class、Evidence 和 CandidateSet 的进程。
- 每张 PNG 使用一个全新进程和空临时目录，禁止跨样本状态。
- 本周期只覆盖冻结 micro-scene 的无文字隐式顶点骨架。不能把通过结果外推成完整化学 OCR。

单纯做 Python import 检查不构成隔离。Python 可以通过 `open`、`ctypes`、直接 syscall、继承文件
描述符、`/proc`、Unix socket 或硬编码表绕过。隔离必须以 Linux namespace、只读最小 rootfs、
seccomp、资源限制和 syscall trace 共同实现。

### 1.1 现有代码的复用边界

对当前代码的审阅结论如下：

- `observation_recovery_v0_2.py` 直接 import `Scene`、fixture、core generator 和 graph
  canonicalizer，并读取冻结 manifest/certificate。它只能留在 GT audit side，任何函数都不能被
  v0.3 extractor 或 candidate enumerator import。
- `run_observation_recovery_v0_2.py` 在同一进程同时持有 frozen truth、observation class、Evidence
  witness 和 decision。该结构适合有限表审计，不是 v0.3 launcher 模板。
- `verify_observation_recovery_v0_2.py` 的 stdlib-only、从逐行产物重算、不信 summary 的方向可以
  保留；但它只验证已报告 protocol choice 与冻结 candidate class 的关系，并没有独立枚举
  raster candidates，不能直接改名充当 v0.3 candidate enumerator。
- `finite_micro_raster_v0_1.py` import Scene generator，并负责 render/GT 对齐，只能在 Host input
  builder 一侧运行。
- `chemvector/core/image_io.py` 同时 import PIL、NumPy、SciPy 和 Torch。`prepare_input_image` 可由
  Host 以冻结 SHA 使用；v0.3 extractor 不得 import 该模块，否则会把庞大的插件、native library
  和文件访问面带入沙箱。

因此 v0.3 不能在现有 v0.2 runner 中新增一个 `extract_png()` 后宣称完成隔离。launcher、extractor
package、candidate enumerator 和 verifier 必须是四个独立依赖闭包。

## 2. 威胁主体与保护目标

### 2.1 保护目标

必须证明以下可观察因子分解，而不是只在结果表上找到一个映射：

```text
extract : CanonicalPngBytes -> NeutralEvidence

相同 CanonicalPngBytes
    -> 相同退出状态
    -> 相同 stdout bytes
    -> 相同 NeutralEvidence
```

同时证明：

```text
enumerate : NeutralEvidence -> CandidateSet Graph
```

在声明有限域内 sound 且 complete。最终保守分类只能是：

```text
|CandidateSet| = 0  -> Reject
|CandidateSet| = 1  -> Graph
|CandidateSet| > 1  -> Ambiguous
```

### 2.2 假设的攻击或误用

审计必须能发现以下行为：

1. extractor import `finite_micro_scene_v0_1`、v0.1/v0.2 runner、graph canonicalizer 或 RDKit；
2. 读取 `scene_manifest.jsonl`、collision certificate、SDF、Render JSON、训练 cache 或 benchmark；
3. 根据 PNG hash、PNG 完整字节、文件名、样本序号或执行顺序查硬编码答案；
4. 从环境变量、当前目录、`argv`、父进程命令行、`/proc/*/environ` 或继承 FD 获取样本身份；
5. 通过网络、Unix socket、共享内存、GPU driver、持久 `/tmp` 或缓存获取旁路数据；
6. 在输出 ID、排序、浮点舍入、时间、随机数、PID 或 hash seed 中制造非确定性；
7. 用选择性崩溃、超时或超大输出跳过困难样本；
8. 把 GT ownership、Atom/Bond、bundle、crossing/junction 意义伪装成“可见字段”；
9. candidate enumerator 与 extractor 共用实现或 canonicalizer，使同一 bug 被相关实现确认；
10. verifier 只相信 summary 数字，不从逐样本原始产物重算。

不声称能从公开 fixture 排除任意恶意硬编码程序。该风险由“实现先冻结、独立 agent 后生成未知
输入、bundle 体积与常量审计、无 GT 可访问”共同降低，而不是由禁止 `hashlib` 单独解决。

## 3. 进程与文件系统隔离

### 3.1 进程生命周期

- 一张图片启动一个新 extractor 进程；不得批处理。
- 命令行固定且不含路径、sample key、split、seed 或序号。
- stdin 只有 canonical PNG bytes；stdout 只有一个 canonical Evidence JSON；成功时 stderr 必须空。
- Host 用 `close_fds=true`，`pass_fds=()`；sandbox 内只能存在 FD `0/1/2`。
- 每次使用新的 PID、mount、network、IPC、UTS 和 user namespace。
- 启动后设置 `no_new_privs`，Linux capability bounding/permitted/effective/ambient 集合全部为空。
- 禁止 fork、clone 新线程/进程、exec、ptrace、`process_vm_readv`、mount、BPF 和 perf event。
- v0.3 仅允许 CPU；不映射 `/dev/nvidia*`、`/dev/dri`、CUDA cache 或模型权重。

一图一进程牺牲速度，但先消除跨样本 cache、顺序信道和状态污染。性能不属于本数学门。

### 3.2 最小只读 rootfs

建议使用固定 digest 的 Bubblewrap/nsjail rootfs，并同时用 seccomp 强制。rootfs 只包含：

```text
/opt/python/bin/python
/opt/python/lib/...              # 经 SHA256 清单冻结的最小解释器和 stdlib
/opt/extractor/main.py           # 冻结入口
/opt/extractor/package/...       # 冻结 extractor 源码
/dev/null
/work                            # 空、只读工作目录
/tmp                             # 每次新建的受限 tmpfs；成功后必须为空
```

禁止挂载：

```text
/home
项目根目录
Data/
formal/chemgraph/audit/results/
/proc
/sys
/run
/var/cache
宿主 /tmp
SSH、Git、Hugging Face、字体、CUDA 和用户 cache
```

动态链接器和共享库也必须进入 rootfs SHA256 allowlist，不能笼统只读绑定宿主 `/usr`、`/lib` 或
`/etc`。所有 symlink 解析后的 inode 必须位于冻结 rootfs；禁止通过 symlink/hardlink 逃逸。

若所选运行时离不开 `/proc`，本协议视为隔离失败，不得用 `hidepid` 代替。必须先换成不依赖
procfs 的运行时镜像。

### 3.3 环境与 Python 启动

Host 先 `clearenv`，再只设置：

```text
LC_ALL=C
LANG=C
TZ=UTC
PYTHONDONTWRITEBYTECODE=1
PYTHONNOUSERSITE=1
PYTHONSAFEPATH=1
```

以绝对路径启动 Python，并使用隔离启动参数；不依赖 `PATH`、`HOME`、`PYTHONPATH`、用户 site、
`.pth`、`sitecustomize` 或 `usercustomize`。生产代码必须对 dict/set 全部显式排序，不能依赖
`PYTHONHASHSEED` 获得确定性。

extractor 可读取的环境 key 必须与上述集合精确相等。任何额外 key，包括代理、GPU、CI、Git、
数据目录和 secret，都是硬失败。

### 3.4 import allowlist

v0.3 第一版建议 extractor 源码只允许直接 import：

```text
__future__, array, collections, dataclasses, fractions, functools,
heapq, itertools, json, math, operator, struct, sys, typing, zlib
```

不允许第三方包。extractor 源码尤其禁止直接或动态引用：

```text
os, pathlib, glob, importlib, inspect, pkgutil, marshal, pickle,
sqlite3, shelve, hashlib, subprocess, multiprocessing, socket,
ctypes, cffi, urllib, http, requests, torch, numpy, scipy, PIL, cv2,
rdkit，以及 formal、chemvector、Data 下的所有模块
```

`dataclasses`、`typing` 等标准库会传递 import 其他模块，甚至可能带入 `os`。因此不能把上面的
源码 allowlist 冒充完整 runtime closure。必须在干净 rootfs 中实际启动一次，记录解释器 bootstrap
和允许直接 import 所产生的完整传递闭包，再逐文件冻结 SHA256。extractor 源码不得通过
`sys.modules` 使用仅因传递依赖而加载的 capability 模块。

`hashlib` 禁止是为了减少合法代码表面积，不是防硬编码的安全证明；攻击者可以自己实现 hash。
extractor bundle 只允许 UTF-8 `.py`，不得携带 `.pyc`、`.so`、zip、pickle、图片或任意数据资产；
本版建议冻结为：最多 32 个源码文件、总大小不超过 1 MiB、单个 bytes/string literal 不超过
4 KiB、全部 bytes/string literal 合计不超过 64 KiB，并由 AST 检查。Python/zlib 运行时自身
需要的冻结 native library 可以存在于 rootfs，但 extractor 不得自带 native extension。

检查必须同时包含：

1. 对 extractor 全部 `.py` 做 AST import/call 审计；
2. 对 rootfs 中 `.py`、`.pyc`、`.pth`、`.so`、zip、pickle 和数据文件做 allowlist；
3. 运行时 Python audit hook 记录 `import/open/socket/subprocess/ctypes.dlopen`；
4. OS syscall trace 独立检查实际访问。

只做第 1 项不能通过。Python audit hook 是取证，不是 sandbox；真正的阻断由 mount namespace 和
seccomp 完成。

### 3.5 seccomp 与资源门

seccomp 采用“允许最小 syscall 集”，而不是只列几个 deny。launcher 必须先进入 namespace 和冻结
rootfs，再只允许一次固定 inode 的最终 Python `execve`；进入解释器后加载第二阶段 policy，拒绝
所有后续 exec。若使用的 sandbox 无法证明这个两阶段边界，应改用在最终进程启动前安装过滤器的
机制，不能简单把 `execve` 加入 deny 后假称解释器已成功启动。第二阶段至少显式拒绝：

```text
socket, socketpair, connect, bind, listen, accept, sendto, recvfrom,
clone, clone3, fork, vfork, execve, execveat,
ptrace, process_vm_readv, process_vm_writev,
mount, umount2, pivot_root, setns, unshare,
bpf, perf_event_open, keyctl, add_key, request_key
```

文件打开即使被 seccomp 允许，也只能落在只读 rootfs allowlist。v0.3 初始资源值建议冻结为：

- wall time 30 s，CPU time 20 s；
- RSS/地址空间 1 GiB，stack 64 MiB；
- `NPROC=1`、`NOFILE=8`、`CORE=0`；
- stdout 8 MiB、stderr 64 KiB；
- tmpfs 16 MiB；
- JSON 深度 32，任一数组最多 1,000,000 项。

数值应在实现前进入主预注册；若 smoke 证明不够，只能提交修正案后重跑，不能在正式样本上临时
放宽。

超时、OOM、signal、非零退出、stderr 非空和输出超限都计为样本失败；100% 门不允许忽略。

## 4. Canonical PNG 输入协议

### 4.1 stdin

stdin 不使用 JSON envelope，不传 request ID 或 hash，内容就是一个 PNG 文件从 signature 到 IEND
的完整 bytes，EOF 后不得有第二条消息。Host 在沙箱外保存 `(sample key -> input/output)` 映射，
该 key 永远不进入子进程。

v0.3 只接受冻结 production observation 的 canonical PNG：

```text
width                 768
height                768
bit depth             8
color type            2 / RGB
compression/filter    PNG 标准方法 0
interlace             0
allowed chunks        IHDR, one-or-more IDAT, IEND
ancillary chunks      全部禁止
trailing bytes        禁止
maximum byte length   4 MiB
decoded byte length   768 * 768 * 3
```

必须验证 chunk 顺序、长度、CRC、zlib 流完整结束、无未消费数据，并使用有限输出解压防止 zip bomb。
冻结 canonical encoder 的源码、Python/zlib/Pillow 版本和容器 digest；正式 A/B 复用已承诺 PNG
bytes，不在两轮之间重新编码。

PNG bytes 是 v0.3 的唯一 observation。不得同时传 tensor、原图、文件名、尺寸 sidecar、EXIF、
Render JSON 或预处理参数。若未来更换尺寸或 tensor 协议，必须升 schema version，不能静默兼容。

### 4.2 预处理边界

`prepare_input_image` 当前包含边框移除、缩放和居中。Host input builder 必须在独立、无 GT 的
步骤中运行冻结版本，并记录：

- 输入原始 bytes commitment；
- `prepare_input_image` 与 canonical PNG encoder SHA256；
- 输出 PNG bytes SHA256，由 Host 记录但不传 extractor；
- duplicate PNG class。

任何依赖 SDF 坐标、Scene family、split 或 GT bbox 的裁剪都会使整轮失败。

## 5. Neutral Evidence 输出 schema

stdout 必须恰好是一行 UTF-8/ASCII canonical JSON，加一个 `\n`。固定 `sort_keys=true`、无空格、
禁止重复 key、NaN、Infinity、`-0` 和浮点数，最大 8 MiB。所有坐标使用最简有理数：

```json
[numerator, denominator]
```

其中 denominator 为正，`gcd(abs(numerator), denominator)=1`。数组与 ID 使用协议定义的规范排序；
ID 是排序后的连续编号，不得包含 hash、sample ID 或输入字节片段。

建议 schema：

```json
{
  "schema_version": "neutral-raster-evidence-0.3",
  "image_space": {
    "width": 768,
    "height": 768,
    "pixel_format": "RGB8",
    "origin": "top_left",
    "x_axis": "right",
    "y_axis": "down"
  },
  "ink_support": {
    "encoding": "row_rle_v1",
    "runs": [[0, 10, 14]]
  },
  "stroke_fragments": [
    {
      "id": "f000000",
      "support_runs": [[0, 10, 14]],
      "centerline": [[[10, 1], [0, 1]], [[14, 1], [0, 1]]],
      "terminal_event_ids": ["t000000", "t000001"]
    }
  ],
  "terminal_events": [
    {
      "id": "t000000",
      "fragment_id": "f000000",
      "point": [[10, 1], [0, 1]],
      "primitive_kind": "geometric_endpoint",
      "tangent": [1, 0]
    }
  ],
  "intersection_events": [
    {
      "id": "i000000",
      "fragment_ids": ["f000000", "f000001"],
      "point": [[12, 1], [8, 1]],
      "geometry": "proper_cross",
      "marker_observation": "none"
    }
  ],
  "glyph_region_hypotheses": [
    {
      "id": "r000000",
      "bbox_xyxy": [10, 10, 20, 24],
      "support_runs": [[10, 11, 18]]
    }
  ],
  "residual_ink_runs": []
}
```

示例只说明类型，不冻结具体算法。正式 schema 必须补齐 JSON Schema 和 canonicalization test
vectors 后再实现。

字段语义：

- `ink_support`：所有非纯白 RGB pixel 的精确 support；不能用 GT mask。
- `stroke_fragments`：未赋化学意义的像素/centerline 片段；双键两条线仍是两个 fragment。
- `terminal_events`：局部几何端点/遮挡边界假设，不是 Atom 或 Bond endpoint。
- `intersection_events`：几何相交、接触或重叠，不判断 Junction/Crossing 化学意义。
- `glyph_region_hypotheses`：疑似非 stroke 区域，不含 OCR 字符或元素类别。
- `residual_ink_runs`：未被其他 primitive 解释的 ink，防止通过静默丢像素获得漂亮指标。

support partition/overlap 规则必须冻结，并保证 `stroke + glyph + residual` 对 `ink_support` 的覆盖可
由 verifier 精确重算。若某类允许重叠，必须显式输出 overlap bitmap，不能重复计数后声称完整。

## 6. 禁止字段

Evidence 递归 unknown-key 检查必须拒绝以下内容及其别名/编码形式：

- `scene_id`、source、dataset、split、sample key、文件路径、输入/输出 SHA 或 hash bucket；
- Atom ID、元素、形式电荷、价态、手性、隐式氢、OCR 字符串；
- Bond ID、Bond order、display type、lane bundle、chemical endpoint ownership；
- Junction/Crossing 的语义选择、graph、adjacency、canonical graph signature；
- SDF、MolBlock、SMILES、InChI/InChIKey、Render JSON；
- grammar layer、collision class、ambiguity kind、prior ID、survivor、GT count；
- 任意 checkpoint、模型输出或 benchmark truth。

仅禁止 key 名不够。独立审计还要检查字符串、整数数组和 ID 是否携带输入 digest、Scene ordinal 或
candidate graph 的隐写编码。Evidence 不得输出 input hash；完整性 hash 由 Host sidecar 计算。

## 7. syscall、strace 与 `/proc` 审计

### 7.1 强制探针

正式 extractor 运行前，用同一 rootfs/seccomp 运行 adversarial probe，以下访问必须失败：

```text
/proc/self/environ
/proc/self/cmdline
/proc/self/fd
/proc/1/environ
/proc/self/mountinfo
/home
项目根目录
Data/
manifest/certificate/SDF 常见文件名
外网 TCP/UDP
Unix socket
fork/exec/ptrace
```

probe 还应枚举环境、argv、cwd、打开 FD 和可见文件树；其 machine report 必须与 allowlist 精确
一致。probe 与正式 extractor 使用相同 launcher SHA256，禁止测试一个宽松/不同的容器。

### 7.2 strace

对 smoke 全量以及正式 A 轮的每个 extractor/candidate 进程使用 Host `strace -ff -yy`，至少记录：

```text
%file, %process, %network, ipc, ptrace,
process_vm_readv, process_vm_writev, bpf, perf_event_open
```

verifier 从原始 trace 重算：

- 成功打开路径全部在 rootfs inode/SHA allowlist；
- `/proc`、`/home`、项目、Data 和宿主 cache 成功访问为 0；
- socket/process/ptrace 等调用尝试为 0，而不只是成功数为 0；
- 子进程/线程数量恒为 1；
- stdin/stdout/stderr 外无继承 FD；
- traced 与 untraced 输出逐字节相同。

strace 是审计证据，不是执行隔离。恶意程序可以检测 tracer；安全结论依赖 seccomp/mount policy，
并要求 traced/untraced 双路径一致。正式 B 轮不 trace，用于全量 A/B 对照；再从 B 轮预注册 1%
样本追加 traced replay，与 A/B 对应输出比较。trace verifier 只统计进入最终 interpreter 后的 payload
PID，不能把 sandbox launcher 建 namespace 的 syscall 混入 payload allowlist。

## 8. Candidate enumerator 隔离

candidate enumerator 必须由独立实现者在 Evidence schema 冻结后编写，不导入 extractor、v0.1/v0.2
Scene generator、graph canonicalizer或 verifier。它在第二个同等级 sandbox 中运行：

```text
stdin  = canonical NeutralEvidence JSON bytes
stdout = canonical CandidateSet JSON bytes
```

CandidateSet 可包含匿名图结构，因为它不再属于 neutral Evidence，但必须：

- 候选无序集合采用规范排序，重复候选为硬失败；
- anonymous vertex ID 由候选内部规范排序生成，不沿用 GT Atom ID；
- 每个候选注明它使用了哪些 fragment/terminal/intersection interpretation；
- 不得输出 chosen candidate、prior score 或 GT agreement；
- 不得读取 PNG、manifest、SDF、GT、输入 hash 或 sample key；
- schema 与 supported grammar profile 固定，不能对测试结果动态增删规则。

Host-only verifier 使用第三份独立 graph canonicalizer，将 CandidateSet 与冻结
`GraphSet_D(observation)` 做集合精确相等比较：

```text
precision = 100% 不是充分条件
recall    = 100% 不是充分条件
每个 observation 的集合 exact equality = 100% 才通过
```

candidate soundness/completeness 只能相对声明的有限域成立，不能写成现实化学空间完整。

## 9. Exact collision 与 strict 门

必须至少冻结以下测试：

### 9.1 crossing / junction

- 两个 witness 的 canonical PNG bytes 必须逐字节相同；只比较像素相同不够。
- 两次全新 sandbox 的 Evidence stdout 必须逐字节相同。
- CandidateSet 必须恰好包含 crossing graph 与 degree-4 junction graph两个签名。
- Conservative decoder 必须输出 `Ambiguous(crossing_junction)`，不得 Graph。

### 9.2 double / parallel singles

- canonical PNG bytes 逐字节相同；
- Evidence stdout 逐字节相同；
- CandidateSet 恰好包含 order=2 bond 与两个独立 order=1 bond graph；
- Conservative decoder 必须输出 `Ambiguous(parallel_multiplicity)`。

### 9.3 strict motif

对全部 strict observation：

- CandidateSet 必须恰好一个 graph；
- 唯一 graph 必须等于冻结 truth；
- Conservative decoder 必须输出该 Graph；
- 不允许以 Reject 维持“无错误 Graph”。

collision 与 strict 测试必须由 Host 随机交错执行，子进程看不到 pair role 或顺序。

## 10. A/B 与未知输入协议

### 10.1 冻结顺序

严格顺序如下：

1. 提交本预注册、PNG/Evidence/Candidate schema 和指标代码；
2. 冻结 extractor bundle、rootfs、launcher、seccomp、import allowlist SHA256；
3. 冻结 candidate enumerator 与 independent verifier SHA256；
4. 独立 agent 在此后生成未知坐标、旋转、反射、整数缩放、线宽和文字干扰集；
5. 先发布输入 bytes commitment，不向实现者公开 GT；
6. 执行 A/B；
7. 冻结 A/B 产物后才解封 GT 给 Host verifier；
8. 任何代码、阈值、schema、rootfs 或 launcher 修改都升版本并重新开始。

### 10.2 两轮差异

A/B 使用完全相同的 committed PNG bytes，但：

- Host 输入目录绝对路径不同；
- 样本顺序使用两个预注册 permutation；
- 每个样本仍是一图一新进程；
- 临时目录、PID 和 Host worker 分配不同；
- A traced/B untraced 与反向子集均执行；
- extractor/candidate stdout 在 Host 按 input commitment key 重排后比较。

以下 commitment 必须逐字节一致：

```text
evidence_manifest.jsonl
candidate_manifest.jsonl
summary.json
validation.json
isolation_audit.json
failure_certificates.json
```

runtime、PID、trace 原始时间戳单独保存，不进入跨轮 byte-equality commitment；其规范化 syscall
统计必须一致。

## 11. 可机器验证的 100% 门

validation 必须逐项包含下列布尔检查，禁止只给一个总 `passed`：

```text
input_schema_valid_count == input_count
canonical_png_decode_success_count == input_count
extractor_exit_zero_count == input_count
extractor_stderr_nonempty_count == 0
extractor_timeout_or_resource_failure_count == 0
evidence_schema_valid_count == input_count
unknown_or_forbidden_evidence_field_count == 0
noncanonical_json_count == 0
ink_partition_exact_count == input_count
same_png_different_evidence_count == 0
same_png_same_evidence_count == duplicate_png_class_count
evidence_ab_byte_mismatch_count == 0
order_or_host_path_dependence_count == 0
successful_forbidden_file_open_count == 0
forbidden_file_syscall_attempt_count == 0
network_or_unix_socket_attempt_count == 0
process_creation_attempt_count == 0
procfs_visible_probe_count == 0
unexpected_import_count == 0
unexpected_rootfs_file_count == 0
inherited_fd_violation_count == 0
traced_untraced_output_mismatch_count == 0
candidate_schema_valid_count == input_count
duplicate_candidate_count == 0
candidate_set_exact_count == observation_count
candidate_set_soundness_failure_count == 0
candidate_set_completeness_failure_count == 0
crossing_junction_exact_pair_pass_count == required_pair_count
double_parallel_exact_pair_pass_count == required_pair_count
strict_singleton_candidate_count == strict_observation_count
strict_wrong_graph_or_reject_count == 0
ambiguous_graph_output_count == 0
conservative_classification_exact_count == observation_count
hidden_suite_generated_after_freeze == true
extractor_bundle_sha256_matches == true
candidate_enumerator_sha256_matches == true
independent_verifier_sha256_matches == true
ab_committed_artifact_mismatch_count == 0
audit_oracle_only == true
deployable_from_opaque_observation == false
training_authorized == false
```

所有分母必须同时报告；空集合不能让比例真空通过。任何一项失败：

```text
observable_extractor_gate = no_go
candidate_completeness_gate = no_go
overall_gate_status = no_go
training_authorized = false
```

即使全部通过，本周期最多授权“在该 renderer/profile 的声明域内训练 neutral evidence
predictor”，不能自动授权单答案 graph decoder、OCR、SDF 或现实分子部署。

## 12. 主要潜在绕过及对应检查

| 绕过 | 防线 | 机器证据 |
|---|---|---|
| PNG hash/完整字节查表 | bundle 先冻结；未知后生成；无 GT 文件 | freeze timeline、bundle scan、hidden commitment |
| import Scene/manifest | hermetic rootfs + import allowlist | rootfs manifest、audit hook、strace |
| `open` 绝对路径 | mount namespace + seccomp/allowlist | forbidden open attempt/success count |
| `/proc` 读父进程或 FD | 不挂 procfs；FD 只留 0/1/2 | adversarial probe、strace |
| 环境变量泄漏 | `clearenv` + exact env allowlist | inside-sandbox env report |
| 继承 FD 泄漏 | `close_fds` + namespace probe | FD inventory |
| 网络/Unix socket | network namespace + seccomp | socket syscall attempt count |
| `ctypes`/扩展直接 syscall | extractor 禁带 native extension + seccomp | runtime `.so` allowlist、strace |
| `.pth/sitecustomize` 注入 | 隔离 Python 启动；rootfs 文件清单 | startup trace、file manifest |
| symlink/hardlink 逃逸 | resolve inode 后 allowlist；只读 rootfs | mount/inode report |
| 持久 `/tmp` 或跨样本 cache | 每图新 tmpfs/进程 | tmpfs emptiness、process count |
| 样本顺序/路径条件分支 | A/B 不同顺序和 Host 路径 | keyed byte comparison |
| tracer 检测 | seccomp 是强制层；traced/untraced 对照 | output mismatch count |
| 选择性崩溃跳过难例 | 100% exit/schema 分母 | failure certificates |
| float/JSON 非确定性 | 有理数、canonical JSON、禁 NaN | schema/canonical re-encode |
| Evidence ID 隐写 GT/hash | 连续规范 ID；字符串/数值审计 | ID recomputation、entropy/prefix scan |
| 将 bundle/Atom 塞入 neutral 字段 | closed schema + recursive unknown-key reject | forbidden field count |
| candidate 只输出 chosen graph | 与完整 GraphSet 逐 observation 相等 | set exact/soundness/completeness |
| extractor/enumerator 共用错误代码 | 独立包、禁止相互 import | source/import dependency graph |
| verifier 相信 summary | 从 JSONL、PNG、冻结 GT 重算 | independent review SHA |
| 测试后调阈值 | freeze-before-hidden，修改即升版 | Git/container/schema commitment |

## 13. 实施前必须补齐的预注册工件

在写 extractor 之前，主负责人还需冻结：

1. canonical PNG encoder 与 test vectors；
2. `neutral-raster-evidence-0.3.schema.json`；
3. `candidate-set-0.3.schema.json`；
4. Bubblewrap/nsjail launcher 与 seccomp policy；
5. rootfs/import/env/file/syscall allowlist；
6. adversarial isolation probe；
7. Host-only A/B orchestrator schema；
8. candidate equality 与 conservative decoder verifier 规格；
9. hidden-suite commitment/reveal 流程；
10. 所有计数的固定分母和非空门。

在这十项先提交并由独立 reviewer 批准前，不应实现 extractor。当前状态保持：

```text
observable_extractor_gate = NOT_RUN
candidate_completeness_gate = NOT_RUN
audit_oracle_only = true
deployable_from_opaque_observation = false
training_authorized = false
```
