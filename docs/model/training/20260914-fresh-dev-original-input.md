# 完整分子原图/SDF 开发验证接入

本轮打通了真实完整分子的验证入口：**200 分子、800 张原图/SDF 全部通过 CPU 输入核验**。训练数据仍需同源 SVG 标签；验证只需要原图和单条 SDF，不会为了满足训练格式而重绘验证图片。没有实际模型训练、GPU 使用或新的识别准确率。

## 固定选择与实际访问

遵循 `Data/sdf/chemvector_joint_v1/README.md`：先读取 aggregate commitment/validation，固定计划并写盘，再访问 fresh-dev 逐样本 manifest 和所选图像/SDF。未运行旧 builder，未查询新的私有身份，未读取 sealed 样本或 work 副本。

- 数据 commitment：`13a456afb2c475a73fc4dca82df5fe62d4a31277d6b584637d9cd92736024512`。
- 访问计划：`outputs/audits/chemvector-fresh-dev-input-20260914-access-plan.json`。
- 计划原始字节 SHA256：`1b9f7c1421ad68adc83795e21cdc80bcbe02b3ffb85889c5791be4ddbc01aca9`。
- 选择：seed=20260914，按 `SHA256(seed:molecule_id)` 排序取 200 分子，保留全部四种变体；不依据像素、结构难度或模型输出筛选。
- 已验证原始 fresh-dev `samples.csv` 的 SHA/字节数，以及两个允许 tar 的完整 SHA/字节数；没有展开 tar 到磁盘。
- 集合计数、分子一致性、四变体完整性在 fresh-dev 全部 8,000 行 manifest 上检查；PNG/SDF payload 检查只覆盖所选 800 张。
- 已固定整个数据 commitment 中的 molecule/key/scaffold、renderer/style、font-group 哈希。它们包含 sealed 的全局集合，所以本工具不会读取 sealed 行来重算这些全局集合。

| variant | 验证图 |
| --- | ---: |
| original | 200 |
| rdkit | 200 |
| coordgen | 200 |
| stress | 200 |

这里“原图”指保留 committed tar 中 PNG 的像素，不代表只有名称为 original 的一种 variant。

## 产物与隔离证据

`outputs/audits/chemvector-fresh-dev-input-20260914/`：

- `validation-candidates.json`：固定验证集合及访问计划绑定，可合并到未来受审 mixed release，但本身 `training_authorized=false`。
- `report.json`：800 图核验、两个 tar 哈希、访问边界，不包含准确率。

验证清单 SHA256：`8abf25f20e643171e459db3cbbdf56dfe055ede6bdec81bbe1e932879020ec9f`。

与前轮剩余训练候选清单（SHA256 `642a283943fea9b0289a3a9608d395b5a6e2ee7dc8607dd95fd405a2aff26aa9`，282 分子 / 2,469 图）从 canonical SMILES 重新计算后比较：

| 跨候选/开发验证交集 | 数量 |
| --- | ---: |
| molecule ID | 0 |
| 完整 InChIKey | 0 |
| connectivity prefix | 0 |
| 去立体/同位素的 core/scaffold 分组 | 0 |

这是本轮所选验证集与固定候选的交集，不证明所有旧训练历史无污染，也不能替代此前独立私有核验。前轮已发现 fresh-dev/sealed 新口径存在 1 个连接层、3 个分组交集，仍保留此限制，没有悄悄调整评估 split。sealed 目录权限仍为 `000`。

## 代码与回归边界

- `fresh_dev.py`：访问计划、manifest 核验、只读 tar 索引和原图/SDF loader，独立于模型和训练标注。
- `validation_truth.py`：评分真值解析，保留立体、电荷、同位素与组分，不访问数据路径。
- `release.py`：兼容旧 `curriculum-same-svg`；新 `curriculum-with-original-validation` 显式绑定 `validation_sources`。同一来源不能改作训练；不允许任意删减已固定的验证集合。
- `launcher.py`：预检后复用任务局部 loader；完整验证不会经过训练 collator，optimizer 只消费 train 行。

合成端到端测试实际执行 1 次微型模型 CPU optimizer step，前后各评估 4 张验证图：训练 collator 只收到训练样本，验证预测器只收到像素，8 次验证推理均禁用梯度。没有提升时仍选择 initial 权重。该测试不加载真实 checkpoint，也不是本模型能力结果。

其他测试覆盖计划/manifest/tar/member 篡改、软链接和不安全成员、重复成员、缺图、多条 SDF、身份错误、错误尺寸/模式、训练用途、跨来源分组污染、固定子集被删减，以及原有 same-SVG 合同的兼容性。

本轮完整相关回归 **258 passed，25.52 秒，CPU**；修改模块编译与 `git diff --check` 通过：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-train/bin/python -m pytest -q tests/chemvector/test_curriculum_*.py \
  tests/chemvector/test_ctab_inventory.py tests/chemvector/test_full_molecule_data.py
```

## 下一步

验证接口这个阻塞已解除；不必重做 renderer，也不应再拿旧 train 重绘当“未见整图”。接下来整理真实混合训练发布：统一课程与完整分子的分组、对齐来源使用依据与现有排重凭据、在独立发布副本中绑定训练准入、冻结一套混合配方与停止条件。然后先测初始权重，再做一次不超过 12 GPU 小时的自研续训，对同一固定开发验证集合比较。

原 quarantine 数据、旧 checkpoint、前端默认和 sealed 访问控制均未改动。本轮没有生成假的 admission，也没有以核验成功代替训练有效性结论。
