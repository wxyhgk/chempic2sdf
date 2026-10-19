# 2026-10-19 开发记录

时区：Asia/Shanghai。范围：仓库发布、首页整理和开发记录建设。

## 目标与结果

| 目标 | 结果 | 记录 |
| --- | --- | --- |
| 上传项目 | 创建 GitHub 仓库并按用户要求改为公开，main 上传成功 | 089ecf5 |
| 整理首页 | 清理历史流水账，补充环境、使用与文档导航 | 8916b69 |
| 说明路线与强项 | 增加路线对比、四项设计强项及当前限制 | d8715d9 |
| 建立开发记录 | 采用日期目录、小 CSV 与 Markdown 配合，加入模板和状态页 | [决策](../../../decisions/D-20261019-01-daily-records.md) |

前面三项在本日记录系统建立之前完成，依据当前会话已执行结果和 Git 提交补记。
当天后续的记录系统提交，通过 Git 历史与 [改动台账](changes.csv) 对应，未预填未知 SHA。
可按已知提交执行 `git show <SHA>` 核对源码改动。

## 验证

- GitHub API 确认仓库 public；首页整理提交推送后，远端 main SHA 与本地一致。
- 首页引用的本地链接已检查存在且纳入 Git。
- `python -m pytest -q tests/chemvector/test_direct_atom_pair_bond.py tests/chemvector/test_direct_atom_role.py`：6 passed。
- `python frontend/server.py --help`：退出码 0。
- 开发记录的 CSV 列数、状态、ID、文件链接与 Markdown 链接已通过静态检查；`git diff --check` 通过。

测试使用本地 `.venv-train`、CPU 环境；没有进行模型精度或吞吐评测。

## 技术决定

日期作为主目录、模块作为表内字段；单 CSV 最多 50 行数据，详细证据写 Markdown。
见 [D-20261019-01](../../../decisions/D-20261019-01-daily-records.md)。

## 问题与下一步

近期模型研究进度尚未在本轮重新核验。下一项为 T-20261019-04：读取已有运行结果、检查进程，
整理已完成、未完成和未采用项；完成后再决定后续研究任务。

本日记录范围内没有新模型实验，不创建空的 experiments.csv；也不将组件测试登记成训练实验。

台账：[任务](tasks.csv)、[改动](changes.csv)。当前状态见 [STATUS](../../../STATUS.md)。
