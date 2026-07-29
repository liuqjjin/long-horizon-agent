# 评测证据目录

本目录只保存已经提交、可以由程序校验的评测证据。运行过程中生成的文件位于
`runs/`，不会因为存在于本机就成为公开结果。

## 当前状态

| 评测 | 状态 | 可公开结论 |
|---|---|---|
| 早期内部消融 | 历史记录 | 仅用于核对旧协议，不代表当前实现 |
| 两次未登记的 schema-v4 运行 | 已披露 | 缺少事前登记和一次性远端 Git 开始记录，不是正式结果 |
| 两次正式 schema-v4 尝试 | `ABANDONED` | 均无可发布的消融结果 |
| Terminal-Bench 2.1 固定子集 | 已完成 | 7/20：7 `PASS`、9 `FAIL`、4 `ERROR` |

正式内部消融只有在登记状态为 `COMPLETED`、完整报告和原始证据同时提交后，
才会成为当前项目结果。现在尚无满足这些条件的内部消融报告。

## 内部消融证据

### 历史 schema-v2 报告

- [`ablation_report.json`](ablation_report.json)：早期消融的机器可读记录。
- [`ablation_report.md`](ablation_report.md)：由该 JSON 生成的文本报告。
- [`horizon_report.json`](horizon_report.json)：早期任务单元、完整重复和组合推演记录。
- [`horizon_report.md`](horizon_report.md) 与
  [`horizon_curve.svg`](horizon_curve.svg)：对应的文本和图形。

这些文件使用旧评分边界保留历史运行，不能作为当前 README 或简历中的实验数字。

### 未登记的 schema-v4 运行

[`formal_ablation_history/`](formal_ablation_history/) 保存两份 schema-v4 格式报告。
它们产生在正式尝试登记和一次性远端 Git 开始记录建立之前，登记簿将其记为
`UNREGISTERED_RUN_RECORDED`。保留这些文件是为了披露历史，不把它们转成正式结果。

### 正式尝试登记

[`formal_ablation_attempts.json`](formal_ablation_attempts.json) 是正式尝试登记簿，
[`formal_ablation_manifest.json`](formal_ablation_manifest.json) 固定 17 个任务及其语料摘要。

登记簿目前包含两次 `REGISTERED → ABANDONED`：

1. 第一次尝试在创建一次性远端 Git 引用时被 GitHub 规则阻止，未进入模型评测。
2. 第二次尝试在 Codex 用量耗尽后中止；按协议保留已有文件，不恢复也不把部分输出发布为结果。

两次尝试都没有 `COMPLETED` 事件。后续正式运行必须使用新的登记和输出目录，
并继续保留这两条失败记录。

内部消融中，`trust` 与 `gate` 对同一份首轮补丁评分；内部检查只负责决定是否放行，
独立评分器在新的仓库副本中运行固定测试。方法和错误处理见
[消融说明](../docs/ABLATION.md)，任务长度推演口径见
[Horizon 说明](../docs/HORIZON.md)。

## Terminal-Bench 2.1 固定 20 题子集

[`terminal_bench_2_1/`](terminal_bench_2_1/) 是已经完成的公开证据包。
结果为 7 个 `PASS`、9 个 `FAIL`、4 个 `ERROR`，四个 `ERROR` 保留在分母中，
因此结果为 7/20。这不是完整数据集或排行榜成绩。

主要文件：

- [`evidence.json`](terminal_bench_2_1/evidence.json)：证据索引和摘要绑定。
- [`protocol.json`](terminal_bench_2_1/protocol.json)：固定任务、预算和运行协议。
- [`scored_manifest.json`](terminal_bench_2_1/scored_manifest.json)：20 个正式任务。
- [`records.json`](terminal_bench_2_1/records.json)：任务结果记录。
- [`summary.json`](terminal_bench_2_1/summary.json) 与
  [`summary.md`](terminal_bench_2_1/summary.md)：机器和文本汇总。
- [`trials/`](terminal_bench_2_1/trials/)：每个任务的公开结果。

16 个 `PASS` 或 `FAIL` 任务保留官方结果 JSON。四个 `ERROR` 使用脱敏记录，
并通过 SHA-256 绑定私有原文；公开仓库不能还原未公开的异常内容。

Harbor 适配器直接运行 Codex，不经过 LHA 的内部放行或修复流程。因此 7/20
只说明这次固定子集运行，不是 LHA 校验机制的成绩。完整解释见
[评测说明](../docs/BENCHMARKS.md)。

## 校验证据

修改公开数字或报告前运行：

```bash
uv run python -m lha.release_claims
uv run python tools/run_terminal_bench_2_1.py \
  validate benchmarks/terminal_bench_2_1
```

第一条命令从已提交记录重新计算统计量，并检查文档与证据是否一致。
第二条命令只校验已提交的 Terminal-Bench 包，不会重新运行评测任务。

探索性运行可以用于调试，但不写入本目录的当前结果。适配器存在也不等于已经获得
Terminal-Bench 或 SWE-bench 成绩。
