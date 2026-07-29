# 评测证据目录

本目录只保存已经提交、可以由程序校验的评测证据。运行过程中生成的文件位于
`runs/`，不会因为存在于本机就成为公开结果。

## 当前状态

| 评测 | 状态 | 可公开结论 |
|---|---|---|
| 当前正式 schema-v4 内部消融 | `COMPLETED` | 计划 204 组，204 组可用，0 组 ERROR |
| 早期内部消融 | Git 历史记录 | 仅用于追溯旧协议，不参与当前结果 |
| 两次未登记的 schema-v4 运行 | 已披露 | 缺少事前登记和一次性远端 Git 开始记录，不是正式结果 |
| 两次已放弃的正式尝试 | `ABANDONED` | 保留失败原因，不参与当前结果 |
| Terminal-Bench 2.1 固定子集 | 已完成 | 7/20：7 `PASS`、9 `FAIL`、4 `ERROR` |

## 内部消融证据

### 当前正式 schema-v4 报告

当前结果对应正式尝试
`56fed4eb17d7e43fbb3d73e21b8ab464489bc19e1ae05731137e7e157ad5f00f`，
模型为 `gpt-5.3-codex-spark`，推理强度为 `high`。评测源码提交为
`4c7452776b9e12322f15df95f236ce6daf455de7`，登记提交为
`95cc1a109c6a4b479e3390a95f437d31d94060f8`。

17 个固定缺陷各重复 12 次，共 204 个配对单元，全部可用且没有 `ERROR`：

- `trust` 直接交付 201 个正确补丁和 3 个错误补丁；
- `gate` 接受 201 个正确补丁，拦截 3 个错误补丁，没有误放或误拒；
- `verify` 在 3 个单元中各修复 1 次，最终 204/204 正确交付。

登记簿中的 `COMPLETED` 事件通过报告摘要
`1b06a6608c9b4c532d3e74ebe251057ec142d2994341a0b18a5b95c01aad954c`
和指纹
`d21b7258dc1c22c05c3e05d322228b5eb80f65f34525e6f611ecacd20c58884a`
绑定这次结果。

主要文件：

- [`ablation_report.json`](ablation_report.json)：机器可读报告和 612 条条件记录。
- [`ablation_report.md`](ablation_report.md)：由 JSON 生成的统计报告。
- [`formal_run.json`](formal_run.json)：运行头和正式协议绑定。
- [`input_snapshots/`](input_snapshots/)：17 个任务的固定输入快照。
- [`artifacts/`](artifacts/)：评分使用的内容寻址补丁。
- [`scorer_evidence/`](scorer_evidence/)：独立 Docker 评分器证据。
- [`llm_call_receipts/`](llm_call_receipts/)：207 份模型调用回执。
- [`results/`](results/)：204 个开始记录和 204 个终态单元。
- [`horizon_report.json`](horizon_report.json)、
  [`horizon_report.md`](horizon_report.md) 和
  [`horizon_curve.svg`](horizon_curve.svg)：任务聚类、完整重复聚合和组合推演。

### 未登记的 schema-v4 运行

[`formal_ablation_history/`](formal_ablation_history/) 保存两份 schema-v4 格式报告。
它们产生在正式尝试登记和一次性远端 Git 开始记录建立之前，登记簿将其记为
`UNREGISTERED_RUN_RECORDED`。保留这些文件是为了披露历史，不把它们转成正式结果。

### 登记历史

[`formal_ablation_attempts.json`](formal_ablation_attempts.json) 是正式尝试登记簿，
[`formal_ablation_manifest.json`](formal_ablation_manifest.json) 固定 17 个任务及其语料摘要。

登记簿保留两次 `REGISTERED → ABANDONED`：

1. 第一次尝试在创建一次性远端 Git 引用时被 GitHub 规则阻止，未进入模型评测。
2. 第二次尝试在 Codex 用量耗尽后中止；按协议保留已有文件，不恢复也不把部分输出发布为结果。

第三次尝试使用新的模型、源码和协议登记，完整执行后写入 `COMPLETED`。失败记录继续
保留，但没有拼接进本次 204 个单元。

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
