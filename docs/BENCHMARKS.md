# 评测说明

LHA 有三条互相独立的评测路径。它们回答的问题不同，结果不能合并使用。

| 评测 | 判定方 | 回答的问题 |
|---|---|---|
| 仓库自测 | LHA 注册的检查器 | 固定工作流是否仍能正常执行 |
| 校验消融 | 独立 Docker 评分器 | 门禁和有限修复能否减少错误交付 |
| 外部基准 | 基准官方执行器 | 固定模型在外部任务集上的表现 |

## 仓库自测

```bash
uv sync
LHA_RUNS_DIR=runs/_eval uv run lha eval
```

六条固定流程覆盖代码检查、审批后恢复、过期索引、上下文后端不可用、实验复现，以及
无法达到指标时必须失败。它是仓库回归测试，不是外部基准成绩。

## 校验消融

消融实验让 `trust`、`gate` 和 `verify` 共用同一个初始补丁，再由独立评分器把固化后的
源码改动应用到干净仓库并运行原始测试。评分器不复用内部门禁结论。

仓库保存了一份旧 schema v2 报告，供历史追溯：

- [`benchmarks/ablation_report.json`](../benchmarks/ablation_report.json)
- [`benchmarks/ablation_report.md`](../benchmarks/ablation_report.md)

当前正式登记历史尚无完整的 schema v4 `COMPLETED` 证据。已经终止的尝试都是
`ABANDONED`，不构成结果，不能拼接、续跑或引用其中的局部数字。只有完整日程、原始
证据、一次性远端 Git 开始记录和 `COMPLETED` 事件一起通过校验后，才能发布当前结果。

具体协议见 [校验消融](ABLATION.md)，任务级和组合统计见
[长链统计](HORIZON.md)。

## Terminal-Bench 2.1 固定 20 题子集

正式运行通过 Harbor 使用官方
[`terminal-bench/terminal-bench-2-1`](https://hub.harborframework.com/datasets/terminal-bench)
数据集。

| 结果 | 数量 |
|---|---:|
| PASS | 7 |
| FAIL | 9 |
| ERROR | 4 |
| 合计 | 20 |

因此固定子集结果为 **7/20**，四个 `ERROR` 全部保留在分母中。这不是完整数据集成绩，
也不是排行榜成绩。

### 固定协议

- 先按 `(SHA-256(instance_id), instance_id)` 排序全部 89 个任务；
- 前 20 个组成计分子集，随后 3 个组成冒烟子集；
- 三个冒烟任务全部结束后，才开始计分任务；
- 每个计分任务只运行一次，只有一个 attempt，Harbor 使用 `--max-retries 0`；
- 观察结果后没有重跑任何一道计分题；
- 模型为 `gpt-5.5`，推理强度为 `xhigh`；
- 运行环境为 Harbor `0.20.0` 和 Codex CLI `0.141.0`。

任务清单、时间限制、镜像摘要、二进制摘要、请求上限和命令约束都保存在：

- [`protocol.json`](../benchmarks/terminal_bench_2_1/protocol.json)
- [`scored_manifest.json`](../benchmarks/terminal_bench_2_1/scored_manifest.json)

### ERROR 处理

两个 `ERROR` 来自正式运行期间发现的适配器问题：

- `caffe-cifar-10` 超过 Harbor 默认异步输出单行上限；
- `video-processing` 被当时过严的请求计数约束拒绝。

当前代码已修复这两处问题，但没有重跑正式题目。另外两个 `ERROR`，
`configure-git-webserver` 和 `make-doom-for-mips`，包含明确的 Codex 错误事件，同样
没有改写成普通失败或从分母删除。

### 这次运行没有测什么

Terminal-Bench 适配器在任务容器内直接调用 Codex，模型输出没有经过 LHA 的门禁或修复
循环。因此 **7/20 只说明这次固定外部任务运行的结果**，不能用来声称 LHA 拦截了多少
错误补丁、误拒了多少正确补丁或修复成功了多少次。这些问题只能由独立的校验消融回答。

### 公开证据

公开 schema v4 证据位于
[`benchmarks/terminal_bench_2_1/`](../benchmarks/terminal_bench_2_1/)。
其中 16 个 `PASS`/`FAIL` 任务保留官方原始 JSON；四个 `ERROR` 只公开脱敏投影和私有
原件 SHA-256，仓库无法还原私有异常栈、凭据或用户路径。

下面的命令只校验已提交证据，不会重新运行基准任务：

```bash
uv run python tools/run_terminal_bench_2_1.py \
  validate benchmarks/terminal_bench_2_1

uv run python tools/verify_terminal_source_build.py \
  --root . \
  --evidence benchmarks/terminal_bench_2_1
```

## SWE-bench Verified 适配器

`lha.bench.swebench` 可以生成官方预测字段并解析评测报告。重复 instance ID 会被拒绝，
执行器错误保留在分母中，适配器不会读取保留的 `FAIL_TO_PASS` 测试。

本仓库没有发布 SWE-bench 成绩。
