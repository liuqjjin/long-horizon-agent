# 长链统计

`lha horizon` 从消融报告读取“是否正确交付”的标签，并分别计算单元级结果、完整语料
重复聚合和组合推演。三者的统计单位不同，不能混成一个样本量。

## 配对单元

一个单元是同一 `(任务, 重复序号)` 下的 `trust` 与 `verify` 配对结果。schema v4 的
`true_success` 要求系统实际交付了产物，而且独立评分器判定产物正确。

同一任务的多次重复不能当成互相独立的推断样本。schema v4 先计算每个任务内
`verify - trust` 的平均配对差，再以任务为单位做双侧精确符号翻转检验。每个任务只有
一份推断权重；原始单元的 discordance 仍会列出，但只作描述，不再给出单元级 McNemar
p 值。

`ERROR` 没有真值标签，既不改写成成功，也不改写成失败；报告会分开列出计划单元、
可用单元和错误单元。

## 完整语料重复聚合

消融实验会分别执行每个 `(任务, 重复序号, 条件)` 单元。`horizon` 在这些执行结束后，
把相同重复序号下的任务标签汇总成一个“完整语料重复聚合”。只有该重复序号下每个任务
都成功，聚合结果才算成功；一个重复中出现多个失败单元，也只形成一个失败聚合。

这个聚合不是实际执行的共享状态长任务。各任务没有共享工作目录、上下文、检查点或前序
任务产生的状态，任务之间也没有在一次运行中顺序传递错误。它只能回答“同一重复序号下，
整套任务是否全部成功”，不能证明系统完成了一条真实长任务链。

聚合级比较以完整重复为配对单位，使用双侧精确 McNemar 检验。它和任务聚类的单元级
检验回答的问题不同，p 值没有必须相等的关系。若某次重复含有 `ERROR` 或缺失任务，
该重复不进入完整语料聚合比较。

## 组合推演

组合曲线把各任务的实测成功率代入独立步骤模型，估算长度为 `k` 时所有步骤都成功的
概率。`src/lha/horizon.py::compounding_curve` 会对大小为 `k` 的任务子集均匀取平均，
并用任务 bootstrap 表示结果对当前任务构成的敏感程度。

当某个任务因 `ERROR` 少了一次观测，报告会保留它的实际样本数，不会假设每个任务的
分母相同。

这条曲线有三个明确边界：

- 它没有执行新的长任务；
- 它不增加独立样本；
- 它没有 McNemar p 值。

因此，组合曲线只能称为基于实测单元的推演，不能当作新增实验结果。

## 标签和区间

| 条件 | 使用的字段 | 含义 |
|---|---|---|
| `trust-chain` | `trust.true_success` | 任一步错误交付都会使链条失败 |
| `verify-chain` | `verify.true_success` | 检查失败后允许在预算内修复 |

链条不使用单独的 `artifact_correct`。正确产物若被系统拒绝，就没有被交付，该步仍是
`true_success=false`。

全零或全一的完整语料聚合比例使用 Wilson 区间。边界样本若直接使用百分位 bootstrap，会
得到误导性的零宽区间。

## JSON 兼容字段

为避免破坏已有读取工具，JSON 目前保留 `independent_episode_count`、
`estimands.episode` 和 `episodes` 这些旧字段名。它们表示完整语料重复聚合及其数量，
不表示相互独立执行的 episode，也不表示实际共享状态长任务。新文档和报告展示统一使用
“完整语料重复聚合”。

schema v1–v3 的历史文件继续保留 `estimands.cell.mcnemar_p` 和原有文本，目的是保证
已提交证据可以按原字节复现。schema v4 的 `estimands.cell` 改为记录任务数、非零任务
数和 `task_cluster_sign_flip_p`，不会同时发布单元级 McNemar p 值。

## 当前状态

当前仓库没有可发布的 schema v4 `COMPLETED` 消融报告，因此也没有当前版本的单元级、
完整语料重复聚合或组合曲线数字。`ABANDONED` 尝试不是结果，不能作为 `lha horizon` 的正式
输入。

仓库保留的旧 horizon 文件来自历史 schema v2 报告。旧协议曾把某些未交付的正确产物
计入成功，这些文件只能用于追溯，不能作为当前链条成功率：

- [`benchmarks/horizon_report.json`](../benchmarks/horizon_report.json)
- [`benchmarks/horizon_report.md`](../benchmarks/horizon_report.md)

`data/long_tasks/` 下的五个多文件流程是实际执行的恢复测试，但它们不是这里的统计单元，
也不会增加 horizon 样本。反过来，完整语料重复聚合也不能替代这些真实流程测试。

## 生成报告

只有完整、已校验且对应 `COMPLETED` 事件的 schema v4 报告才能生成正式 horizon：

```bash
uv run lha horizon \
  --from-report runs/formal_ablation/<attempt-id>/ablation_report.json \
  --out runs/horizon
```

输出包括：

```text
runs/horizon/horizon_report.json
runs/horizon/horizon_report.md
runs/horizon/horizon_curve.svg
```

引用结果前，应确认模型和运行环境与登记协议一致，所有 `ERROR` 都已计入覆盖情况，
单元推断按任务聚类，完整语料重复聚合使用自己的配对单位，并且组合部分仍明确写着
新增样本数为零。
