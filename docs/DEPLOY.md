# 构建与发布检查

LHA 的候选产物包括 Python CLI、wheel、源码包和应用镜像，不是托管服务。候选版本
必须从同一提交完成源码、安装包、容器和 Docker 执行后端检查，不能沿用旧提交的测试
数字。持续集成只构建并检查镜像，不会自动推送镜像仓库。

主机侧支持 Linux、macOS 和 WSL2。原生 Windows 尚未覆盖 POSIX 进程组回收和文件锁
行为，应使用 WSL2 或 Docker。

## 本地质量门禁

先在仓库根目录运行：

```bash
uv sync
uv run ruff check .
uv run python -m lha.release_claims
uv run python tools/verify_terminal_source_build.py \
  --root . \
  --evidence benchmarks/terminal_bench_2_1
uv run pyright src/lha
uv run pytest -q
LHA_RUNS_DIR=runs/_release uv run lha eval
```

`release_claims` 会核对公开文字与已提交证据；Terminal-Bench 源码重建检查会从评测提交
重新构建 wheel，并验证其摘要。两者都不能用“命令无法执行”替代通过。

CocoIndex 只能出现在 `lha.live_context` 内：

```bash
if grep -rnE "^[[:space:]]*(import|from)[[:space:]]+(cocoindex|cocoindex_code)" \
     --include='*.py' src/lha | grep -v "src/lha/live_context/"; then
  echo "CocoIndex import escaped the live_context facade"
  exit 1
fi
```

## 构建和检查 Python 包

```bash
uv build --clear
```

wheel 和源码包都必须包含索引流程和固定 Terminal-Bench 语料：

```bash
unzip -l dist/*.whl | grep 'lha/live_context/flows/common.py'
unzip -l dist/*.whl | grep 'lha/bench/resources/terminal_bench_2_1_corpus.json'
tar -tf dist/*.tar.gz | grep 'src/lha/live_context/flows/common.py'
tar -tf dist/*.tar.gz | grep 'src/lha/bench/resources/terminal_bench_2_1_corpus.json'
```

安装测试必须离开源码目录，避免 Python 意外导入当前工作区：

```bash
REPO_ROOT="$PWD"
CORPUS_CHECK='
from lha.bench.terminal_bench import load_terminal_bench_corpus
corpus = load_terminal_bench_corpus()
assert corpus.dataset == "terminal-bench/terminal-bench-2-1"
assert len(corpus.tasks) == 89
assert corpus.resolution_failures == ()
'

WHEEL_TMP="$(mktemp -d)"
cd "$WHEEL_TMP"
uv run --no-project --with "$REPO_ROOT"/dist/*.whl lha --version
uv run --no-project --with "$REPO_ROOT"/dist/*.whl \
  python -c "from lha.live_context.flows import common, experiments_flow, papers_flow, skills_flow"
uv run --no-project --with "$REPO_ROOT"/dist/*.whl python -c "$CORPUS_CHECK"
uv run --no-project --with "$REPO_ROOT"/dist/*.whl lha eval --quick

SDIST_TMP="$(mktemp -d)"
cd "$SDIST_TMP"
uv run --no-project --with "$REPO_ROOT"/dist/*.tar.gz lha --version
uv run --no-project --with "$REPO_ROOT"/dist/*.tar.gz \
  python -c "from lha.live_context.flows import common, experiments_flow, papers_flow, skills_flow"
uv run --no-project --with "$REPO_ROOT"/dist/*.tar.gz python -c "$CORPUS_CHECK"
uv run --no-project --with "$REPO_ROOT"/dist/*.tar.gz lha eval --quick

cd "$REPO_ROOT"
```

Linux 和 Windows 的开发环境通过 `uv` source 配置使用 CPU 版 PyTorch。wheel 元数据不会
携带这项 source 设置；在其他项目中安装 `context` extra 时，需要自行配置 CPU 索引，
或者直接使用应用镜像。

## 构建应用镜像

```bash
docker build -t lha:release .
docker run --rm lha:release lha --version
docker run --rm lha:release git --version
docker run --rm lha:release \
  python -c "import lha, cocoindex, sentence_transformers"
docker run --network none --rm lha:release lha eval
```

镜像使用固定摘要的 Python 和 `uv` 基础镜像，以非 root 用户 `lha` 运行，并包含
`context` extra 及固定版本的 `all-MiniLM-L6-v2` 文件，因此断网自测不需要下载模型。
镜像不包含外部 `ccc` 命令。

确认常见凭据和开发目录没有进入镜像：

```bash
docker run --rm lha:release python -c \
  "from pathlib import Path; assert not any((Path('/app') / p).exists() for p in ('.env', '.codex', '.claude', '.agents', '.mcp.json', 'auth.json', '.ssh', '.aws', '.config/gcloud', '.netrc', '.pypirc', 'dist'))"
```

不要通过 build argument、环境层或复制开发者主目录的方式传入认证信息。

需要保留运行记录时挂载独立 volume：

```bash
docker volume create lha-runs
docker run --rm \
  -v lha-runs:/app/runs \
  lha:release lha eval
```

## Docker 执行后端

应用镜像承载 LHA CLI；执行镜像则运行目标仓库声明的命令。默认
`python:3.12-slim` 不含 Pytest、`pytest-json-report` 或 Ruff，代码任务必须选择包含
全部检查工具的镜像。

真实容器测试：

```bash
LHA_DOCKER_TESTS=1 \
LHA_DOCKER_TEST_IMAGE=lha:release \
uv run pytest tests/test_sandbox.py -q
```

Docker 后端默认断网、只读根文件系统、受限 tmpfs、非 root UID、资源上限，并使用
`--cap-drop ALL`、`no-new-privileges` 和 init 进程。缺少 Docker daemon 或安全属性
检查失败，都属于发布失败。

## Terminal-Bench 代理镜像

CI 还会单独构建凭据代理，并确认固定非 root 用户和模块可导入：

```bash
docker build \
  --file docker/terminal-bench-proxy.Dockerfile \
  --tag lha-terminal-proxy:release \
  .
test "$(docker image inspect --format '{{.Config.User}}' \
  lha-terminal-proxy:release)" = "65532:65532"
docker run --rm --entrypoint python lha-terminal-proxy:release \
  -c "from lha.bench import terminal_proxy_server"
```

这一步只检查镜像，不运行 Terminal-Bench 题目。

## 凭据边界

普通 Codex 后端把必要认证复制到单次调用的临时 `CODEX_HOME`，并在进程组停止后清理。
不要把开发者 home 目录挂载进应用镜像。

基准容器中的认证必须：

- 只在任务启动时注入；
- 不进入镜像、仓库、命令参数、日志或报告；
- 在清理临时文件前先停止 Codex 进程组；
- 只记录版本、模型参数、镜像摘要和预算。

只有外层已经是一次性隔离容器，并设置 `LHA_CODEX_EXTERNAL_SANDBOX=1` 时，Codex 才能
使用 `danger-full-access`。

## 与 CI 的对应关系

`.github/workflows/ci.yml` 包含三组任务：

- Ubuntu `gate`：声明校验、源码重建、静态检查、完整测试、包构建、空目录安装和自测；
- macOS：Ruff、Pyright、完整测试和自测；
- Docker：应用镜像、断网自测、代理镜像和真实执行后端测试。

本页命令应与 CI 同步。增加或删除发布检查时，需要在同一修改中更新两处。
