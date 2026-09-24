# =============================================================================
#  LCR-Agent —— 常用开发命令
#  依赖 GNUMake。Windows 用户可使用 `choco install make` 或在 Git Bash 中运行。
# =============================================================================

.DEFAULT_GOAL := help
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

PY        := poetry run python
PYTEST    := poetry run pytest
UVICORN   := poetry run uvicorn
APP       := api.main:app
APP_DIR   := src
HOST      ?= 0.0.0.0
PORT      ?= 8000
CONFIG    ?= configs/default.yaml

.PHONY: help install dev run run-prod test test-unit test-cov lint fmt fmt-check \
        typecheck check index download-cuad clean clean-all tree

## help: 显示所有可用目标
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /' | column -t -s ':'

# ---------------------------------------------------------------- 依赖 / 环境
## install: 安装全部依赖（含 dev）
install:
	poetry install --with dev

## dev: 安装依赖并以热重载模式启动 API（开发常用入口）
dev: install
	$(UVICORN) $(APP) --app-dir $(APP_DIR) --reload --host $(HOST) --port $(PORT)

# ---------------------------------------------------------------------- 运行
## run: 以热重载模式启动 API 服务
run:
	$(UVICORN) $(APP) --app-dir $(APP_DIR) --reload --host $(HOST) --port $(PORT)

## run-prod: 以多 worker 方式启动 API 服务（无热重载）
run-prod:
	$(UVICORN) $(APP) --app-dir $(APP_DIR) --host $(HOST) --port $(PORT) --workers 4

# ---------------------------------------------------------------------- 测试
## test: 运行全部测试
test:
	$(PYTEST)

## test-unit: 只运行不依赖外部模型/网络的单元测试
test-unit:
	$(PYTEST) -m "not integration and not slow"

## test-cov: 运行测试并生成覆盖率报告（终端 + htmlcov/）
test-cov:
	$(PYTEST) --cov=src --cov-report=term-missing --cov-report=html

# ------------------------------------------------------------------ 质量保障
## lint: 静态检查
lint:
	poetry run ruff check src tests

## fmt: 自动格式化与导入排序
fmt:
	poetry run ruff format src tests
	poetry run ruff check --fix src tests

## fmt-check: 仅校验格式，不修改文件（CI 使用）
fmt-check:
	poetry run ruff format --check src tests
	poetry run ruff check src tests

## typecheck: mypy 类型检查
typecheck:
	poetry run mypy $(APP_DIR)

## check: 一键跑完 lint + typecheck + test（提交前自检）
check: fmt-check typecheck test

# ------------------------------------------------------------------ 数据 / 索引
## download-cuad: 下载并解压 CUAD 数据集到 data/cuad/
download-cuad:
	$(PY) scripts/download_cuad.py

## index: 基于 CUAD 构建 BM25 + 向量索引
index:
	$(PY) scripts/build_index.py --config $(CONFIG)

# ---------------------------------------------------------------------- 其他
## tree: 打印项目文件树（忽略缓存目录）
tree:
	find . -type d \( -name __pycache__ -o -name .venv -o -name .git \
		-o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -prune -o -print \
		| sort | sed 's|[^/]*/|  |g'

## clean: 清理 Python 缓存与测试产物
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov coverage.xml

## clean-all: clean + 删除向量索引
clean-all: clean
	rm -rf data/index
