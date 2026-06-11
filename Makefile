.PHONY: all build lint lint-fix typecheck test format benchmark trace \
        build-rust build-python build-node build-wasm build-bench-rust build-bench-python \
        lint-rust lint-python lint-node lint-wasm \
        lint-fix-rust lint-fix-python lint-fix-node lint-fix-wasm \
        typecheck-python typecheck-node typecheck-wasm \
        format-rust format-python format-node format-wasm \
        test-rust test-python test-node test-wasm \
        benchmark-rust benchmark-python benchmark-node benchmark-wasm

WORKERS ?= 2
THREADS ?= 2
ITERATIONS ?= 25
FILE ?= assets/frame_4K_10bit_e1_d0p5_fd4.jxl
BENCHMARK_ARGS = --workers $(WORKERS) --iterations $(ITERATIONS) --file $(FILE) --threads $(THREADS) $(if $(filter planar,$(LAYOUT)),--layout planar,)

all: build lint typecheck test

build: build-rust build-python build-node build-wasm

build-rust:
	cargo build --workspace --exclude jxlit_python_bindings --exclude jxlit_wasm_bindings

build-python:
	cd src/python-jxlit && uv sync && uv run maturin develop --uv

build-node:
	pnpm --dir src/node-jxlit build

build-wasm:
	pnpm --dir src/wasm-jxlit build

lint: lint-rust lint-python lint-node lint-wasm

lint-rust:
	cargo fmt --all -- --check
	cargo clippy -p jxlit -p jxlit_node_bindings -p jxlit_wasm_bindings -- -D warnings

lint-python:
	cd src/python-jxlit && uv run ruff check .
	cd src/python-jxlit && uv run ruff format --check .

lint-node:
	pnpm --dir src/node-jxlit lint

lint-wasm:
	pnpm --dir src/wasm-jxlit lint

lint-fix: lint-fix-rust lint-fix-python lint-fix-node lint-fix-wasm

lint-fix-rust:
	cargo fmt --all
	cargo clippy --fix --allow-dirty --allow-staged \
		-p jxlit -p jxlit_node_bindings -p jxlit_wasm_bindings -- -D warnings

lint-fix-python:
	cd src/python-jxlit && uv run ruff check --fix .
	cd src/python-jxlit && uv run ruff format .

lint-fix-node:
	pnpm --dir src/node-jxlit lint:fix

lint-fix-wasm:
	pnpm --dir src/wasm-jxlit lint:fix

typecheck: typecheck-python typecheck-node typecheck-wasm

typecheck-python:
	cd src/python-jxlit && uv run mypy jxlit tests

typecheck-node:
	pnpm --dir src/node-jxlit typecheck

typecheck-wasm:
	pnpm --dir src/wasm-jxlit typecheck

format: format-rust format-python format-node format-wasm

format-rust:
	cargo fmt --all

format-python:
	cd src/python-jxlit && uv run ruff format .

format-node:
	pnpm --dir src/node-jxlit format

format-wasm:
	pnpm --dir src/wasm-jxlit format

test: test-rust test-python test-node test-wasm

test-rust:
	cargo test -p jxlit -p jxlit_wasm_bindings

test-python: build-python
	cd src/python-jxlit && uv run pytest

test-node: build-node
	pnpm --dir src/node-jxlit test

test-wasm: build-wasm
	pnpm --dir src/wasm-jxlit test

build-bench-rust:
	cargo build --release -p jxlit --bin jxlit-benchmark

build-bench-python:
	cd src/python-jxlit && uv sync && uv run maturin develop --release --uv

benchmark: build-bench-rust build-bench-python build-node build-wasm
	python3 scripts/benchmark.py $(BENCHMARK_ARGS)

benchmark-rust: build-bench-rust
	python3 scripts/benchmark.py $(BENCHMARK_ARGS) --langs rust

benchmark-python: build-bench-python
	python3 scripts/benchmark.py $(BENCHMARK_ARGS) --langs python

benchmark-node: build-node
	python3 scripts/benchmark.py $(BENCHMARK_ARGS) --langs node

benchmark-wasm: build-wasm
	python3 scripts/benchmark.py $(BENCHMARK_ARGS) --langs wasm

trace:
	python3 scripts/trace.py
