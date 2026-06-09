.PHONY: all build lint typecheck test \
        build-rust build-python build-node build-wasm \
        lint-rust lint-python lint-node \
        typecheck-python typecheck-node \
        test-rust test-python test-node test-wasm

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

lint: lint-rust lint-python lint-node

lint-rust:
	cargo fmt --all -- --check
	cargo clippy -p jxlit -p jxlit_node_bindings -p jxlit_wasm_bindings -- -D warnings

lint-python:
	cd src/python-jxlit && uv run ruff check .

lint-node:
	pnpm --dir src/node-jxlit exec prettier --check .

typecheck: typecheck-python typecheck-node

typecheck-python:
	cd src/python-jxlit && uv run mypy jxlit tests

typecheck-node:
	pnpm --dir src/node-jxlit exec tsc --noEmit

test: test-rust test-python test-node test-wasm

test-rust:
	cargo test -p jxlit -p jxlit_wasm_bindings

test-python: build-python
	cd src/python-jxlit && uv run pytest

test-node: build-node
	pnpm --dir src/node-jxlit test

test-wasm: build-wasm
	pnpm --dir src/wasm-jxlit test
