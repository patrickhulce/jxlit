# jxlit

A blazing fast Rust-first JPEG XL decoder with bindings for Python, Node.js, and WebAssembly.

## Prerequisites

- [Rust](https://rustup.rs/) (stable)
- `rustup target add wasm32-unknown-unknown`
- [uv](https://docs.astral.sh/uv/)
- [pnpm](https://pnpm.io/)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) (`cargo install wasm-pack`)

## Development

The [Makefile](Makefile) is the primary interface:

```bash
make              # build all targets
make lint         # fmt, clippy, ruff, prettier
make typecheck    # mypy, tsc
make test         # run all tests
make test-rust    # cargo test only
make test-python  # pytest only
make test-node    # node tests only
make test-wasm    # wasm tests only
```

## API

All bindings expose a single function for now:

```text
decode(bytes) -> bytes   # pixel buffer (stub returns empty)
```

### Python

```python
from jxlit import decode

pixels = decode(jxl_bytes)
```

### Node.js

```javascript
const {decode} = require('@jxlit/node')

const pixels = decode(jxlBuffer)
```

### WebAssembly

```javascript
import {decode} from '@jxlit/wasm'

const pixels = decode(jxlBytes)
```

## Layout

```text
src/
├── rust-jxlit/          # language-agnostic core (std only)
├── python-jxlit/        # PyO3 bindings + Python package
├── node-jxlit/          # napi-rs bindings + npm package
└── wasm-jxlit/          # wasm-bindgen bindings + npm package
```
