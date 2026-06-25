# jxlit

A fast Rust-first, GPU-accelerated JPEG XL decoder with bindings for Python, Node.js, and WebAssembly.

## When is this useful?

jxlit shines in two situations:

- **You want pixels in GPU memory.** When the decoded image is headed straight for a GPU
  (rendering, ML inference, GPU compositing), jxlit decodes directly into device VRAM and
  avoids the host round-trip that CPU-only decoders pay for. In that case jxlit is the
  fastest option here.
- **You want a portable, Rust-only decode path.** If you can't ship `libjxl` (the C++
  reference decoder) — e.g. WebAssembly, locked-down build environments, or pure-Rust
  deployments — jxlit gives you a self-contained decoder with sane multi-worker scaling.

If you only need pixels in **host RAM** and can link against the C++ reference decoder,
`libjxl` is still the fastest choice and you should prefer it. jxlit does not try to beat
`libjxl` on CPU; it trades some CPU throughput for GPU-native output and a Rust-only stack.

### How it works (hybrid CPU/GPU)

jxlit uses a hybrid pipeline that splits the JPEG XL decode across the CPU and GPU:

- **CPU:** entropy decoding (ANS) and the inverse DCT, which are branchy, serial, and a poor
  fit for GPUs.
- **GPU:** the final, highly parallel color stages (upsampling, XYB → RGB, colorspace and
  transfer-function conversion) run on-device, so the result lands in VRAM ready to use.

This keeps the inherently sequential work on the CPU while offloading the embarrassingly
parallel pixel math to the GPU and skipping the upload step entirely.

## Benchmarks

Aggregate decode throughput (MP/s) as the number of parallel workers scales from 1 to 8.

### Decoding to CPU memory

When the output needs to live in host RAM, `libjxl` leads; jxlit lands mid-pack while
staying pure Rust.

![Worker scaling, decoding to CPU memory](https://github.com/user-attachments/assets/08bb3824-468c-417e-81a6-50474140e193)

> Benchmarks from a MacBook Pro M3 Pro.

### Decoding to GPU memory

When the output is destined for the GPU, jxlit's hybrid pipeline delivers the highest
throughput by decoding straight into VRAM.

![Worker scaling, decoding to GPU memory](https://github.com/user-attachments/assets/ca3ce86c-aa9c-47af-afdf-aee63bb758bc)

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
make lint         # fmt, clippy, ruff, eslint, prettier
make lint-fix     # auto-fix lint issues across all languages
make typecheck    # mypy, tsc
make format       # apply formatters across all languages
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
├── python-jxlit/        # PyO3 bindings + idiomatic Python package (jxlit/)
├── node-jxlit/          # napi-rs bindings + TypeScript wrapper (src/ -> dist/)
└── wasm-jxlit/          # wasm-bindgen bindings + TypeScript wrapper (src/ -> dist/)
```
