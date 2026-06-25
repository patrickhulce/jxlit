# jxlit candidate

This candidate has no build of its own. It is a thin config shim that runs the
repository's `jxlit-benchmark` binary, which is built by:

```bash
make build-bench-rust   # cargo build --release -p jxlit --bin jxlit-benchmark
```

The binary lives at `target/release/jxlit-benchmark` (repo root). The
[`manifest.toml`](manifest.toml) here declares three configs the compare
orchestrator runs:

- `cpu` / `cpu` - full CPU pipeline (baseline against the other CPU candidates)
- `gpu` / `cpu` - hybrid GPU compute, pixels downloaded to CPU
- `gpu` / `gpu` - hybrid GPU compute, pixels left in a GPU buffer

The `gpu` configs require a wgpu adapter (Metal on macOS, Vulkan on Linux). When
no GPU is detected the orchestrator skips them. With no adapter the binary itself
silently falls back to CPU, so those rows would otherwise be mislabeled.
