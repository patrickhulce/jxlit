fn main() {
    // pyo3's `extension-module` feature intentionally does not link libpython:
    // the CPython symbols are resolved at runtime by the interpreter that loads
    // the extension. On macOS, linking such a cdylib therefore requires telling
    // the linker to allow undefined symbols (resolved dynamically at load time).
    //
    // `maturin develop`/`maturin build` inject this automatically, but a plain
    // `cargo build` does not, which makes building this crate via cargo fail with
    // "symbol(s) not found for architecture arm64". Emitting the flag here (scoped
    // to this cdylib, macOS only) keeps `cargo build`/`cargo build --workspace`
    // working without affecting any other crate; it's idempotent with maturin.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-cdylib=-undefined");
        println!("cargo:rustc-link-arg-cdylib=dynamic_lookup");
    }
}
