import { decode as decodeNative } from "../pkg/jxlit_wasm_bindings.js";

/**
 * Decode JPEG XL bytes into a pixel buffer.
 *
 * This is the idiomatic WebAssembly entry point. The wasm-bindgen output in
 * `pkg/` should not be imported directly.
 */
export function decode(input: Uint8Array): Uint8Array {
  return decodeNative(input);
}
