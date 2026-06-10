import {
  decode as decodeNative,
  DecodeOptions as DecodeOptionsNative,
} from "../pkg/jxlit_wasm_bindings.js";

export interface DecodeOptions {
  /** Ignored on WASM; threading is not available in this build. */
  threads?: number;
}

export interface DecodedImage {
  height: number;
  width: number;
  channels: number;
  pixels: Float32Array;
}

/**
 * Decode JPEG XL bytes into an f32 HWC pixel buffer.
 *
 * This is the idiomatic WebAssembly entry point. The wasm-bindgen output in
 * `pkg/` should not be imported directly.
 */
export function decode(
  input: Uint8Array,
  options?: DecodeOptions,
): DecodedImage {
  const nativeOptions =
    options === undefined
      ? undefined
      : new DecodeOptionsNative(options.threads ?? undefined);

  const decoded = decodeNative(input, nativeOptions);

  const result: DecodedImage = {
    height: decoded.height,
    width: decoded.width,
    channels: decoded.channels,
    pixels: decoded.pixels,
  };

  decoded.free();
  return result;
}
