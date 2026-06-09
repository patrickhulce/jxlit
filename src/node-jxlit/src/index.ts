import { decode as decodeNative } from "../binding.js";

/**
 * Decode JPEG XL bytes into a pixel buffer.
 *
 * This is the idiomatic Node.js entry point. The native binding lives in
 * `binding.js` and should not be imported directly.
 */
export function decode(input: Buffer): Buffer {
  return decodeNative(input);
}
