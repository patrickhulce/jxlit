import * as np from "numpy-ts";

import { decode as decodeNative } from "../binding.js";

export interface DecodedImage {
  height: number;
  width: number;
  channels: number;
  pixels: np.NDArray;
}

/**
 * Decode JPEG XL bytes into an f32 HWC pixel array.
 *
 * This is the idiomatic Node.js entry point. The native binding lives in
 * `binding.js` and should not be imported directly.
 */
export function decode(input: Buffer): DecodedImage {
  const decoded = decodeNative(input);
  const pixels = np
    .array(decoded.pixels, "float32")
    .reshape(decoded.height, decoded.width, decoded.channels);

  return {
    height: decoded.height,
    width: decoded.width,
    channels: decoded.channels,
    pixels,
  };
}
