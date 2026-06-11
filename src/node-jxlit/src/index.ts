import * as np from "numpy-ts";
import { hrtime } from "node:process";

import { decode as decodeNative } from "../binding.js";
import type { DecodeMetadata as DecodeMetadataNative } from "../binding.js";
import {
  rebaseTelemetry,
  unixTimeMs,
  type DecodeTelemetry,
  type Measure,
  type NativeDecodeTelemetry,
} from "./telemetry.js";

export interface DecodeOptions {
  threads?: number;
  telemetry?: boolean;
}

export type { DecodeTelemetry, Measure };

export interface JxlitMeta {
  version: string;
  telemetry?: DecodeTelemetry;
}

export interface DecodeMetadata {
  _jxlit: JxlitMeta;
}

export interface DecodedImage {
  height: number;
  width: number;
  channels: number;
  pixels: np.NDArray;
  metadata: DecodeMetadata;
}

function metadataFromNative(metadata: DecodeMetadataNative): DecodeMetadata {
  return {
    _jxlit: {
      version: metadata._jxlit.version,
      telemetry: undefined,
    },
  };
}

function rebaseMetadata(
  metadata: DecodeMetadataNative,
  timebase: number,
  wallMs: number,
): DecodeMetadata {
  const nativeTelemetry = metadata._jxlit.telemetry as NativeDecodeTelemetry | undefined;
  if (nativeTelemetry === undefined) {
    return metadataFromNative(metadata);
  }
  const telemetry = rebaseTelemetry(
    nativeTelemetry,
    timebase,
    wallMs,
    "node_decode",
  );
  return {
    _jxlit: {
      version: metadata._jxlit.version,
      telemetry,
    },
  };
}

/**
 * Decode JPEG XL bytes into an f32 HWC pixel array.
 *
 * This is the idiomatic Node.js entry point. The native binding lives in
 * `binding.js` and should not be imported directly.
 */
export function decode(input: Buffer, options?: DecodeOptions): DecodedImage {
  const telemetry = options?.telemetry === true;
  const timebase = telemetry ? unixTimeMs() : 0;
  const start = telemetry ? hrtime.bigint() : 0n;

  const decoded = decodeNative(input, options);
  const wallMs = telemetry
    ? Number(hrtime.bigint() - start) / 1_000_000
    : 0;

  const pixels = np
    .array(decoded.pixels, "float32")
    .reshape(decoded.height, decoded.width, decoded.channels);

  const metadata = telemetry
    ? rebaseMetadata(decoded.metadata, timebase, wallMs)
    : metadataFromNative(decoded.metadata);

  return {
    height: decoded.height,
    width: decoded.width,
    channels: decoded.channels,
    pixels,
    metadata,
  };
}
