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

export type PixelLayout = "interleaved" | "planar";

export interface DecodeOptions {
  threads?: number;
  telemetry?: boolean;
  layout?: PixelLayout;
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
  pixels: Float32Array;
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
  const nativeTelemetry = metadata._jxlit.telemetry as
    | NativeDecodeTelemetry
    | undefined;
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
 * Decode JPEG XL bytes into an f32 pixel buffer.
 *
 * This is the idiomatic Node.js entry point. The native binding lives in
 * `binding.js` and should not be imported directly.
 */
export function decode(input: Buffer, options?: DecodeOptions): DecodedImage {
  const telemetry = options?.telemetry === true;
  const timebase = telemetry ? unixTimeMs() : 0;
  const start = telemetry ? hrtime.bigint() : 0n;

  const decoded = decodeNative(input, options);
  const pixels = decoded.pixels;

  const wallMs = telemetry ? Number(hrtime.bigint() - start) / 1_000_000 : 0;
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
