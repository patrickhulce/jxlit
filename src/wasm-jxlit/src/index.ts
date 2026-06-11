import {
  decode as decodeNative,
  DecodeOptions as DecodeOptionsNative,
} from "../pkg/jxlit_wasm_bindings.js";

import {
  rebaseTelemetry,
  unixTimeMs,
  type DecodeTelemetry,
  type Measure,
  type NativeDecodeTelemetry,
} from "./telemetry.js";

export interface DecodeOptions {
  /** Ignored on WASM; threading is not available in this build. */
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
  pixels: Float32Array;
  metadata: DecodeMetadata;
}

function metadataFromNative(metadata: {
  _jxlit: {
    version: string;
    telemetry?: NativeDecodeTelemetry;
  };
}): DecodeMetadata {
  return {
    _jxlit: {
      version: metadata._jxlit.version,
      telemetry: undefined,
    },
  };
}

function rebaseMetadata(
  metadata: {
    _jxlit: {
      version: string;
      telemetry?: NativeDecodeTelemetry;
    };
  },
  timebase: number,
  wallMs: number,
): DecodeMetadata {
  const nativeTelemetry = metadata._jxlit.telemetry;
  if (nativeTelemetry === undefined) {
    return metadataFromNative(metadata);
  }
  const telemetry = rebaseTelemetry(
    nativeTelemetry,
    timebase,
    wallMs,
    "wasm_decode",
  );
  return {
    _jxlit: {
      version: metadata._jxlit.version,
      telemetry,
    },
  };
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
  const telemetry = options?.telemetry === true;
  const nativeOptions =
    options === undefined
      ? undefined
      : new DecodeOptionsNative(
          options.threads ?? undefined,
          options.telemetry ?? false,
        );

  const timebase = telemetry ? unixTimeMs() : 0;
  const start = telemetry ? performance.now() : 0;

  const decoded = decodeNative(input, nativeOptions);
  const wallMs = telemetry ? performance.now() - start : 0;

  const height = decoded.height;
  const width = decoded.width;
  const channels = decoded.channels;
  const pixels = new Float32Array(decoded.pixels);
  const metadata = telemetry
    ? rebaseMetadata(decoded.metadata, timebase, wallMs)
    : metadataFromNative(decoded.metadata);

  decoded.free();

  return {
    height,
    width,
    channels,
    pixels,
    metadata,
  };
}
