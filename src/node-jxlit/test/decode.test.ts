import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import * as assert from "node:assert/strict";
import { PNG } from "pngjs";

import { decode } from "../dist/index.js";

const ASSETS_DIR = join(process.cwd(), "../../assets");
const JXL_FIXTURE = join(ASSETS_DIR, "colors_e1_d0p5_fd4.jxl");
const PNG_FIXTURE = join(ASSETS_DIR, "colors.png");
const MAE_TOLERANCE = 0.02;

function loadPngRgbF32(path: string): {
  height: number;
  width: number;
  channels: number;
  pixels: Float32Array;
} {
  const png = PNG.sync.read(readFileSync(path));
  const channels = 3;
  const pixels = new Float32Array(png.width * png.height * channels);

  for (let y = 0; y < png.height; y += 1) {
    for (let x = 0; x < png.width; x += 1) {
      const src = (png.width * y + x) * 4;
      const dst = (png.width * y + x) * channels;
      pixels[dst] = png.data[src] / 255;
      pixels[dst + 1] = png.data[src + 1] / 255;
      pixels[dst + 2] = png.data[src + 2] / 255;
    }
  }

  return {
    height: png.height,
    width: png.width,
    channels,
    pixels,
  };
}

function meanAbsError(a: Float32Array, b: Float32Array): number {
  assert.equal(a.length, b.length);
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    total += Math.abs(a[i] - b[i]);
  }
  return total / a.length;
}

test("decode colors fixture is close to png", () => {
  const decoded = decode(readFileSync(JXL_FIXTURE));
  const expected = loadPngRgbF32(PNG_FIXTURE);

  assert.equal(decoded.height, expected.height);
  assert.equal(decoded.width, expected.width);
  assert.equal(decoded.channels, expected.channels);
  assert.equal(decoded.pixels.length, expected.pixels.length);

  const mae = meanAbsError(decoded.pixels, expected.pixels);
  assert.ok(
    mae < MAE_TOLERANCE,
    `mean absolute error ${mae} exceeds ${MAE_TOLERANCE}`,
  );
});

test("decode metadata includes version", () => {
  const decoded = decode(readFileSync(JXL_FIXTURE));
  assert.ok(decoded.metadata._jxlit.version);
  assert.equal(decoded.metadata._jxlit.telemetry, undefined);
});

test("decode telemetry collects measures", () => {
  const decoded = decode(readFileSync(JXL_FIXTURE), { telemetry: true });
  const telemetry = decoded.metadata._jxlit.telemetry;
  assert.ok(telemetry);
  assert.ok(telemetry.timebase > 0);
  assert.ok(telemetry.totalMs > 0);
  assert.ok(telemetry.measures.length > 0);
  const names = new Set(telemetry.measures.map((measure) => measure.name));
  assert.ok(names.has("node_decode"));
  assert.ok(names.has("decode"));
  assert.ok(names.has("parse"));
  assert.ok(names.has("render"));

  const outer = telemetry.measures.find(
    (measure) => measure.name === "node_decode",
  );
  assert.ok(outer);
  assert.equal(outer.startMs, 0);
  assert.equal(outer.durationMs, telemetry.totalMs);

  const innerDecode = telemetry.measures.find(
    (measure) => measure.name === "decode",
  );
  assert.ok(innerDecode);
  assert.ok(innerDecode.startMs > 0);
});
