import { test } from "node:test";
import * as assert from "node:assert/strict";
import { decode } from "../dist/index.js";

test("decode returns empty buffer", () => {
  const result = decode(Buffer.from("not-a-jxl"));
  assert.equal(result.length, 0);
});
