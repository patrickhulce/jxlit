const { test } = require("node:test");
const assert = require("node:assert/strict");
const { decode } = require("../index.js");

test("decode returns empty buffer", () => {
  const result = decode(Buffer.from("not-a-jxl"));
  assert.equal(result.length, 0);
});
