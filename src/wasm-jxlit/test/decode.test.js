import { test } from 'node:test'
import assert from 'node:assert/strict'
import { decode } from '../index.js'

test('decode returns empty buffer', () => {
  const result = decode(new Uint8Array([1, 2, 3]))
  assert.equal(result.length, 0)
})
