import assert from 'node:assert/strict';
import test from 'node:test';
import { downloadBlob } from '../src/state/download';

test('blob download keeps its URL alive while the browser starts saving', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const anchor = { href: '', download: '', click: t.mock.fn(), remove: t.mock.fn() };
  const originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document');
  Object.defineProperty(globalThis, 'document', { configurable: true, value: {
    createElement: () => anchor, body: { appendChild: t.mock.fn() },
  } });
  const create = t.mock.method(URL, 'createObjectURL', () => 'blob:download-test');
  const revoke = t.mock.method(URL, 'revokeObjectURL', () => {});
  try {
    downloadBlob(new Blob(['primitive bytes']), 'example.npz');
    assert.equal(create.mock.callCount(), 1);
    assert.equal(anchor.href, 'blob:download-test');
    assert.equal(anchor.download, 'example.npz');
    assert.equal(anchor.click.mock.callCount(), 1);
    assert.equal(revoke.mock.callCount(), 0);
    t.mock.timers.tick(60_000);
    assert.equal(revoke.mock.callCount(), 1);
  } finally {
    if (originalDocument) Object.defineProperty(globalThis, 'document', originalDocument);
    else Reflect.deleteProperty(globalThis, 'document');
  }
});
