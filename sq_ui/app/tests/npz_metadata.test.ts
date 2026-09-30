import assert from 'node:assert/strict';
import test from 'node:test';
import JSZip from 'jszip';
import { exportNpz, type PrimitiveExport } from '../src/mesh/npzExport';
import { importNpzWithMetadata, type NpzSpaceflowMetadata } from '../src/mesh/npzImport';

const primitive: PrimitiveExport = {
  scales: [0.13, 0.11, 0.16], shapes: [0.4, 0.48],
  translation: [0, 0, 0], rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  controlLevel: 'low', tapering: [0.1, 0.2], bending: [0, 0, 0, 0, 0, 0],
};

test('NPZ preserves primitives, names, text conditions, and run settings', async () => {
  const metadata: NpzSpaceflowMetadata = {
    projectName: 'teacup', textPrompt: 'A blue teacup', outputName: 'blue_teacup',
    textureMode: 'text', globalTextureText: 'blue glazed ceramic',
    globalTextureImagePath: '', textureExperimentPrompt: 'A blue ceramic teacup.',
    primitiveNames: ['handle'], localTextureTexts: ['gold'], localTextureImagePaths: [''],
    lowTau: 3, highTau: 10, polyakTau: 0, repaintSteps: 0, textureOptimSteps: 300,
    convertYupToZup: false, lowControlBBoxMargin: 0,
  };
  const blob = await exportNpz([primitive], { metadata });
  const restored = await importNpzWithMetadata(blob, 'other', { skipEditorRescale: true });
  assert.deepEqual(restored.metadata, metadata);
  const actual = restored.primitives[0];
  assert.equal(actual.name, 'handle');
  assert.equal(actual.localTextureText, 'gold');
  for (const key of ['scales', 'shapes', 'translation', 'rotation', 'controlLevel', 'tapering', 'bending'] as const) {
    assert.deepEqual(actual[key], primitive[key]);
  }
});

test('geometry-only NPZ inputs remain compatible', async () => {
  const restored = await importNpzWithMetadata(await exportNpz([primitive]), 'legacy', {
    skipEditorRescale: true,
  });
  assert.equal(restored.metadata, null);
  assert.equal(restored.primitives.length, 1);
  assert.deepEqual(restored.primitives[0].scales, primitive.scales);
  assert.equal(restored.primitives[0].controlLevel, 'low');
});

test('empty prompts and output names survive a round trip', async () => {
  const metadata = { globalTextureText: '', globalTextureImagePath: '', outputName: '' };
  const restored = await importNpzWithMetadata(await exportNpz([primitive], { metadata }), 'empty');
  assert.deepEqual(restored.metadata, metadata);
});

test('legacy np.savez byte-string metadata restores saved prompts', async () => {
  const metadata = { textPrompt: 'a teacup', globalTextureText: 'white ceramic',
    primitiveNames: ['cup body'], localTextureTexts: ['blue ceramic'] };
  const json = new TextEncoder().encode(JSON.stringify(metadata));
  let header = `{'descr': '|S${json.length}', 'fortran_order': False, 'shape': (), }`;
  const length = Math.ceil((10 + header.length + 1) / 64) * 64 - 10;
  header = header.padEnd(length - 1, ' ') + '\n';
  const encoded = new Uint8Array(10 + length + json.length);
  encoded.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, length & 0xff, length >> 8]);
  encoded.set(new TextEncoder().encode(header), 10);
  encoded.set(json, 10 + length);
  const zip = await JSZip.loadAsync(await (await exportNpz([primitive])).arrayBuffer());
  zip.file('spaceflow_metadata.json.npy', encoded);
  const restored = await importNpzWithMetadata(await zip.generateAsync({ type: 'blob' }), 'legacy');
  assert.deepEqual(restored.metadata, metadata);
  assert.equal(restored.primitives[0].name, 'cup body');
  assert.equal(restored.primitives[0].localTextureText, 'blue ceramic');
});
