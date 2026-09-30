/**
 * Load superquadric .npz (ZIP of .npy) produced by the SQ Editor export or pipeline.
 */
import JSZip from 'jszip';
import type { Primitive } from '../state/store';
import { matrixToEuler } from '../state/rotation';
import type { PrimitiveExport } from './npzExport';

/**
 * Maps a Z-up world frame to the editor's Y-up frame (Three.js):
 * x' = x, y' = z, z' = -y. Use when pipeline assets were authored with vertical Z.
 * World pose: p' = S @ p, so R' = S @ R, t' = S @ t for column vectors.
 */
const S_ZUP_TO_YUP: number[][] = [
  [1, 0, 0],
  [0, 0, 1],
  [0, -1, 0],
];

function matMul3x3(A: number[][], B: number[][]): number[][] {
  const C: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    }
  }
  return C;
}

function matVec3(R: number[][], v: [number, number, number]): [number, number, number] {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

export function applyWorldBasisZUpToYUp(
  R: number[][],
  t: [number, number, number],
): { rotation: number[][]; translation: [number, number, number] } {
  return {
    rotation: matMul3x3(S_ZUP_TO_YUP, R),
    translation: matVec3(S_ZUP_TO_YUP, t),
  };
}

interface ParsedNpy {
  shape: number[];
  /** C-order row-major values as float64 */
  values: Float64Array;
}

function squeezeSingletonAxis(
  parsed: ParsedNpy,
  trailingShape: number[],
  label: string,
): ParsedNpy {
  const shape = [...parsed.shape];
  while (shape.length > trailingShape.length + 1 && shape[1] === 1) {
    shape.splice(1, 1);
  }
  if (shape.length === trailingShape.length + 2 && shape[1] === 1) {
    shape.splice(1, 1);
  }
  const expected = [shape[0], ...trailingShape];
  if (shape.length !== expected.length || !expected.every((v, i) => shape[i] === v)) {
    throw new Error(`${label} expected shape (N,${trailingShape.join(',')}), got (${parsed.shape.join(',')})`);
  }
  return { ...parsed, shape };
}

function parseNpyHeader(headerText: string): { descr: string; shape: number[]; fortran: boolean } {
  const descrM = /'descr':\s*'([^']+)'/.exec(headerText);
  if (!descrM) throw new Error('NPY header missing descr');
  const descr = descrM[1];
  const shapeM = /'shape':\s*\(([^)]*)\)/.exec(headerText);
  if (!shapeM) throw new Error('NPY header missing shape');
  const raw = shapeM[1].trim();
  const shape =
    raw === ''
      ? []
      : raw.split(',')
          .map(s => s.trim())
          .filter(s => s.length > 0)
          .map(s => {
            const n = Number(s);
            if (!Number.isFinite(n)) throw new Error(`Invalid shape segment: ${s}`);
            return n;
          });
  const fortran = /'fortran_order':\s*True/.test(headerText);
  return { descr, shape, fortran };
}

function parseNpyScalarBool(buffer: ArrayBuffer): boolean | null {
  const u8 = new Uint8Array(buffer);
  if (u8.length < 10 || u8[0] !== 0x93 || u8[1] !== 0x4e || u8[2] !== 0x55 || u8[3] !== 0x4d || u8[4] !== 0x50 || u8[5] !== 0x59) {
    return null;
  }
  if (u8[6] !== 1 || u8[7] !== 0) return null;
  const headerLen = u8[8] | (u8[9] << 8);
  const headerText = new TextDecoder('latin1').decode(u8.slice(10, 10 + headerLen));
  const { descr, shape, fortran } = parseNpyHeader(headerText);
  if (fortran) return null;
  const n = shape.reduce((acc, value) => acc * value, 1);
  if (n !== 1) return null;
  const offset = 10 + headerLen;
  if (descr === '|b1' || descr === '?' || descr === '<b1') {
    return u8[offset] !== 0;
  }
  if (descr === '<f8' || descr === '<f4') {
    return readValues(u8, offset, descr, 1)[0] !== 0;
  }
  return null;
}

function readValues(
  data: Uint8Array,
  offset: number,
  descr: string,
  n: number,
): Float64Array {
  const out = new Float64Array(n);
  if (descr === '<f8') {
    const view = new DataView(data.buffer, data.byteOffset + offset, n * 8);
    for (let i = 0; i < n; i++) out[i] = view.getFloat64(i * 8, true);
    return out;
  }
  if (descr === '<f4') {
    const view = new DataView(data.buffer, data.byteOffset + offset, n * 4);
    for (let i = 0; i < n; i++) out[i] = view.getFloat32(i * 4, true);
    return out;
  }
  throw new Error(`Unsupported NPY descr ${descr} (need <f8 or <f4)`);
}

export function parseNpyBuffer(buffer: ArrayBuffer): ParsedNpy {
  const u8 = new Uint8Array(buffer);
  if (u8.length < 10 || u8[0] !== 0x93 || u8[1] !== 0x4e || u8[2] !== 0x55 || u8[3] !== 0x4d || u8[4] !== 0x50 || u8[5] !== 0x59) {
    throw new Error('Not a NumPy .npy file');
  }
  if (u8[6] !== 1 || u8[7] !== 0) {
    throw new Error('Only NumPy format v1.0 is supported');
  }
  const headerLen = u8[8] | (u8[9] << 8);
  const headerText = new TextDecoder('latin1').decode(u8.slice(10, 10 + headerLen));
  const { descr, shape, fortran } = parseNpyHeader(headerText);
  if (fortran) {
    throw new Error('Fortran-ordered arrays are not supported');
  }
  let numEl = 1;
  for (const s of shape) numEl *= s;
  const offset = 10 + headerLen;
  const values = readValues(u8, offset, descr, numEl);
  return { shape, values };
}

function reshape2(values: Float64Array, rows: number, cols: number): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) row.push(values[i * cols + j]);
    out.push(row);
  }
  return out;
}

function reshape3(values: Float64Array, a: number, b: number, c: number): number[][][] {
  const out: number[][][] = [];
  let idx = 0;
  for (let i = 0; i < a; i++) {
    const plane: number[][] = [];
    for (let j = 0; j < b; j++) {
      const row: number[] = [];
      for (let k = 0; k < c; k++) row.push(values[idx++]);
      plane.push(row);
    }
    out.push(plane);
  }
  return out;
}

/** Parse arrays from an exported / pipeline-compatible .npz into primitive data. */
export function npzArraysToExports(
  scales: ParsedNpy,
  shapes: ParsedNpy,
  translations: ParsedNpy,
  rotations: ParsedNpy,
  tapering?: ParsedNpy | null,
  bending?: ParsedNpy | null,
): PrimitiveExport[] {
  const scalesSq = squeezeSingletonAxis(scales, [3], 'scales.npy');
  const shapesSq = squeezeSingletonAxis(shapes, [2], 'shapes.npy');
  const translationsSq = squeezeSingletonAxis(translations, [3], 'translations.npy');
  const rotationsSq = squeezeSingletonAxis(rotations, [3, 3], 'rotations.npy');

  const N = scalesSq.shape[0];
  if (shapesSq.shape[0] !== N || translationsSq.shape[0] !== N || rotationsSq.shape[0] !== N) {
    throw new Error(`Array length mismatch: scales ${N}, shapes ${shapesSq.shape[0]}, translations ${translationsSq.shape[0]}, rotations ${rotationsSq.shape[0]}`);
  }

  const scalesM = reshape2(scalesSq.values, N, 3);
  const shapesM = reshape2(shapesSq.values, N, 2);
  const transM = reshape2(translationsSq.values, N, 3);
  const rotM = reshape3(rotationsSq.values, N, 3, 3);

  let taperM: number[][] | null = null;
  if (tapering) {
    const tSq = squeezeSingletonAxis(tapering, [2], 'tapering.npy');
    if (tSq.shape[0] !== N) {
      throw new Error(`tapering.npy length ${tSq.shape[0]} != scales ${N}`);
    }
    taperM = reshape2(tSq.values, N, 2);
  }

  let bendM: number[][] | null = null;
  if (bending) {
    const bSq = squeezeSingletonAxis(bending, [6], 'bending.npy');
    if (bSq.shape[0] !== N) {
      throw new Error(`bending.npy length ${bSq.shape[0]} != scales ${N}`);
    }
    bendM = reshape2(bSq.values, N, 6);
  }

  const out: PrimitiveExport[] = [];
  for (let i = 0; i < N; i++) {
    const row: PrimitiveExport = {
      scales: [scalesM[i][0], scalesM[i][1], scalesM[i][2]],
      shapes: [shapesM[i][0], shapesM[i][1]],
      translation: [transM[i][0], transM[i][1], transM[i][2]],
      rotation: rotM[i],
    };
    if (taperM) {
      row.tapering = [taperM[i][0], taperM[i][1]];
    }
    if (bendM) {
      row.bending = [
        bendM[i][0], bendM[i][1], bendM[i][2], bendM[i][3], bendM[i][4], bendM[i][5],
      ];
    }
    out.push(row);
  }
  return out;
}

/** Median half-axis after auto-rescale (middle of 0–5 scale sliders). */
export const EDITOR_TYPICAL_HALF_AXIS = 2.5;
/** If every half-axis is tiny, expand the preset into the editor's slider range. */
export const AUTO_RESCALE_SCENE_MAX_HALF_AXIS = 0.04;

function median(nums: number[]): number {
  if (nums.length === 0) return 0;
  const s = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid]! : ((s[mid - 1]! + s[mid]!) / 2);
}

/**
 * Uniformly scale scales and translations so typical half-axes land near EDITOR_TYPICAL_HALF_AXIS.
 * Geometry stays the same up to one global scale; rotations unchanged.
 */
export function maybeRescalePrimitivesForEditor(primitives: Primitive[]): Primitive[] {
  if (primitives.length === 0) return primitives;
  const halves = primitives.flatMap(p => [...p.scales]);
  const sceneMax = Math.max(...halves);
  if (sceneMax >= AUTO_RESCALE_SCENE_MAX_HALF_AXIS) return primitives;
  const med = median(halves);
  if (!Number.isFinite(med) || med <= 0) return primitives;
  const k = EDITOR_TYPICAL_HALF_AXIS / med;
  if (!Number.isFinite(k) || k <= 0 || k > 1e6) return primitives;
  return primitives.map(p => ({
    ...p,
    scales: [p.scales[0] * k, p.scales[1] * k, p.scales[2] * k] as [number, number, number],
    translation: [p.translation[0] * k, p.translation[1] * k, p.translation[2] * k] as [
      number,
      number,
      number,
    ],
    ...(p.bending !== undefined
      ? {
          bending: [
            p.bending[0] / k,
            p.bending[1],
            p.bending[2] / k,
            p.bending[3],
            p.bending[4] / k,
            p.bending[5],
          ] as [number, number, number, number, number, number],
        }
      : {}),
  }));
}

export interface ImportNpzOptions {
  /** If true, apply S @ R and S @ t so Z-up pipeline assets sit upright in Y-up Three.js. */
  basisZUpToYUp?: boolean | 'auto';
  /** If true, skip expanding tiny normalized fits for slider range (default false). */
  skipEditorRescale?: boolean;
}

export interface NpzSpaceflowMetadata {
  projectName?: string;
  textPrompt?: string;
  outputName?: string;
  textureMode?: 'text' | 'image';
  globalTextureText?: string;
  globalTextureImagePath?: string;
  textureExperimentPrompt?: string;
  primitiveNames?: string[];
  localTextureTexts?: string[];
  localTextureImagePaths?: string[];
  lowTau?: number;
  highTau?: number;
  polyakTau?: number;
  repaintSteps?: number;
  textureOptimSteps?: number;
  convertYupToZup?: boolean;
  lowControlBBoxMargin?: number;
}

export interface ImportedNpz {
  primitives: Primitive[];
  metadata: NpzSpaceflowMetadata | null;
}

function findZipEntry(zip: JSZip, basename: string): JSZip.JSZipObject | null {
  const direct = zip.file(basename);
  if (direct && !direct.dir) return direct;
  for (const k of Object.keys(zip.files)) {
    const entry = zip.files[k];
    if (!entry || entry.dir) continue;
    if (k === basename || k.endsWith(`/${basename}`)) return entry;
  }
  return null;
}

function findZipNpy(zip: JSZip, basename: string): JSZip.JSZipObject | null {
  return findZipEntry(zip, basename);
}

function parsedVector(parsed: ParsedNpy, expectedLength: number, label: string): number[] {
  const shape = parsed.shape;
  const isVector = shape.length === 1 && shape[0] === expectedLength;
  const isColumn = shape.length === 2 && shape[0] === expectedLength && shape[1] === 1;
  if (!isVector && !isColumn) {
    throw new Error(`${label} expected shape (${expectedLength}) or (${expectedLength},1), got (${shape.join(',')})`);
  }
  return Array.from(parsed.values);
}

function filterByConfidence(exports: PrimitiveExport[], confidence: ParsedNpy, threshold = 0.5): PrimitiveExport[] {
  const scores = parsedVector(confidence, exports.length, 'confidence/exist');
  let keep = scores
    .map((score, index) => ({ score, index }))
    .filter(item => item.score > threshold);
  if (keep.length === 0 && scores.length > 0) {
    let bestIndex = 0;
    for (let i = 1; i < scores.length; i++) {
      if (scores[i]! > scores[bestIndex]!) bestIndex = i;
    }
    keep = [{ score: scores[bestIndex]!, index: bestIndex }];
  }
  keep.sort((a, b) => b.score - a.score);
  return keep.map(item => exports[item.index]!).filter(Boolean);
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' ? value.trim() : undefined;
}

function firstString(...values: unknown[]): string | undefined {
  for (const value of values) {
    const str = stringValue(value);
    if (str) return str;
  }
  return undefined;
}

function stringArrayValue(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return value.map(item => (typeof item === 'string' ? item : ''));
}

function firstStringArray(...values: unknown[]): string[] | undefined {
  for (const value of values) {
    const arr = stringArrayValue(value);
    if (arr) return arr;
  }
  return undefined;
}

function normalizeTextureMode(value: unknown): 'text' | 'image' | undefined {
  const raw = stringValue(value)?.toLowerCase();
  return raw === 'text' || raw === 'image' ? raw : undefined;
}

function normalizeSpaceflowMetadata(raw: unknown): NpzSpaceflowMetadata | null {
  const root = objectValue(raw);
  if (!root) return null;
  const texture = objectValue(root.texture_guidance) ?? objectValue(root.textureGuidance);
  const metadata: NpzSpaceflowMetadata = {};

  const projectName = firstString(root.projectName, root.project_name);
  if (projectName) metadata.projectName = projectName;
  const textPrompt = firstString(root.textPrompt, root.text_prompt, root.prompt);
  if (textPrompt) metadata.textPrompt = textPrompt;
  const outputName = firstString(root.outputName, root.output_name);
  if (typeof root.outputName === 'string') metadata.outputName = root.outputName;
  else if (outputName) metadata.outputName = outputName;
  const textureExperimentPrompt = firstString(root.textureExperimentPrompt, root.texture_experiment_prompt);
  if (textureExperimentPrompt) metadata.textureExperimentPrompt = textureExperimentPrompt;

  const textureMode = normalizeTextureMode(root.textureMode)
    ?? normalizeTextureMode(root.appearanceMode)
    ?? normalizeTextureMode(texture?.mode);
  if (textureMode) metadata.textureMode = textureMode;
  const globalTextureText = firstString(
    root.globalTextureText,
    root.appearanceText,
    root.global_text,
    texture?.global_text,
  );
  if (typeof root.globalTextureText === 'string') metadata.globalTextureText = root.globalTextureText;
  else if (globalTextureText) metadata.globalTextureText = globalTextureText;
  const globalTextureImagePath = firstString(
    root.globalTextureImagePath,
    root.appearanceImagePath,
    root.global_image_path,
    texture?.global_image_path,
  );
  if (typeof root.globalTextureImagePath === 'string') metadata.globalTextureImagePath = root.globalTextureImagePath;
  else if (globalTextureImagePath) metadata.globalTextureImagePath = globalTextureImagePath;

  const primitiveNames = firstStringArray(root.primitiveNames, root.primitive_names);
  if (primitiveNames) metadata.primitiveNames = primitiveNames;
  const localTextureTexts = firstStringArray(
    root.localTextureTexts,
    root.local_text_prompts,
    texture?.local_text_prompts,
  );
  if (localTextureTexts) metadata.localTextureTexts = localTextureTexts;
  const localTextureImagePaths = firstStringArray(
    root.localTextureImagePaths,
    root.local_image_paths,
    texture?.local_image_paths,
  );
  if (localTextureImagePaths) metadata.localTextureImagePaths = localTextureImagePaths;

  for (const key of ['lowTau', 'highTau', 'polyakTau', 'repaintSteps',
    'textureOptimSteps', 'lowControlBBoxMargin'] as const) {
    const value = root[key];
    if (typeof value === 'number' && Number.isFinite(value)) metadata[key] = value;
  }
  if (typeof root.convertYupToZup === 'boolean') {
    metadata.convertYupToZup = root.convertYupToZup;
  }

  return Object.keys(metadata).length > 0 ? metadata : null;
}

async function readSpaceflowMetadata(zip: JSZip): Promise<NpzSpaceflowMetadata | null> {
  const file = findZipEntry(zip, 'spaceflow_metadata.json') ?? findZipEntry(zip, 'spaceflow/metadata.json');
  if (!file) return null;
  const text = await file.async('string');
  return normalizeSpaceflowMetadata(JSON.parse(text));
}

function withSpaceflowMetadata(primitives: Primitive[], metadata: NpzSpaceflowMetadata | null): Primitive[] {
  if (!metadata) return primitives;
  const names = metadata.primitiveNames ?? [];
  const localTexts = metadata.localTextureTexts ?? [];
  const localImagePaths = metadata.localTextureImagePaths ?? [];
  return primitives.map((primitive, index) => {
    const name = names[index]?.trim();
    const localTextureText = localTexts[index]?.trim();
    const localTextureImagePath = localImagePaths[index]?.trim();
    return {
      ...primitive,
      ...(name ? { name } : {}),
      ...(localTextureText ? { localTextureText } : {}),
      ...(localTextureImagePath ? { localTextureImagePath } : {}),
    };
  });
}

export async function importNpzWithMetadata(
  blob: Blob,
  namePrefix = 'npz',
  options?: ImportNpzOptions,
): Promise<ImportedNpz> {
  const zip = await JSZip.loadAsync(await blob.arrayBuffer());
  const readNpy = async (names: string | string[]): Promise<ParsedNpy> => {
    const candidates = Array.isArray(names) ? names : [names];
    const f = candidates.map(name => findZipNpy(zip, name)).find((entry): entry is JSZip.JSZipObject => !!entry);
    if (!f) throw new Error(`Missing ${candidates[0]} in archive`);
    const buf = await f.async('arraybuffer');
    return parseNpyBuffer(buf);
  };

  const hasLegacyArrayNames = !!findZipNpy(zip, 'scale.npy') && !!findZipNpy(zip, 'shape.npy')
    && !!findZipNpy(zip, 'trans.npy') && !!findZipNpy(zip, 'rotate.npy');
  const [scales, shapes, translations, rotations] = await Promise.all([
    readNpy(['scales.npy', 'scale.npy']),
    readNpy(['shapes.npy', 'shape.npy']),
    readNpy(['translations.npy', 'trans.npy']),
    readNpy(['rotations.npy', 'rotate.npy']),
  ]);

  const taperFile = findZipNpy(zip, 'tapering.npy');
  const bendFile = findZipNpy(zip, 'bending.npy');
  const controlLevelsFile = findZipNpy(zip, 'control_levels.npy');
  const confidenceFile = findZipNpy(zip, 'confidence.npy') ?? findZipNpy(zip, 'exist.npy');
  const zUpFile = findZipNpy(zip, 'z_up.npy');
  const tapering = taperFile ? await taperFile.async('arraybuffer').then(parseNpyBuffer) : null;
  const bending = bendFile ? await bendFile.async('arraybuffer').then(parseNpyBuffer) : null;
  const controlLevels = controlLevelsFile ? await controlLevelsFile.async('arraybuffer').then(parseNpyBuffer) : null;
  const confidence = confidenceFile ? await confidenceFile.async('arraybuffer').then(parseNpyBuffer) : null;
  const zUp = zUpFile ? await zUpFile.async('arraybuffer').then(parseNpyScalarBool).catch(() => null) : null;
  const metadata = await readSpaceflowMetadata(zip);

  let exports = npzArraysToExports(scales, shapes, translations, rotations, tapering, bending);
  if (controlLevels) {
    const levels = parsedVector(controlLevels, exports.length, 'control_levels');
    exports = exports.map((e, i) => ({
      ...e,
      controlLevel: levels[i]! <= 0.5 ? 'low' : 'high',
    }));
  }
  if (hasLegacyArrayNames && confidence) {
    exports = filterByConfidence(exports, confidence);
  }
  const basisZUpToYUp = options?.basisZUpToYUp === true || (options?.basisZUpToYUp === 'auto' && zUp === true);
  if (basisZUpToYUp) {
    exports = exports.map(e => {
      const { rotation, translation } = applyWorldBasisZUpToYUp(e.rotation, e.translation);
      return { ...e, rotation, translation };
    });
  }

  const t = Date.now();
  let prims: Primitive[] = exports.map((e, i): Primitive => {
    const euler = matrixToEuler(e.rotation);
    const tapering: [number, number] | undefined = e.tapering !== undefined
      ? ([...e.tapering] as [number, number])
      : undefined;
    const bending: [number, number, number, number, number, number] | undefined = e.bending !== undefined
      ? ([...e.bending] as [number, number, number, number, number, number])
      : undefined;
    return {
      id: `npz_${i}_${t}`,
      name: `${namePrefix}_${i}`,
      visible: true,
      controlLevel: e.controlLevel ?? 'high',
      scales: e.scales,
      shapes: e.shapes,
      translation: e.translation,
      rotation: e.rotation,
      eulerDeg: euler,
      ...(tapering !== undefined ? { tapering } : {}),
      ...(bending !== undefined ? { bending } : {}),
    };
  });
  if (!options?.skipEditorRescale) {
    prims = maybeRescalePrimitivesForEditor(prims);
  }
  return {
    primitives: withSpaceflowMetadata(prims, metadata),
    metadata,
  };
}

export async function importNpzToPrimitives(
  blob: Blob,
  namePrefix = 'npz',
  options?: ImportNpzOptions,
): Promise<Primitive[]> {
  return (await importNpzWithMetadata(blob, namePrefix, options)).primitives;
}
