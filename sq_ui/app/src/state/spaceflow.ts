import { importNpzWithMetadata, type ImportedNpz, type NpzSpaceflowMetadata } from '../mesh/npzImport';
import {
  buildSpaceflowSqBundleBlobs,
  buildSpaceflowSqBundleData,
  type SpaceflowSqBundleData,
} from '../mesh/spaceflowExport';
import type { Primitive } from './store';
import {
  clampLowControlBBoxMargin,
  DEFAULT_LOW_CONTROL_BBOX_MARGIN,
} from './spaceflowConfig';
import { serviceBaseUrl } from './devServiceUrl';

const PUBLIC_DEMO = String(import.meta.env.VITE_PUBLIC_DEMO ?? '').toLowerCase() === '1'
  || String(import.meta.env.VITE_PUBLIC_DEMO ?? '').toLowerCase() === 'true';

function spaceflowBase(): string {
  if (PUBLIC_DEMO) return '';
  return serviceBaseUrl(import.meta.env.VITE_SPACEFLOW_URL, 'http://localhost:11438');
}

function resolveUrl(pathOrUrl: string): string {
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
  const base = spaceflowBase().replace(/\/$/, '');
  const path = pathOrUrl.startsWith('/') ? pathOrUrl : `/${pathOrUrl}`;
  return base ? `${base}${path}` : path;
}

export function resolveSpaceflowUrl(pathOrUrl: string): string {
  return resolveUrl(pathOrUrl);
}

export interface SpaceflowHistoryEntry {
  id: string;
  project_name: string;
  saved_at: string;
  asset_dir: string;
  manifest_path: string;
  paths: {
    all: string;
    high_control: string;
    low_control_bbox: string;
  };
  counts?: {
    all?: number;
    high?: number;
    low?: number;
  };
}

export interface SpaceflowRunConfig {
  textPrompt: string;
  appearanceMode: 'text' | 'image';
  appearanceText?: string;
  appearanceImagePath?: string;
  textureMode?: 'text' | 'image';
  globalTextureText?: string;
  globalTextureImagePath?: string;
  localTextureTexts?: string[];
  localTextureImagePaths?: string[];
  textureExperimentPrompt?: string;
  lowTau: number;
  highTau: number;
  polyakTau: number;
  repaintSteps: number;
  textureOptimSteps: number;
  outputName: string;
  convertYupToZup: boolean;
  lowControlBBoxMargin: number;
  dryRun?: boolean;
  experimentMode?: boolean;
  experimentType?: 'geometry' | 'texture' | 'full';
}

export interface SpaceflowOutputFile {
  name: string;
  path: string;
  relative_path: string;
  kind: 'mesh' | 'data' | 'image' | 'video' | 'log' | 'file' | string;
  size: number;
  mtime: number;
  url: string;
}

export interface SpaceflowRunWarning {
  kind?: string;
  severity?: 'warning' | 'error' | 'info' | string;
  message: string;
  source?: string;
  sq_indices?: number[];
  sq_numbers?: number[];
}

export interface SpaceflowRunStatus {
  run_id: string;
  status: 'running' | 'cancelling' | 'cancelled' | 'succeeded' | 'failed' | 'dry_run' | string;
  project_name?: string;
  output_dir?: string;
  log_path?: string;
  command?: string[];
  returncode?: number;
  pipeline_stage?: string;
  experiment_mode?: boolean;
  experiment_variants?: Array<{
    name: string;
    output_dir: string;
    mode?: string;
    low_tau?: number;
    high_tau?: number | null;
    polyak_tau?: number;
    n_repaint_steps?: number;
    texture_optim_steps?: number;
  }>;
  output_files?: SpaceflowOutputFile[];
  warnings?: SpaceflowRunWarning[];
}

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `SpaceFlow service returned ${res.status}`;
    try {
      const payload = await res.json() as { error?: { message?: string; log_tail?: string } };
      if (payload.error?.message) message = payload.error.message;
      if (payload.error?.log_tail) message += `\n${payload.error.log_tail.slice(-2000)}`;
    } catch {
      const text = await res.text().catch(() => '');
      if (text) message += `: ${text.slice(0, 400)}`;
    }
    throw new Error(message);
  }
  return await res.json() as T;
}

function manifestFor(
  projectName: string,
  primitives: Primitive[],
  bundle: SpaceflowSqBundleData,
  options: { lowTau: number; highTau: number; lowControlBBoxMargin: number; metadata?: NpzSpaceflowMetadata },
) {
  return {
    project_name: projectName,
    counts: bundle.counts,
    low_tau: options.lowTau,
    high_tau: options.highTau,
    bbox_margin_fraction: options.lowControlBBoxMargin,
    bbox: bundle.bbox,
    spaceflow_metadata: options.metadata,
    primitives: primitives.map((p, index) => ({
      index,
      name: p.name,
      controlLevel: p.controlLevel,
      visible: p.visible,
    })),
  };
}

async function buildBundleForm(
  projectName: string,
  primitives: Primitive[],
  options: { lowTau: number; highTau: number; lowControlBBoxMargin?: number; metadata?: NpzSpaceflowMetadata },
): Promise<{ form: FormData; bundle: SpaceflowSqBundleData }> {
  const visiblePrimitives = primitives.filter(p => p.visible);
  const lowControlBBoxMargin = clampLowControlBBoxMargin(
    options.lowControlBBoxMargin ?? DEFAULT_LOW_CONTROL_BBOX_MARGIN,
  );
  const bundle = buildSpaceflowSqBundleData(visiblePrimitives, { lowControlBBoxMargin });
  const metadata: NpzSpaceflowMetadata = {
    ...options.metadata,
    projectName,
    lowTau: options.lowTau,
    highTau: options.highTau,
    lowControlBBoxMargin,
    primitiveNames: visiblePrimitives.map(p => p.name),
    localTextureTexts: visiblePrimitives.map(p => p.localTextureText ?? ''),
    localTextureImagePaths: visiblePrimitives.map(p => p.localTextureImagePath ?? ''),
  };
  const blobs = await buildSpaceflowSqBundleBlobs(bundle, metadata);
  const form = new FormData();
  form.append('projectName', projectName);
  form.append('manifest', JSON.stringify(manifestFor(projectName, visiblePrimitives, bundle, {
    lowTau: options.lowTau,
    highTau: options.highTau,
    lowControlBBoxMargin,
    metadata,
  })));
  form.append('all', blobs.all, 'all.npz');
  form.append('high_control', blobs.highControl, 'high_control.npz');
  form.append('low_control_bbox', blobs.lowControlBbox, 'low_control_bbox.npz');
  return { form, bundle };
}

export async function saveSpaceflowAsset(options: {
  projectName: string;
  primitives: Primitive[];
  lowTau?: number;
  highTau?: number;
  lowControlBBoxMargin?: number;
  metadata?: NpzSpaceflowMetadata;
}): Promise<{ entry: SpaceflowHistoryEntry; bundle: SpaceflowSqBundleData }> {
  const { form, bundle } = await buildBundleForm(options.projectName, options.primitives, {
    lowTau: options.lowTau ?? 3,
    highTau: options.highTau ?? 10,
    lowControlBBoxMargin: options.lowControlBBoxMargin,
    metadata: options.metadata,
  });
  const res = await fetch(resolveUrl('/spaceflow/assets/save'), {
    method: 'POST',
    body: form,
  });
  const payload = await parseJson<{ status: 'ok'; entry: SpaceflowHistoryEntry }>(res);
  return { entry: payload.entry, bundle };
}

export async function fetchSpaceflowHistory(limit = 50): Promise<SpaceflowHistoryEntry[]> {
  const res = await fetch(resolveUrl(`/spaceflow/assets/history?limit=${encodeURIComponent(String(limit))}`));
  const payload = await parseJson<{ status: 'ok'; entries: SpaceflowHistoryEntry[] }>(res);
  return payload.entries;
}

export async function openSpaceflowAsset(entry: SpaceflowHistoryEntry): Promise<ImportedNpz> {
  const res = await fetch(resolveUrl(`/spaceflow/assets/open?path=${encodeURIComponent(entry.paths.all)}`));
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Could not open saved NPZ: ${res.status} ${text.slice(0, 200)}`);
  }
  const blob = await res.blob();
  return importNpzWithMetadata(blob, entry.project_name || 'spaceflow_asset', {
    basisZUpToYUp: false,
  });
}

export async function startSpaceflowRun(options: {
  projectName: string;
  primitives: Primitive[];
  runConfig: SpaceflowRunConfig;
  appearanceImageFile?: File | null;
  textureImageFile?: File | null;
  localTextureImageFiles?: Array<File | null | undefined>;
}): Promise<{ run: SpaceflowRunStatus; bundle: SpaceflowSqBundleData }> {
  const { form, bundle } = await buildBundleForm(options.projectName, options.primitives, {
    lowTau: options.runConfig.lowTau,
    highTau: options.runConfig.highTau,
    lowControlBBoxMargin: options.runConfig.lowControlBBoxMargin,
    metadata: options.runConfig,
  });
  if (bundle.counts.high === 0) {
    throw new Error('Mark at least one primitive as high control before running SpaceFlow.');
  }
  if (bundle.counts.low === 0) {
    throw new Error('Mark at least one primitive as low control before running SpaceFlow.');
  }
  form.append('runConfig', JSON.stringify(options.runConfig));
  if (options.textureImageFile) {
    form.append('texture_image', options.textureImageFile, options.textureImageFile.name);
  }
  if (options.appearanceImageFile) {
    form.append('appearance_image', options.appearanceImageFile, options.appearanceImageFile.name);
  }
  options.localTextureImageFiles?.forEach((file, index) => {
    if (file) form.append(`local_texture_image_${index}`, file, file.name);
  });
  const res = await fetch(resolveUrl('/spaceflow/runs/start'), {
    method: 'POST',
    body: form,
  });
  const payload = await parseJson<{ status: 'ok'; run_id: string; run: SpaceflowRunStatus }>(res);
  return { run: payload.run, bundle };
}

export async function getSpaceflowRunStatus(runId: string): Promise<{
  run: SpaceflowRunStatus;
  logTail: string;
}> {
  const res = await fetch(resolveUrl(`/spaceflow/runs/status?run_id=${encodeURIComponent(runId)}`));
  const payload = await parseJson<{ status: 'ok'; run: SpaceflowRunStatus; log_tail: string }>(res);
  return { run: payload.run, logTail: payload.log_tail };
}

export async function stopSpaceflowRun(runId: string): Promise<SpaceflowRunStatus> {
  const res = await fetch(resolveUrl('/spaceflow/runs/stop'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId }),
  });
  const payload = await parseJson<{ status: 'ok'; run: SpaceflowRunStatus }>(res);
  return payload.run;
}
