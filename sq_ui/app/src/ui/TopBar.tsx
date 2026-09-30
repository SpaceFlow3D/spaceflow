import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { exportNpz, type PrimitiveExport } from '../mesh/npzExport';
import { importNpzToPrimitives, importNpzWithMetadata, maybeRescalePrimitivesForEditor } from '../mesh/npzImport';
import type { NpzSpaceflowMetadata } from '../mesh/npzImport';
import { primitiveToExport } from '../mesh/spaceflowExport';
import { eulerToMatrix, isOrthogonal, matrixToEuler } from '../state/rotation';
import {
  fetchSpaceflowHistory,
  getSpaceflowRunStatus,
  openSpaceflowAsset,
  resolveSpaceflowUrl,
  saveSpaceflowAsset,
  startSpaceflowRun,
  stopSpaceflowRun,
  type SpaceflowHistoryEntry,
  type SpaceflowOutputFile,
  type SpaceflowRunStatus,
} from '../state/spaceflow';
import { npzEditorUrl } from '../state/npzUrl';
import { useStore, type Primitive } from '../state/store';
import { useTextureUploadStore } from '../state/textureUploads';
import { useSpaceflowUiStore } from '../state/spaceflowUi';
import { captureSuperquadricRenderBlob } from '../state/viewportCapture';
import {
  AlertTriangleIcon,
  ChairIcon,
  CircleIcon,
  DownloadIcon,
  MoonIcon,
  RedoIcon,
  RotateIcon,
  SparklesIcon,
  SunIcon,
  TableIcon,
  UndoIcon,
  UploadIcon,
  XIcon,
} from './icons';

type ThemeMode = 'dark' | 'light';
type SpaceflowExperimentType = 'geometry' | 'texture' | 'full';

interface DownloadableGlbMesh {
  url: string;
  filename: string;
  label: string;
}

interface DemoPreset {
  id: string;
  label: string;
  prompt: string;
  texturePrompt: string;
  outputName: string;
  npzUrl: string;
  imageUrl: string;
}

interface TopBarProps {
  themeMode: ThemeMode;
  onThemeModeChange: (mode: ThemeMode) => void;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function writeClipboardText(text: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.left = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand('copy');
  document.body.removeChild(textarea);
  if (!copied) throw new Error('Clipboard is unavailable in this browser context.');
}

function safeName(name: string, fallback: string) {
  return name.replace(/[^a-zA-Z0-9_-]/g, '_') || fallback;
}

function formatFileSize(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '';
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  if (mb < 1024) return `${mb.toFixed(1)} MB`;
  return `${(mb / 1024).toFixed(1)} GB`;
}

function outputNameFromPrompt(prompt: string) {
  return prompt
    .trim()
    .replace(/[^a-zA-Z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'spaceflow_run';
}

function textureExperimentPromptTemplate(
  shapePrompt: string,
  globalTextureText: string,
  primitives: Primitive[],
) {
  const shape = shapePrompt.trim() || 'a 3D asset';
  const globalTexture = globalTextureText.trim() || shape;
  const parts = [
    `${shape}.`,
    `Overall appearance and texture: ${globalTexture}.`,
  ];
  const localParts = primitives
    .map((primitive, index) => {
      const prompt = (primitive.localTextureText ?? '').trim();
      if (!prompt) return null;
      const name = primitive.name.trim() || `spaceflow_${index}`;
      return `part ${index + 1} (${name}): ${prompt}`;
    })
    .filter((value): value is string => Boolean(value));
  if (localParts.length > 0) {
    parts.push(`Local texture overrides: ${localParts.join('; ')}.`);
    parts.push('All unspecified parts should use the overall appearance and texture.');
  } else {
    parts.push('Apply the overall appearance and texture consistently to every part.');
  }
  return parts.join(' ');
}

function outputFileLabel(file: SpaceflowOutputFile) {
  const basename = file.relative_path.toLowerCase().split('/').pop() ?? file.relative_path.toLowerCase();
  if (
    file.relative_path === 'input_superquadrics_colored.glb' ||
    file.relative_path.endsWith('/input_superquadrics_colored.glb')
  ) {
    return 'Colored input superquadrics';
  }
  switch (basename) {
    case 'out_sim.glb':
      return 'Textured refined mesh';
    case 'out_sim_geometry.glb':
      return 'White refined geometry';
    case 'out_gaussian_sim.mp4':
      return 'Refined Gaussian video';
    case 'sample.glb':
      return 'Structure mesh';
    case 'struct_mesh_zup.glb':
      return 'Structure mesh Z-up';
    case 'struct_mesh.glb':
      return 'Structure mesh Y-up';
    case 'spatial_control_mesh.ply':
      return 'All SQ control mesh';
    case 'high_control_spatial_control_mesh.ply':
      return 'High-control mesh';
    case 'low_control_superquadric_mask.ply':
      return 'Low-control mask';
    case 'struct_voxels.ply':
      return 'Structure voxels';
    case '000.png':
      return 'Structure preview';
    case 'denoising_evolution.mp4':
      return 'Denoising video';
    default:
      return file.relative_path;
  }
}

function isGlbPath(value: string | undefined) {
  return Boolean(value?.toLowerCase().split(/[?#]/, 1)[0]?.endsWith('.glb'));
}

function filenameBaseFromPath(value: string | undefined, fallback: string) {
  const clean = value?.split(/[?#]/, 1)[0] ?? '';
  const basename = clean.split(/[\\/]/).pop() ?? '';
  return basename.replace(/\.glb$/i, '') || fallback;
}

function downloadableGlbFromOutput(file: SpaceflowOutputFile, runId?: string): DownloadableGlbMesh {
  const relBase = filenameBaseFromPath(file.relative_path, file.name.replace(/\.glb$/i, '') || 'mesh');
  const prefix = runId ? `${runId}_` : '';
  return {
    url: resolveSpaceflowUrl(file.url),
    filename: `${safeName(`${prefix}${relBase}`, 'spaceflow_mesh')}.glb`,
    label: outputFileLabel(file),
  };
}

const SPACEFLOW_INSPECTION_MESH_PRIORITY = [
  'out_sim.glb',
  'out_sim_geometry.glb',
  'struct_mesh_zup.glb',
  'sample.glb',
  'struct_mesh.glb',
];

function pickSpaceflowInspectionMesh(files: SpaceflowOutputFile[]) {
  const byPath = new Map(files.map(file => [file.relative_path.toLowerCase(), file]));
  for (const relativePath of SPACEFLOW_INSPECTION_MESH_PRIORITY) {
    const file = byPath.get(relativePath);
    if (file) return file;
  }
  for (const filename of SPACEFLOW_INSPECTION_MESH_PRIORITY) {
    const file = files.find(candidate => candidate.relative_path.toLowerCase().split('/').pop() === filename);
    if (file) return file;
  }
  const generatedMesh = files.find(file => (
    file.kind === 'mesh' &&
    /\.(glb|gltf)$/i.test(file.relative_path) &&
    file.relative_path.toLowerCase().split('/').pop() !== 'input_superquadrics_colored.glb'
  ));
  return generatedMesh ?? files.find(file => file.kind === 'mesh' && /\.(glb|gltf)$/i.test(file.relative_path));
}

let nextPresetId = 0;
const PUBLIC_DEMO = String(import.meta.env.VITE_PUBLIC_DEMO ?? '').toLowerCase() === '1'
  || String(import.meta.env.VITE_PUBLIC_DEMO ?? '').toLowerCase() === 'true';

const DEMO_PRESETS: DemoPreset[] = [
  {
    id: 'chair',
    label: 'Chair',
    prompt: 'A comfy chair',
    texturePrompt: 'white comfy chair',
    outputName: 'chair',
    npzUrl: '/demo-presets/chair.npz',
    imageUrl: '/demo-presets/chair.png?v=sq-render-2',
  },
  {
    id: 'car',
    label: 'Car',
    prompt: 'A car',
    texturePrompt: 'A car',
    outputName: 'car',
    npzUrl: '/demo-presets/car.npz',
    imageUrl: '/demo-presets/car.png?v=sq-render-2',
  },
  {
    id: 'bench',
    label: 'Bench',
    prompt: 'An elegant legged bench',
    texturePrompt: 'An elegant legged bench',
    outputName: 'bench',
    npzUrl: '/demo-presets/bench.npz',
    imageUrl: '/demo-presets/bench.png?v=sq-render-2',
  },
  {
    id: 'trophy',
    label: 'Trophy',
    prompt: 'A gold trophy',
    texturePrompt: 'A gold trophy',
    outputName: 'trophy',
    npzUrl: '/demo-presets/trophy.npz',
    imageUrl: '/demo-presets/trophy.png?v=sq-render-2',
  },
];

export default function TopBar({ themeMode, onThemeModeChange }: TopBarProps) {
  const primitives = useStore(s => s.primitives);
  const loadPreset = useStore(s => s.loadPreset);
  const meshInspection = useStore(s => s.meshInspection);
  const setMeshInspection = useStore(s => s.setMeshInspection);
  const rotateAllWorld = useStore(s => s.rotateAllWorld);
  const undo = useStore(s => s.undo);
  const redo = useStore(s => s.redo);
  const undoStack = useStore(s => s.undoStack);
  const redoStack = useStore(s => s.redoStack);
  const lowControlBBoxMargin = useStore(s => s.lowControlBBoxMargin);
  const setLowControlBBoxMargin = useStore(s => s.setLowControlBBoxMargin);
  const spaceflowLocalTextureImageFiles = useTextureUploadStore(s => s.localTextureImageFiles);

  const [toast, setToast] = useState<string | null>(null);
  const [showImport, setShowImport] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const [showRotateAll, setShowRotateAll] = useState(false);
  const [showSpaceflow, setShowSpaceflow] = useState(false);
  const [spaceflowSaving, setSpaceflowSaving] = useState(false);
  const [spaceflowRunning, setSpaceflowRunning] = useState(false);
  const [spaceflowHistory, setSpaceflowHistory] = useState<SpaceflowHistoryEntry[]>([]);
  const [spaceflowHistoryLoading, setSpaceflowHistoryLoading] = useState(false);
  const [showSpaceflowHistoryPanel, setShowSpaceflowHistoryPanel] = useState(false);
  const [spaceflowRun, setSpaceflowRun] = useState<SpaceflowRunStatus | null>(null);
  const [spaceflowLogTail, setSpaceflowLogTail] = useState('');
  const [spaceflowTextPrompt, setSpaceflowTextPrompt] = useState('A chair');
  const spaceflowTextureMode = useSpaceflowUiStore(s => s.textureMode);
  const setSpaceflowTextureMode = useSpaceflowUiStore(s => s.setTextureMode);
  const importedNpzMetadata = useSpaceflowUiStore(s => s.importedNpzMetadata);
  const [spaceflowGlobalTextureText, setSpaceflowGlobalTextureText] = useState('');
  const [spaceflowTextureExperimentPrompt, setSpaceflowTextureExperimentPrompt] = useState('');
  const [spaceflowTextureExperimentPromptEdited, setSpaceflowTextureExperimentPromptEdited] = useState(false);
  const [spaceflowGlobalTextureImagePath, setSpaceflowGlobalTextureImagePath] = useState('');
  const [spaceflowGlobalTextureImageFile, setSpaceflowGlobalTextureImageFile] = useState<File | null>(null);
  const [spaceflowLowTau, setSpaceflowLowTau] = useState('3.0');
  const [spaceflowHighTau, setSpaceflowHighTau] = useState('10.0');
  const [spaceflowPolyakTau, setSpaceflowPolyakTau] = useState('0.18');
  const [spaceflowRepaintSteps, setSpaceflowRepaintSteps] = useState('10');
  const [spaceflowTextureOptimSteps, setSpaceflowTextureOptimSteps] = useState('300');
  const [spaceflowOutputName, setSpaceflowOutputName] = useState('');
  const [spaceflowConvertYupToZup, setSpaceflowConvertYupToZup] = useState(true);
  const [spaceflowDryRun, setSpaceflowDryRun] = useState(false);
  const [projectName, setProjectName] = useState('spaceflow');

  const spaceflowPromptRef = useRef<HTMLInputElement>(null);
  const globalTextureFileInputRef = useRef<HTMLInputElement>(null);
  const inspectedSpaceflowRunRef = useRef<string | null>(null);

  const showToast = useCallback((msg: string, durationMs = 3000) => {
    setToast(msg);
    window.setTimeout(() => setToast(null), durationMs);
  }, []);

  const applySpaceflowMetadata = useCallback((metadata: NpzSpaceflowMetadata | null) => {
    if (!metadata) return;
    if (metadata.projectName) setProjectName(metadata.projectName);
    if (metadata.textPrompt) setSpaceflowTextPrompt(metadata.textPrompt);
    if (metadata.outputName !== undefined) setSpaceflowOutputName(metadata.outputName);
    if (metadata.textureMode) setSpaceflowTextureMode(metadata.textureMode);
    if (metadata.globalTextureText !== undefined) {
      setSpaceflowGlobalTextureText(metadata.globalTextureText);
      if (!metadata.textureMode && metadata.globalTextureText) setSpaceflowTextureMode('text');
    }
    if (!PUBLIC_DEMO && metadata.globalTextureImagePath !== undefined) {
      setSpaceflowGlobalTextureImagePath(metadata.globalTextureImagePath);
      if (!metadata.textureMode && metadata.globalTextureImagePath) setSpaceflowTextureMode('image');
    }
    if (metadata.textureExperimentPrompt) {
      setSpaceflowTextureExperimentPrompt(metadata.textureExperimentPrompt);
      setSpaceflowTextureExperimentPromptEdited(true);
    } else {
      setSpaceflowTextureExperimentPrompt('');
      setSpaceflowTextureExperimentPromptEdited(false);
    }
    setSpaceflowGlobalTextureImageFile(null);
    if (metadata.lowTau !== undefined) setSpaceflowLowTau(String(metadata.lowTau));
    if (metadata.highTau !== undefined) setSpaceflowHighTau(String(metadata.highTau));
    if (metadata.polyakTau !== undefined) setSpaceflowPolyakTau(String(metadata.polyakTau));
    if (metadata.repaintSteps !== undefined) setSpaceflowRepaintSteps(String(metadata.repaintSteps));
    if (metadata.textureOptimSteps !== undefined) setSpaceflowTextureOptimSteps(String(metadata.textureOptimSteps));
    if (metadata.convertYupToZup !== undefined) setSpaceflowConvertYupToZup(metadata.convertYupToZup);
    if (metadata.lowControlBBoxMargin !== undefined) setLowControlBBoxMargin(metadata.lowControlBBoxMargin);
  }, [setLowControlBBoxMargin, setSpaceflowTextureMode]);

  const inspectSpaceflowRunMesh = useCallback((run: SpaceflowRunStatus, automatic = false) => {
    const meshFile = pickSpaceflowInspectionMesh(run.output_files ?? []);
    if (!meshFile) return null;
    if (automatic && inspectedSpaceflowRunRef.current === run.run_id) return meshFile;
    inspectedSpaceflowRunRef.current = run.run_id;
    setMeshInspection({
      url: resolveSpaceflowUrl(meshFile.url),
      name: outputFileLabel(meshFile),
      runId: run.run_id,
      path: PUBLIC_DEMO ? undefined : meshFile.path,
      relativePath: meshFile.relative_path,
    });
    setShowSpaceflow(false);
    return meshFile;
  }, [setMeshInspection]);

  useEffect(() => {
    if (showSpaceflow) {
      window.setTimeout(() => spaceflowPromptRef.current?.focus(), 50);
    }
  }, [showSpaceflow]);

  const visiblePrimitives = useMemo(() => primitives.filter(p => p.visible), [primitives]);
  const spaceflowRunActive = spaceflowRun?.status === 'running' || spaceflowRun?.status === 'cancelling';
  const generatedTextureExperimentPrompt = useMemo(
    () => textureExperimentPromptTemplate(
      spaceflowTextPrompt,
      spaceflowGlobalTextureText,
      visiblePrimitives,
    ),
    [spaceflowGlobalTextureText, spaceflowTextPrompt, visiblePrimitives],
  );
  const textureExperimentPromptValue = spaceflowTextureExperimentPromptEdited
    ? spaceflowTextureExperimentPrompt
    : generatedTextureExperimentPrompt;

  const spaceflowMetadata = useMemo<NpzSpaceflowMetadata>(() => ({
    projectName,
    textPrompt: spaceflowTextPrompt,
    outputName: spaceflowOutputName,
    textureMode: spaceflowTextureMode,
    globalTextureText: spaceflowGlobalTextureText,
    globalTextureImagePath: PUBLIC_DEMO ? '' : spaceflowGlobalTextureImagePath,
    textureExperimentPrompt: textureExperimentPromptValue,
    primitiveNames: visiblePrimitives.map(p => p.name),
    localTextureTexts: visiblePrimitives.map(p => p.localTextureText ?? ''),
    localTextureImagePaths: visiblePrimitives.map(p => PUBLIC_DEMO ? '' : p.localTextureImagePath ?? ''),
    lowTau: Number(spaceflowLowTau),
    highTau: Number(spaceflowHighTau),
    polyakTau: Number(spaceflowPolyakTau),
    repaintSteps: Number(spaceflowRepaintSteps),
    textureOptimSteps: Number(spaceflowTextureOptimSteps),
    convertYupToZup: spaceflowConvertYupToZup,
    lowControlBBoxMargin,
  }), [projectName, spaceflowTextPrompt, spaceflowOutputName, spaceflowTextureMode,
    spaceflowGlobalTextureText, spaceflowGlobalTextureImagePath, textureExperimentPromptValue,
    visiblePrimitives, spaceflowLowTau, spaceflowHighTau, spaceflowPolyakTau,
    spaceflowRepaintSteps, spaceflowTextureOptimSteps, spaceflowConvertYupToZup,
    lowControlBBoxMargin]);

  useEffect(() => {
    if (importedNpzMetadata) applySpaceflowMetadata(importedNpzMetadata.metadata);
  }, [applySpaceflowMetadata, importedNpzMetadata]);

  useEffect(() => {
    if (!spaceflowRun?.run_id || !spaceflowRunActive) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await getSpaceflowRunStatus(spaceflowRun.run_id);
        if (cancelled) return;
        setSpaceflowRun(status.run);
        setSpaceflowLogTail(status.logTail);
        if (status.run.status !== 'running' && status.run.status !== 'cancelling') {
          setSpaceflowRunning(false);
          const inspectedFile =
            status.run.status === 'succeeded'
              ? inspectSpaceflowRunMesh(status.run, true)
              : null;
          showToast(
            inspectedFile
              ? `SpaceFlow succeeded: loaded ${outputFileLabel(inspectedFile)}`
              : `SpaceFlow ${status.run.status}: ${PUBLIC_DEMO ? status.run.run_id : status.run.output_dir ?? spaceflowRun.run_id}`,
            8000,
          );
        }
      } catch (err) {
        if (!cancelled) setSpaceflowLogTail(`Status check failed: ${err instanceof Error ? err.message : err}`);
      }
    };
    const id = window.setInterval(() => void poll(), 5000);
    void poll();
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [inspectSpaceflowRunMesh, showToast, spaceflowRun?.run_id, spaceflowRunActive]);

  const refreshSpaceflowHistory = useCallback(async () => {
    setSpaceflowHistoryLoading(true);
    try {
      const entries = await fetchSpaceflowHistory(30);
      setSpaceflowHistory(entries);
    } catch (err) {
      showToast(`History failed: ${err instanceof Error ? err.message : err}`, 8000);
    } finally {
      setSpaceflowHistoryLoading(false);
    }
  }, [showToast]);

  const handleSaveSpaceflowInputs = useCallback(async () => {
    if (PUBLIC_DEMO || spaceflowSaving || visiblePrimitives.length === 0) return;
    setSpaceflowSaving(true);
    try {
      const lowTau = Number(spaceflowLowTau);
      const highTau = Number(spaceflowHighTau);
      if (!Number.isFinite(lowTau) || !Number.isFinite(highTau)) {
        throw new Error('Enter finite low and high tau values before saving.');
      }
      const { entry, bundle } = await saveSpaceflowAsset({
        projectName,
        primitives: visiblePrimitives,
        lowTau,
        highTau,
        lowControlBBoxMargin,
        metadata: spaceflowMetadata,
      });
      showToast(
        `Saved SpaceFlow inputs (${bundle.counts.high} high, ${bundle.counts.low} low)\n${entry.asset_dir}`,
        8000,
      );
      if (showSpaceflowHistoryPanel) await refreshSpaceflowHistory();
    } catch (err) {
      showToast(`SpaceFlow save failed: ${err instanceof Error ? err.message : err}`, 10000);
    } finally {
      setSpaceflowSaving(false);
    }
  }, [
    lowControlBBoxMargin,
    projectName,
    refreshSpaceflowHistory,
    showSpaceflowHistoryPanel,
    showToast,
    spaceflowHighTau,
    spaceflowLowTau,
    spaceflowSaving,
    spaceflowMetadata,
    visiblePrimitives,
  ]);

  const handleOpenSpaceflowHistory = useCallback(async (entry: SpaceflowHistoryEntry) => {
    try {
      const { primitives: prims, metadata } = await openSpaceflowAsset(entry);
      loadPreset(prims);
      applySpaceflowMetadata(metadata);
      setProjectName(entry.project_name || projectName);
      showToast(`Loaded ${prims.length} primitives from ${entry.project_name}`);
      setShowSpaceflow(false);
    } catch (err) {
      showToast(`Open saved asset failed: ${err instanceof Error ? err.message : err}`, 8000);
    }
  }, [applySpaceflowMetadata, loadPreset, projectName, showToast]);

  const handleStartSpaceflowRun = useCallback(async (experimentType?: SpaceflowExperimentType) => {
    if (spaceflowRunning || visiblePrimitives.length === 0) return;
    const experimentMode = Boolean(experimentType);
    const lowTau = Number.parseFloat(spaceflowLowTau);
    const highTau = Number.parseFloat(spaceflowHighTau);
    const polyakTau = Number.parseFloat(spaceflowPolyakTau);
    const repaintStepsRaw = spaceflowRepaintSteps.trim();
    const repaintSteps = Number.parseInt(repaintStepsRaw, 10);
    const textureOptimStepsRaw = spaceflowTextureOptimSteps.trim();
    const textureOptimSteps = Number.parseInt(textureOptimStepsRaw, 10);
    if (!Number.isFinite(lowTau) || !Number.isFinite(highTau) || highTau < lowTau) {
      showToast('High tau must be greater than or equal to low tau.', 5000);
      return;
    }
    if (!/^\d+$/.test(repaintStepsRaw) || !Number.isInteger(repaintSteps)) {
      showToast('Repaint steps must be a non-negative whole number.', 5000);
      return;
    }
    if (!/^\d+$/.test(textureOptimStepsRaw) || !Number.isInteger(textureOptimSteps) || textureOptimSteps < 2) {
      showToast('Texture optimization steps must be at least 2.', 5000);
      return;
    }
    if (!spaceflowTextPrompt.trim()) {
      showToast('Enter a SpaceFlow text prompt.', 5000);
      return;
    }
    const includesTextureExperiment = experimentType === 'texture' || experimentType === 'full';
    if (includesTextureExperiment && spaceflowTextureMode !== 'text') {
      showToast('Texture and full experiments support text texture guidance only.', 6000);
      return;
    }
    const textureExperimentPrompt = includesTextureExperiment
      ? textureExperimentPromptValue.trim()
      : '';
    if (includesTextureExperiment && !textureExperimentPrompt) {
      showToast('Enter a TRELLIS texture experiment prompt.', 6000);
      return;
    }
    if (
      spaceflowTextureMode === 'image' &&
      !(PUBLIC_DEMO ? '' : spaceflowGlobalTextureImagePath.trim()) &&
      !spaceflowGlobalTextureImageFile
    ) {
      showToast(PUBLIC_DEMO ? 'Choose a global texture image.' : 'Choose a global texture image or enter a cluster image path.', 5000);
      return;
    }
    setSpaceflowRunning(true);
    setSpaceflowLogTail('');
    try {
      const globalTextureText = spaceflowGlobalTextureText.trim() || spaceflowTextPrompt.trim();
      const globalTextureImagePath = PUBLIC_DEMO ? '' : spaceflowGlobalTextureImagePath.trim();
      const localTextureTexts = visiblePrimitives.map(p => (p.localTextureText ?? '').trim());
      const localTextureImagePaths = visiblePrimitives.map(p => PUBLIC_DEMO ? '' : (p.localTextureImagePath ?? '').trim());
      const localTextureImageFiles = visiblePrimitives.map(p => spaceflowLocalTextureImageFiles[p.id] ?? null);
      const outputName = spaceflowOutputName.trim() || outputNameFromPrompt(spaceflowTextPrompt);
      const experimentSuffix = experimentType === 'texture'
        ? '_texture_experiment'
        : experimentType === 'full'
          ? '_full_experiment'
          : '_experiment';
      const runOutputName = experimentMode && !outputName.endsWith(experimentSuffix)
        ? `${outputName}${experimentSuffix}`
        : outputName;
      const runLabel = experimentType === 'texture'
        ? 'SpaceFlow texture experiment'
        : experimentType === 'full'
          ? 'SpaceFlow full experiment'
          : experimentMode
          ? 'SpaceFlow experiment'
          : 'SpaceFlow';
      const { run, bundle } = await startSpaceflowRun({
        projectName,
        primitives: visiblePrimitives,
        textureImageFile: spaceflowGlobalTextureImageFile,
        localTextureImageFiles,
        runConfig: {
          textPrompt: spaceflowTextPrompt.trim(),
          appearanceMode: spaceflowTextureMode,
          appearanceText: globalTextureText,
          appearanceImagePath: globalTextureImagePath,
          textureMode: spaceflowTextureMode,
          globalTextureText,
          globalTextureImagePath,
          localTextureTexts,
          localTextureImagePaths,
          textureExperimentPrompt: includesTextureExperiment ? textureExperimentPrompt : undefined,
          lowTau,
          highTau,
          polyakTau: Number.isFinite(polyakTau) ? polyakTau : 0.18,
          repaintSteps,
          textureOptimSteps,
          outputName: runOutputName,
          convertYupToZup: spaceflowConvertYupToZup,
          lowControlBBoxMargin,
          dryRun: spaceflowDryRun,
          experimentMode,
          experimentType,
        },
      });
      setSpaceflowRun(run);
      if (run.status === 'succeeded') {
        inspectSpaceflowRunMesh(run, true);
      }
      const runLocation = PUBLIC_DEMO ? run.run_id : run.output_dir ?? run.run_id;
      showToast(
        `${spaceflowDryRun ? 'Prepared' : 'Started'} ${runLabel} (${bundle.counts.high} high, ${bundle.counts.low} low)\n${runLocation}`,
        8000,
      );
      if (showSpaceflowHistoryPanel) await refreshSpaceflowHistory();
      if (run.status !== 'running') setSpaceflowRunning(false);
    } catch (err) {
      showToast(`SpaceFlow run failed: ${err instanceof Error ? err.message : err}`, 10000);
      setSpaceflowRunning(false);
    }
  }, [
    inspectSpaceflowRunMesh,
    lowControlBBoxMargin,
    projectName,
    refreshSpaceflowHistory,
    showSpaceflowHistoryPanel,
    showToast,
    spaceflowConvertYupToZup,
    spaceflowDryRun,
    spaceflowGlobalTextureImageFile,
    spaceflowGlobalTextureImagePath,
    spaceflowGlobalTextureText,
    spaceflowHighTau,
    spaceflowLocalTextureImageFiles,
    spaceflowLowTau,
    spaceflowOutputName,
    spaceflowPolyakTau,
    spaceflowRepaintSteps,
    spaceflowRunning,
    spaceflowTextPrompt,
    spaceflowTextureOptimSteps,
    spaceflowTextureMode,
    textureExperimentPromptValue,
    visiblePrimitives,
  ]);

  const handleStopSpaceflowRun = useCallback(async () => {
    if (!spaceflowRun?.run_id || !spaceflowRunActive) return;
    try {
      const run = await stopSpaceflowRun(spaceflowRun.run_id);
      setSpaceflowRun(run);
      setSpaceflowRunning(run.status === 'running' || run.status === 'cancelling');
      showToast(`Stopping SpaceFlow: ${PUBLIC_DEMO ? run.run_id : run.output_dir ?? run.run_id}`, 5000);
    } catch (err) {
      showToast(`Stop failed: ${err instanceof Error ? err.message : err}`, 8000);
    }
  }, [showToast, spaceflowRun?.run_id, spaceflowRunActive]);

  const handleDownloadNpz = useCallback(async () => {
    try {
      const exports: PrimitiveExport[] = visiblePrimitives.map(primitiveToExport);
      const blob = await exportNpz(exports, { metadata: spaceflowMetadata });
      const filename = `${safeName(projectName, 'spaceflow')}.npz`;
      downloadBlob(blob, filename);
      showToast(`Downloaded ${filename}`);
    } catch (err) {
      showToast(`Export failed: ${err instanceof Error ? err.message : err}`, 8000);
    }
    setShowExport(false);
  }, [projectName, showToast, spaceflowMetadata, visiblePrimitives]);

  const handleDownloadRendering = useCallback(async () => {
    if (visiblePrimitives.length === 0) return;
    try {
      const blob = await captureSuperquadricRenderBlob();
      if (!blob) {
        showToast('Could not render the superquadrics. Switch back to the viewport and try again.', 5000);
        return;
      }
      const filename = `${safeName(projectName, 'spaceflow')}_render.png`;
      downloadBlob(blob, filename);
      showToast(`Downloaded ${filename}`);
    } catch (err) {
      showToast(`Render export failed: ${err instanceof Error ? err.message : err}`, 8000);
    } finally {
      setShowExport(false);
    }
  }, [projectName, showToast, visiblePrimitives.length]);

  const handleCopyJson = useCallback(async () => {
    const data = visiblePrimitives.map(p => ({
      name: p.name,
      controlLevel: p.controlLevel,
      scales: p.scales,
      shapes: p.shapes,
      translation: p.translation,
      eulerDeg: p.eulerDeg,
      ...(p.tapering !== undefined ? { tapering: p.tapering } : {}),
      ...(p.bending !== undefined ? { bending: p.bending } : {}),
      ...(p.localTextureText ? { localTextureText: p.localTextureText } : {}),
      ...(!PUBLIC_DEMO && p.localTextureImagePath ? { localTextureImagePath: p.localTextureImagePath } : {}),
    }));
    try {
      await writeClipboardText(JSON.stringify(data, null, 2));
      showToast('Copied JSON preset to clipboard');
      setShowExport(false);
    } catch (err) {
      showToast(`Copy failed: ${err instanceof Error ? err.message : err}`, 8000);
    }
  }, [showToast, visiblePrimitives]);

  const handleImportJson = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,application/json';
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        const text = await file.text();
        const data = JSON.parse(text) as Array<{
          name?: string;
          controlLevel?: 'high' | 'low';
          scales: [number, number, number];
          shapes: [number, number];
          translation: [number, number, number];
          eulerDeg?: [number, number, number];
          rotation?: number[][];
          tapering?: [number, number];
          bending?: [number, number, number, number, number, number];
          localTextureText?: string;
          localTextureImagePath?: string;
        }>;
        const prims: Primitive[] = data.map((d) => {
          const euler: [number, number, number] = d.eulerDeg ?? (d.rotation ? matrixToEuler(d.rotation) : [0, 0, 0]);
          const rotation = d.rotation ?? eulerToMatrix(euler);
          return {
            id: `imported_${++nextPresetId}_${Date.now()}`,
            name: d.name ?? `Imported ${nextPresetId}`,
            visible: true,
            controlLevel: d.controlLevel ?? 'high',
            scales: d.scales,
            shapes: d.shapes,
            translation: d.translation,
            rotation,
            eulerDeg: euler,
            ...(d.tapering !== undefined ? { tapering: d.tapering } : {}),
            ...(d.bending !== undefined ? { bending: d.bending } : {}),
            ...(d.localTextureText ? { localTextureText: d.localTextureText } : {}),
            ...(!PUBLIC_DEMO && d.localTextureImagePath ? { localTextureImagePath: d.localTextureImagePath } : {}),
          };
        });
        loadPreset(maybeRescalePrimitivesForEditor(prims));
        showToast(`Loaded ${prims.length} primitives from JSON`);
      } catch (err) {
        showToast(`Import failed: ${err instanceof Error ? err.message : err}`, 8000);
      }
    };
    input.click();
    setShowImport(false);
  }, [loadPreset, showToast]);

  const handleImportNpz = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.npz,application/octet-stream';
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        const stem = safeName(file.name.replace(/\.npz$/i, ''), 'npz');
        const { primitives: prims, metadata } = await importNpzWithMetadata(file, stem, { basisZUpToYUp: false });
        loadPreset(prims);
        applySpaceflowMetadata(metadata);
        const textureCount = prims.filter(p => (p.localTextureText ?? '').trim() || (p.localTextureImagePath ?? '').trim()).length;
        const withTexture = metadata?.globalTextureText || metadata?.globalTextureImagePath || textureCount > 0;
        showToast(
          `Loaded ${prims.length} primitives from ${file.name}${withTexture ? ` with ${textureCount} local texture prompt${textureCount === 1 ? '' : 's'}` : ''}`,
        );
      } catch (err) {
        showToast(`NPZ import failed: ${err instanceof Error ? err.message : err}`, 6000);
      }
    };
    input.click();
    setShowImport(false);
  }, [applySpaceflowMetadata, loadPreset, showToast]);

  const handleLoadDemoPreset = useCallback(async (preset: DemoPreset) => {
    try {
      const response = await fetch(preset.npzUrl);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      const prims = await importNpzToPrimitives(blob, preset.id, { basisZUpToYUp: false });
      loadPreset(prims);
      setProjectName(preset.id);
      setSpaceflowTextPrompt(preset.prompt);
      setSpaceflowGlobalTextureText(preset.texturePrompt);
      setSpaceflowOutputName(preset.outputName);
      setSpaceflowTextureMode('text');
      setSpaceflowGlobalTextureImageFile(null);
      setSpaceflowGlobalTextureImagePath('');
      setSpaceflowTextureExperimentPromptEdited(false);
      setSpaceflowTextureExperimentPrompt('');
      const high = prims.filter(prim => prim.controlLevel === 'high').length;
      const low = prims.filter(prim => prim.controlLevel === 'low').length;
      showToast(`Loaded ${preset.label} demo (${high} high, ${low} low)`);
    } catch (err) {
      showToast(`Demo preset failed: ${err instanceof Error ? err.message : err}`, 8000);
    } finally {
      setShowImport(false);
    }
  }, [loadPreset, setSpaceflowTextureMode, showToast]);

  const handleOpenNpzPath = useCallback(() => {
    const path = window.prompt('Path to .npz');
    const source = path?.trim();
    if (!source) return;
    if (!source.toLowerCase().split(/[?#]/, 1)[0]?.endsWith('.npz')) {
      showToast('Please enter a .npz path.', 5000);
      return;
    }
    setShowImport(false);
    window.location.assign(npzEditorUrl(source));
  }, [showToast]);

  const handleLoadTemplate = useCallback((template: string) => {
    const templates: Record<string, () => Primitive[]> = {
      'Single Ellipsoid': () => {
        const euler: [number, number, number] = [0, 0, 0];
        return [{
          id: `t_${++nextPresetId}_${Date.now()}`,
          name: 'Ellipsoid',
          visible: true,
          controlLevel: 'high',
          scales: [0.5, 0.5, 1],
          shapes: [1, 1],
          translation: [0, 0, 0],
          rotation: eulerToMatrix(euler),
          eulerDeg: euler,
        }];
      },
      'Table (5 parts)': () => {
        const leg = (x: number, y: number, idx: number): Primitive => ({
          id: `t_${++nextPresetId}_${Date.now()}`,
          name: `Leg ${idx}`,
          visible: true,
          controlLevel: 'high',
          scales: [0.06, 0.06, 0.4],
          shapes: [0.4, 0.4],
          translation: [x, y, -0.44],
          rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          eulerDeg: [0, 0, 0],
        });
        return [
          {
            id: `t_${++nextPresetId}_${Date.now()}`,
            name: 'Top',
            visible: true,
            controlLevel: 'high',
            scales: [0.8, 0.5, 0.04],
            shapes: [0.3, 0.3],
            translation: [0, 0, 0],
            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            eulerDeg: [0, 0, 0],
          },
          leg(-0.65, -0.4, 1),
          leg(0.65, -0.4, 2),
          leg(-0.65, 0.4, 3),
          leg(0.65, 0.4, 4),
        ];
      },
      'Chair (6 parts)': () => {
        const leg = (x: number, y: number, idx: number): Primitive => ({
          id: `t_${++nextPresetId}_${Date.now()}`,
          name: `Leg ${idx}`,
          visible: true,
          controlLevel: 'high',
          scales: [0.05, 0.05, 0.35],
          shapes: [0.4, 0.4],
          translation: [x, y, -0.39],
          rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          eulerDeg: [0, 0, 0],
        });
        return [
          {
            id: `t_${++nextPresetId}_${Date.now()}`,
            name: 'Seat',
            visible: true,
            controlLevel: 'high',
            scales: [0.5, 0.45, 0.04],
            shapes: [0.3, 0.3],
            translation: [0, 0, 0],
            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            eulerDeg: [0, 0, 0],
          },
          {
            id: `t_${++nextPresetId}_${Date.now()}`,
            name: 'Backrest',
            visible: true,
            controlLevel: 'high',
            scales: [0.5, 0.04, 0.35],
            shapes: [0.3, 0.3],
            translation: [0, -0.43, 0.36],
            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            eulerDeg: [0, 0, 0],
          },
          leg(-0.42, 0.35, 1),
          leg(0.42, 0.35, 2),
          leg(-0.42, -0.35, 3),
          leg(0.42, -0.35, 4),
        ];
      },
    };
    const factory = templates[template];
    if (factory) {
      loadPreset(factory());
      showToast(`Loaded "${template}" template`);
    }
    setShowImport(false);
    setShowExport(false);
  }, [loadPreset, showToast]);

  const highCount = visiblePrimitives.filter(p => p.controlLevel === 'high').length;
  const lowCount = visiblePrimitives.filter(p => p.controlLevel === 'low').length;
  const allOrtho = visiblePrimitives.every(p => isOrthogonal(p.rotation));
  const hasWarnings = !allOrtho;
  const canStartSpaceflow = !spaceflowRunning && visiblePrimitives.length > 0 && highCount > 0 && lowCount > 0;
  const canStartTextureExperiment = canStartSpaceflow && spaceflowTextureMode === 'text';
  const spaceflowOutputFiles = useMemo(() => spaceflowRun?.output_files ?? [], [spaceflowRun?.output_files]);
  const spaceflowRunWarnings = (spaceflowRun?.warnings ?? [])
    .map(warning => String(warning.message ?? '').trim())
    .filter(Boolean);
  const spaceflowInspectionMesh = pickSpaceflowInspectionMesh(spaceflowOutputFiles);
  const spaceflowDownloadGlb = useMemo<DownloadableGlbMesh | null>(() => {
    const runGlb = pickSpaceflowInspectionMesh(spaceflowOutputFiles);
    if (runGlb) return downloadableGlbFromOutput(runGlb, spaceflowRun?.run_id);
    if (
      meshInspection &&
      (
        isGlbPath(meshInspection.relativePath) ||
        isGlbPath(meshInspection.path) ||
        isGlbPath(meshInspection.url)
      )
    ) {
      const fallbackBase = filenameBaseFromPath(
        meshInspection.relativePath ?? meshInspection.path ?? meshInspection.url,
        meshInspection.name || 'mesh',
      );
      return {
        url: meshInspection.url,
        filename: `${safeName(fallbackBase, 'spaceflow_mesh')}.glb`,
        label: meshInspection.name,
      };
    }
    return null;
  }, [meshInspection, spaceflowOutputFiles, spaceflowRun?.run_id]);
  const spaceflowPreviewImage =
    spaceflowOutputFiles.find(file => (
      file.relative_path === 'struct_renders/000.png' ||
      file.relative_path.endsWith('/struct_renders/000.png')
    )) ??
    spaceflowOutputFiles.find(file => file.kind === 'image');
  const spaceflowVisibleOutputs = spaceflowOutputFiles
    .filter(file => file.kind !== 'log')
    .slice(0, 14);
  const handleDownloadGlbMesh = useCallback(async () => {
    if (!spaceflowDownloadGlb) return;
    try {
      const res = await fetch(spaceflowDownloadGlb.url);
      if (!res.ok) throw new Error(`download returned ${res.status}`);
      const blob = await res.blob();
      downloadBlob(blob, spaceflowDownloadGlb.filename);
      showToast(`Downloaded ${spaceflowDownloadGlb.filename}`);
    } catch (err) {
      showToast(`GLB download failed: ${err instanceof Error ? err.message : err}`, 8000);
    } finally {
      setShowExport(false);
    }
  }, [showToast, spaceflowDownloadGlb]);

  return (
    <div className="top-bar">
      <div className="top-left">
        <span className="app-name">SpaceFlow</span>
      </div>

      <div className="top-center">
        <button className="toolbar-btn" onClick={undo} disabled={undoStack.length === 0} title="Undo" aria-label="Undo">
          <UndoIcon size={16} />
        </button>
        <button className="toolbar-btn" onClick={redo} disabled={redoStack.length === 0} title="Redo" aria-label="Redo">
          <RedoIcon size={16} />
        </button>
        <span className="separator" />
        <button className="toolbar-btn" onClick={() => handleLoadTemplate('Single Ellipsoid')} title="Single ellipsoid" aria-label="Load single ellipsoid template">
          <CircleIcon size={16} />
        </button>
        <button className="toolbar-btn" onClick={() => handleLoadTemplate('Table (5 parts)')} title="Table template" aria-label="Load table template">
          <TableIcon size={16} />
        </button>
        <button className="toolbar-btn" onClick={() => handleLoadTemplate('Chair (6 parts)')} title="Chair template" aria-label="Load chair template">
          <ChairIcon size={16} />
        </button>
        <span className="separator" />
        <div className={`export-dropdown rotate-all-group${showRotateAll ? ' is-open' : ''}`}>
          <button
            type="button"
            className={`toolbar-btn toolbar-btn-menu${showRotateAll ? ' is-open' : ''}`}
            onClick={() => {
              setShowRotateAll(v => !v);
              setShowImport(false);
              setShowExport(false);
              setShowSpaceflow(false);
            }}
            disabled={primitives.length === 0}
            title="Rotate all primitives"
          >
            <RotateIcon className="toolbar-btn-menu-icon" size={14} />
            <span className="toolbar-btn-menu-label">Rotate All</span>
          </button>
          {showRotateAll && (
            <div className="dropdown-menu rotate-all-menu">
              <button type="button" className="dropdown-item" onClick={() => rotateAllWorld([90, 0, 0])}>X +90</button>
              <button type="button" className="dropdown-item" onClick={() => rotateAllWorld([-90, 0, 0])}>X -90</button>
              <button type="button" className="dropdown-item" onClick={() => rotateAllWorld([0, 90, 0])}>Y +90</button>
              <button type="button" className="dropdown-item" onClick={() => rotateAllWorld([0, -90, 0])}>Y -90</button>
              <button type="button" className="dropdown-item" onClick={() => rotateAllWorld([0, 0, 90])}>Z +90</button>
              <button type="button" className="dropdown-item" onClick={() => rotateAllWorld([0, 0, -90])}>Z -90</button>
            </div>
          )}
        </div>
        <span className="separator" />
        <div className={`generate-group ${showSpaceflow ? 'is-open' : ''}`}>
          {!showSpaceflow ? (
            <button
              className="btn-generate btn-spaceflow"
              onClick={() => {
                setShowImport(false);
                setShowExport(false);
                setShowRotateAll(false);
                setShowSpaceflow(true);
              }}
              disabled={spaceflowSaving}
              title="Run SpaceFlow from the current superquadrics"
            >
              <SparklesIcon size={14} />
              {spaceflowRunning ? 'Running...' : 'SpaceFlow'}
            </button>
          ) : (
            <>
              <div
                className="generate-popover-backdrop"
                onClick={() => {
                  if (!spaceflowSaving && !spaceflowRunning) setShowSpaceflow(false);
                }}
                aria-hidden
              />
              <div className="generate-open-bar">
                <div className="gen-mode-row" role="group" aria-label="SpaceFlow">
                  <span className="spaceflow-open-label">SpaceFlow refinement</span>
                </div>
                <button
                  className="toolbar-btn toolbar-btn-close"
                  onClick={() => setShowSpaceflow(false)}
                  disabled={spaceflowSaving || spaceflowRunning}
                  title="Close"
                  type="button"
                  aria-label="Close SpaceFlow panel"
                >
                  <XIcon size={15} />
                </button>
              </div>
              <div
                className="generate-popover spaceflow-popover"
                role="dialog"
                aria-label="Run SpaceFlow"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="spaceflow-panel-block">
                  <div className="spaceflow-panel-head">
                    <span className="spaceflow-panel-title">Inputs</span>
                    {!PUBLIC_DEMO && (
                      <div className="spaceflow-head-actions">
                        <button
                          type="button"
                          className="spaceflow-panel-action"
                          onClick={() => {
                            const next = !showSpaceflowHistoryPanel;
                            setShowSpaceflowHistoryPanel(next);
                            if (next) void refreshSpaceflowHistory();
                          }}
                          disabled={spaceflowHistoryLoading}
                          title="Show saved SpaceFlow input bundles"
                        >
                          {showSpaceflowHistoryPanel ? 'Hide saved' : 'Saved inputs'}
                        </button>
                        <button
                          type="button"
                          className="spaceflow-panel-action"
                          onClick={handleSaveSpaceflowInputs}
                          disabled={spaceflowSaving || visiblePrimitives.length === 0}
                          title="Save SpaceFlow input files"
                        >
                          {spaceflowSaving ? 'Saving...' : 'Save inputs'}
                        </button>
                      </div>
                    )}
                  </div>
                  <p className="spaceflow-summary">
                    {highCount} high control, {lowCount} low control
                  </p>
                  <p className="spaceflow-panel-hint">
                    The current scene is exported as all, high-control, and low-control bounding-box inputs for SpaceFlow.
                  </p>
                </div>

                {!PUBLIC_DEMO && showSpaceflowHistoryPanel && (
                  <div className="spaceflow-panel-block spaceflow-history-block">
                    <div className="spaceflow-panel-head">
                      <span className="spaceflow-panel-title">Saved inputs</span>
                      <button
                        type="button"
                        className="spaceflow-panel-action"
                        onClick={() => void refreshSpaceflowHistory()}
                        disabled={spaceflowHistoryLoading}
                      >
                        {spaceflowHistoryLoading ? 'Loading...' : 'Refresh'}
                      </button>
                    </div>
                    <div className="spaceflow-history-list">
                      {spaceflowHistory.length === 0 && (
                        <p className="spaceflow-panel-hint">No saved SpaceFlow input bundles yet.</p>
                      )}
                      {spaceflowHistory.map(entry => (
                        <button
                          key={entry.id}
                          type="button"
                          className="spaceflow-history-item"
                          onClick={() => void handleOpenSpaceflowHistory(entry)}
                          title={`${entry.paths.all}\n${entry.paths.high_control}\n${entry.paths.low_control_bbox}`}
                        >
                          <span className="spaceflow-history-title">{entry.project_name}</span>
                          <span className="spaceflow-history-meta">
                            {entry.counts?.high ?? '?'} high, {entry.counts?.low ?? '?'} low, {entry.saved_at}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <label className="generate-popover-label" htmlFor="sq-spaceflow-prompt-input">
                  Shape prompt
                </label>
                <input
                  id="sq-spaceflow-prompt-input"
                  ref={spaceflowPromptRef}
                  type="text"
                  className="generate-input generate-input-popover"
                  value={spaceflowTextPrompt}
                  onChange={(e) => setSpaceflowTextPrompt(e.target.value)}
                  disabled={spaceflowRunning}
                  placeholder="e.g. A chair"
                />

                <div className="spaceflow-panel-block spaceflow-texture-block">
                  <div className="spaceflow-panel-head">
                    <span className="spaceflow-panel-title">Texture guidance</span>
                    <div className="gen-mode-row">
                      <button
                        type="button"
                        className={`gen-mode-btn ${spaceflowTextureMode === 'text' ? 'active' : ''}`}
                        onClick={() => setSpaceflowTextureMode('text')}
                        disabled={spaceflowRunning}
                      >
                        Text
                      </button>
                      <button
                        type="button"
                        className={`gen-mode-btn ${spaceflowTextureMode === 'image' ? 'active' : ''}`}
                        onClick={() => setSpaceflowTextureMode('image')}
                        disabled={spaceflowRunning}
                      >
                        Image
                      </button>
                    </div>
                  </div>
                  {spaceflowTextureMode === 'text' ? (
                    <>
                      <input
                        type="text"
                        className="generate-input generate-input-popover"
                        value={spaceflowGlobalTextureText}
                        onChange={(e) => setSpaceflowGlobalTextureText(e.target.value)}
                        disabled={spaceflowRunning}
                        placeholder="Global texture text; defaults to the shape prompt"
                      />
                      <label className="spaceflow-experiment-prompt-field">
                        <span>TRELLIS experiment prompt</span>
                        <textarea
                          className="generate-input generate-input-popover spaceflow-experiment-prompt-input"
                          value={textureExperimentPromptValue}
                          onChange={(e) => {
                            setSpaceflowTextureExperimentPromptEdited(true);
                            setSpaceflowTextureExperimentPrompt(e.target.value);
                          }}
                          disabled={spaceflowRunning}
                          placeholder={generatedTextureExperimentPrompt}
                        />
                      </label>
                    </>
                  ) : (
                    <div className="spaceflow-image-inputs">
                      {!PUBLIC_DEMO && (
                        <input
                          type="text"
                          className="generate-input generate-input-popover"
                          value={spaceflowGlobalTextureImagePath}
                          onChange={(e) => setSpaceflowGlobalTextureImagePath(e.target.value)}
                          disabled={spaceflowRunning}
                          placeholder="Global image path on cluster, or choose file below"
                        />
                      )}
                      <label className="spaceflow-file-picker">
                        <input
                          ref={globalTextureFileInputRef}
                          type="file"
                          accept="image/*"
                          onClick={(e) => {
                            e.currentTarget.value = '';
                          }}
                          onChange={(e) => setSpaceflowGlobalTextureImageFile(e.target.files?.[0] ?? null)}
                          disabled={spaceflowRunning}
                        />
                        <span>{spaceflowGlobalTextureImageFile ? spaceflowGlobalTextureImageFile.name : 'Choose global texture image'}</span>
                      </label>
                      {((!PUBLIC_DEMO && spaceflowGlobalTextureImagePath) || spaceflowGlobalTextureImageFile) && (
                        <button
                          type="button"
                          className="spaceflow-file-clear-btn"
                          onClick={() => {
                            setSpaceflowGlobalTextureImagePath('');
                            setSpaceflowGlobalTextureImageFile(null);
                            if (globalTextureFileInputRef.current) {
                              globalTextureFileInputRef.current.value = '';
                            }
                          }}
                          disabled={spaceflowRunning}
                        >
                          Clear
                        </button>
                      )}
                      </div>
                  )}
                </div>

                <div className="spaceflow-panel-block">
                  <div className="spaceflow-panel-head">
                    <span className="spaceflow-panel-title">Run options</span>
                  </div>
                  <div className="spaceflow-number-grid">
                    <label className="spaceflow-number-field">
                      <span>Low tau</span>
                      <input className="num-input" type="number" step="0.5" value={spaceflowLowTau} onChange={(e) => setSpaceflowLowTau(e.target.value)} disabled={spaceflowRunning} />
                    </label>
                    <label className="spaceflow-number-field">
                      <span>High tau</span>
                      <input className="num-input" type="number" step="0.5" value={spaceflowHighTau} onChange={(e) => setSpaceflowHighTau(e.target.value)} disabled={spaceflowRunning} />
                    </label>
                    <label className="spaceflow-number-field">
                      <span>Polyak tau</span>
                      <input className="num-input" type="number" step="0.01" value={spaceflowPolyakTau} onChange={(e) => setSpaceflowPolyakTau(e.target.value)} disabled={spaceflowRunning} />
                    </label>
                    <label className="spaceflow-number-field">
                      <span>Repaint steps</span>
                      <input className="num-input" type="number" min="0" step="1" value={spaceflowRepaintSteps} onChange={(e) => setSpaceflowRepaintSteps(e.target.value)} disabled={spaceflowRunning} />
                    </label>
                    <label className="spaceflow-number-field">
                      <span>Texture optim steps</span>
                      <input className="num-input" type="number" min="2" step="1" value={spaceflowTextureOptimSteps} onChange={(e) => setSpaceflowTextureOptimSteps(e.target.value)} disabled={spaceflowRunning} />
                    </label>
                    <label className="spaceflow-number-field">
                      <span>Output name</span>
                      <input className="num-input" type="text" value={spaceflowOutputName} onChange={(e) => setSpaceflowOutputName(e.target.value)} disabled={spaceflowRunning} placeholder={outputNameFromPrompt(spaceflowTextPrompt)} />
                    </label>
                  </div>
                  <label className="spaceflow-toggle-row">
                    <input
                      type="checkbox"
                      checked={spaceflowConvertYupToZup}
                      onChange={(e) => setSpaceflowConvertYupToZup(e.target.checked)}
                      disabled={spaceflowRunning}
                    />
                    <span>Convert generated mesh from Y-up to Z-up</span>
                  </label>
                  <label className="spaceflow-toggle-row">
                    <input
                      type="checkbox"
                      checked={spaceflowDryRun}
                      onChange={(e) => setSpaceflowDryRun(e.target.checked)}
                      disabled={spaceflowRunning}
                    />
                    <span>Dry run: save inputs and build command without GPU work</span>
                  </label>
                </div>

                <div className="generate-popover-footer">
                  <button
                    className="btn-generate-go"
                    type="button"
                    onClick={() => void handleStartSpaceflowRun()}
                    disabled={!canStartSpaceflow}
                    title="Requires at least one visible high-control and one visible low-control primitive"
                  >
                    {spaceflowRunning ? 'Running' : spaceflowDryRun ? 'Dry Run' : 'Run'}
                  </button>
                  <button
                    className="btn-generate-go btn-spaceflow-experiment"
                    type="button"
                    onClick={() => void handleStartSpaceflowRun('geometry')}
                    disabled={!canStartSpaceflow}
                    title="Run the preset SpaceFlow comparison variants"
                  >
                    Structure exp
                  </button>
                  <button
                    className="btn-generate-go btn-spaceflow-texture-experiment"
                    type="button"
                    onClick={() => void handleStartSpaceflowRun('texture')}
                    disabled={!canStartTextureExperiment}
                    title={spaceflowTextureMode === 'text'
                      ? 'Run texture-focused TRELLIS and SpaceFlow comparison variants'
                      : 'Texture experiment requires text texture guidance'}
                  >
                    Texture exp
                  </button>
                  <button
                    className="btn-generate-go btn-spaceflow-texture-experiment"
                    type="button"
                    onClick={() => void handleStartSpaceflowRun('full')}
                    disabled={!canStartTextureExperiment}
                    title={spaceflowTextureMode === 'text'
                      ? 'Run structure and texture comparison variants'
                      : 'Full experiment requires text texture guidance'}
                  >
                    Full exp
                  </button>
                  {spaceflowRunActive && (
                    <button
                      className="btn-generate-go btn-spaceflow-stop"
                      type="button"
                      onClick={() => void handleStopSpaceflowRun()}
                      disabled={spaceflowRun.status === 'cancelling'}
                      title="Stop the running SpaceFlow job"
                    >
                      {spaceflowRun.status === 'cancelling' ? 'Stopping' : 'Stop'}
                    </button>
                  )}
                  {spaceflowRun?.run_id && (
                    <span className={`spaceflow-run-state ${spaceflowRun.status}`}>
                      {spaceflowRun.status}
                    </span>
                  )}
                </div>

                {!PUBLIC_DEMO && spaceflowRun?.output_dir && (
                  <div className="spaceflow-output-path">
                    <span>Output directory</span>
                    <code className="spaceflow-path-block" title={spaceflowRun.output_dir}>
                      {spaceflowRun.output_dir}
                    </code>
                  </div>
                )}

                {spaceflowRunWarnings.length > 0 && (
                  <div className="spaceflow-run-warnings" role="status">
                    <span>Warnings</span>
                    {spaceflowRunWarnings.map((message, index) => (
                      <p key={`${index}-${message}`}>{message}</p>
                    ))}
                  </div>
                )}

                {spaceflowVisibleOutputs.length > 0 && (
                  <div className="spaceflow-panel-block spaceflow-results-block">
                    <div className="spaceflow-panel-head">
                      <span className="spaceflow-panel-title">Generated files</span>
                      <div className="spaceflow-head-actions">
                        {spaceflowRun?.pipeline_stage === 'structure_only' ? (
                          <span className="spaceflow-stage-pill">structure stage</span>
                        ) : spaceflowRun?.pipeline_stage === 'full_pipeline' ? (
                          <span className="spaceflow-stage-pill">full pipeline</span>
                        ) : null}
                        {spaceflowRun && spaceflowInspectionMesh && (
                          <button
                            type="button"
                            className="spaceflow-panel-action"
                            onClick={() => {
                              const inspectedFile = inspectSpaceflowRunMesh(spaceflowRun);
                              showToast(
                                inspectedFile
                                  ? `Loaded ${outputFileLabel(inspectedFile)}`
                                  : 'No GLB mesh found in this run.',
                                5000,
                              );
                            }}
                          >
                            Inspect mesh
                          </button>
                        )}
                      </div>
                    </div>
                    {spaceflowPreviewImage && (
                      <a
                        className="spaceflow-output-preview"
                        href={resolveSpaceflowUrl(spaceflowPreviewImage.url)}
                        target="_blank"
                        rel="noreferrer"
                        title={PUBLIC_DEMO ? outputFileLabel(spaceflowPreviewImage) : spaceflowPreviewImage.path}
                      >
                        <img src={resolveSpaceflowUrl(spaceflowPreviewImage.url)} alt="SpaceFlow preview" />
                      </a>
                    )}
                    <div className="spaceflow-output-list">
                      {spaceflowVisibleOutputs.map(file => (
                        <a
                          key={file.relative_path}
                          className={`spaceflow-output-item ${file.kind}`}
                          href={resolveSpaceflowUrl(file.url)}
                          target="_blank"
                          rel="noreferrer"
                          title={PUBLIC_DEMO ? outputFileLabel(file) : file.path}
                        >
                          <span>{outputFileLabel(file)}</span>
                          <small>{file.kind} {formatFileSize(file.size)}</small>
                        </a>
                      ))}
                    </div>
                  </div>
                )}

                {spaceflowLogTail && (
                  <div className="spaceflow-panel-block spaceflow-log-block">
                    <div className="spaceflow-panel-head">
                      <span className="spaceflow-panel-title">Run log</span>
                    </div>
                    <pre className="spaceflow-log-tail">{spaceflowLogTail}</pre>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="top-right">
        {hasWarnings && (
          <span className="validation-warn" title="Some rotation matrices are not orthogonal">
            <AlertTriangleIcon size={17} />
          </span>
        )}
        <button
          type="button"
          className={`theme-toggle ${themeMode}`}
          role="switch"
          aria-checked={themeMode === 'dark'}
          aria-label={`Switch to ${themeMode === 'dark' ? 'light' : 'dark'} mode`}
          title={`Switch to ${themeMode === 'dark' ? 'light' : 'dark'} mode`}
          onClick={() => onThemeModeChange(themeMode === 'dark' ? 'light' : 'dark')}
        >
          <span className="theme-toggle-thumb" aria-hidden />
          <span className={`theme-toggle-icon ${themeMode === 'light' ? 'active' : ''}`} aria-hidden>
            <SunIcon size={14} strokeWidth={2.2} />
          </span>
          <span className={`theme-toggle-icon ${themeMode === 'dark' ? 'active' : ''}`} aria-hidden>
            <MoonIcon size={14} strokeWidth={2.2} />
          </span>
        </button>
        <div className="export-dropdown">
          <button
            type="button"
            className="btn-accent"
            onClick={() => {
              setShowImport(!showImport);
              setShowExport(false);
              setShowRotateAll(false);
              setShowSpaceflow(false);
            }}
            title="Import presets or open NPZ files"
          >
            <UploadIcon size={14} />
            Import
          </button>
          {showImport && (
            <div className={`dropdown-menu${PUBLIC_DEMO ? ' demo-preset-menu' : ''}`}>
              {PUBLIC_DEMO && (
                <>
                  <div className="demo-preset-grid">
                    {DEMO_PRESETS.map(preset => (
                      <button
                        key={preset.id}
                        type="button"
                        className="demo-preset-item"
                        onClick={() => void handleLoadDemoPreset(preset)}
                        title={`Load ${preset.label}`}
                      >
                        <img src={preset.imageUrl} alt="" className="demo-preset-thumb" />
                        <span className="demo-preset-label">{preset.label}</span>
                      </button>
                    ))}
                  </div>
                  <div className="dropdown-separator" />
                </>
              )}
              <button type="button" className="dropdown-item" onClick={handleImportJson}>
                Import JSON preset
              </button>
              {!PUBLIC_DEMO && (
                <button type="button" className="dropdown-item" onClick={handleOpenNpzPath}>
                  Open .npz path...
                </button>
              )}
              <button type="button" className="dropdown-item" onClick={handleImportNpz}>
                Import NPZ
              </button>
            </div>
          )}
        </div>
        <div className="export-dropdown">
          <button
            type="button"
            className="btn-accent"
            onClick={() => {
              setShowExport(!showExport);
              setShowImport(false);
              setShowRotateAll(false);
              setShowSpaceflow(false);
            }}
            title="Export or copy presets"
          >
            <DownloadIcon size={14} />
            Export
          </button>
          {showExport && (
            <div className="dropdown-menu">
              <button
                type="button"
                  className="dropdown-item"
                  onClick={handleDownloadNpz}
                  disabled={visiblePrimitives.length === 0}
              >
                Download .npz
              </button>
              <button
                type="button"
                  className="dropdown-item"
                  onClick={handleDownloadRendering}
                  disabled={visiblePrimitives.length === 0}
              >
                Download rendering (.png)
              </button>
              <button
                type="button"
                className="dropdown-item"
                onClick={() => void handleDownloadGlbMesh()}
                disabled={!spaceflowDownloadGlb}
                title={
                  spaceflowDownloadGlb
                    ? `Download ${spaceflowDownloadGlb.label}`
                    : 'No generated GLB mesh is available yet'
                }
              >
                Download GLB mesh
              </button>
              <button
                  type="button"
                  className="dropdown-item"
                  onClick={() => void handleCopyJson()}
                  disabled={visiblePrimitives.length === 0}
              >
                Copy JSON preset
              </button>
            </div>
          )}
        </div>
      </div>

      {toast && <div className="app-toast" onClick={() => setToast(null)}>{toast}</div>}
    </div>
  );
}
