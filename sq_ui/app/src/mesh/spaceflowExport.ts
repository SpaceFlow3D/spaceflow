import type { Primitive } from '../state/store';
import { eulerToMatrix } from '../state/rotation';
import {
  clampLowControlBBoxMargin,
  DEFAULT_LOW_CONTROL_BBOX_MARGIN,
} from '../state/spaceflowConfig';
import { createSuperquadricMesh } from './superquadric';
import { exportNpz, type PrimitiveExport } from './npzExport';
import type { NpzSpaceflowMetadata } from './npzImport';

const BBOX_MIN_HALF_EXTENT = 1e-4;
const BBOX_RESOLUTION = 32;
const CUBE_ROTATION = eulerToMatrix([90, 0, 0]);

export interface SpaceflowBBox {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  halfExtents: [number, number, number];
}

export interface SpaceflowSqBundleData {
  all: PrimitiveExport[];
  highControl: PrimitiveExport[];
  lowControlBbox: PrimitiveExport;
  bbox: SpaceflowBBox;
  bboxMarginFraction: number;
  counts: {
    all: number;
    high: number;
    low: number;
  };
}

export interface SpaceflowSqBundleBlobs {
  all: Blob;
  highControl: Blob;
  lowControlBbox: Blob;
}

function deformForPrimitive(p: Primitive) {
  if (p.tapering === undefined && p.bending === undefined) return undefined;
  return {
    tapering: (p.tapering ?? [0, 0]) as [number, number],
    bending: (p.bending ?? [0, 0, 0, 0, 0, 0]) as [
      number,
      number,
      number,
      number,
      number,
      number,
    ],
  };
}

export function primitiveToExport(p: Primitive): PrimitiveExport {
  return {
    scales: p.scales,
    shapes: p.shapes,
    translation: p.translation,
    rotation: p.rotation,
    controlLevel: p.controlLevel,
    ...(p.tapering !== undefined ? { tapering: p.tapering } : {}),
    ...(p.bending !== undefined ? { bending: p.bending } : {}),
  };
}

function paddedBBoxFromLowPrimitives(
  lowPrimitives: Primitive[],
  marginFraction = DEFAULT_LOW_CONTROL_BBOX_MARGIN,
): SpaceflowBBox {
  if (lowPrimitives.length === 0) {
    throw new Error('Mark at least one primitive as low control before creating SpaceFlow inputs.');
  }

  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const p of lowPrimitives) {
    const { vertices } = createSuperquadricMesh(
      p.scales[0],
      p.scales[1],
      p.scales[2],
      p.shapes[0],
      p.shapes[1],
      p.rotation,
      p.translation,
      BBOX_RESOLUTION,
      deformForPrimitive(p),
    );
    for (let i = 0; i < vertices.length; i += 3) {
      min[0] = Math.min(min[0], vertices[i]);
      min[1] = Math.min(min[1], vertices[i + 1]);
      min[2] = Math.min(min[2], vertices[i + 2]);
      max[0] = Math.max(max[0], vertices[i]);
      max[1] = Math.max(max[1], vertices[i + 1]);
      max[2] = Math.max(max[2], vertices[i + 2]);
    }
  }

  const dx = max[0] - min[0];
  const dy = max[1] - min[1];
  const dz = max[2] - min[2];
  const diagonal = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const pad = diagonal * clampLowControlBBoxMargin(marginFraction);
  const paddedMin: [number, number, number] = [min[0] - pad, min[1] - pad, min[2] - pad];
  const paddedMax: [number, number, number] = [max[0] + pad, max[1] + pad, max[2] + pad];
  const center: [number, number, number] = [
    (paddedMin[0] + paddedMax[0]) / 2,
    (paddedMin[1] + paddedMax[1]) / 2,
    (paddedMin[2] + paddedMax[2]) / 2,
  ];
  const halfExtents: [number, number, number] = [
    Math.max((paddedMax[0] - paddedMin[0]) / 2, BBOX_MIN_HALF_EXTENT),
    Math.max((paddedMax[1] - paddedMin[1]) / 2, BBOX_MIN_HALF_EXTENT),
    Math.max((paddedMax[2] - paddedMin[2]) / 2, BBOX_MIN_HALF_EXTENT),
  ];
  return { min: paddedMin, max: paddedMax, center, halfExtents };
}

export function buildLowControlBoundingBoxPrimitive(
  primitives: Primitive[],
  marginFraction = DEFAULT_LOW_CONTROL_BBOX_MARGIN,
): {
  primitive: PrimitiveExport;
  bbox: SpaceflowBBox;
} {
  const bbox = paddedBBoxFromLowPrimitives(
    primitives.filter(p => p.visible && p.controlLevel === 'low'),
    marginFraction,
  );

  // Existing examples use a +90deg X cube rotation. Because that swaps local Y/Z
  // in world space, choose local scales that still realize the requested bbox.
  const [hx, hy, hz] = bbox.halfExtents;
  return {
    bbox,
    primitive: {
      scales: [hx, hz, hy],
      shapes: [0.05, 0.05],
      translation: bbox.center,
      rotation: CUBE_ROTATION,
      controlLevel: 'low',
    },
  };
}

export function buildSpaceflowSqBundleData(
  primitives: Primitive[],
  options: { lowControlBBoxMargin?: number } = {},
): SpaceflowSqBundleData {
  const visiblePrimitives = primitives.filter(p => p.visible);
  if (visiblePrimitives.length === 0) throw new Error('No visible primitives to save.');
  const bboxMarginFraction = clampLowControlBBoxMargin(
    options.lowControlBBoxMargin ?? DEFAULT_LOW_CONTROL_BBOX_MARGIN,
  );
  const all = visiblePrimitives.map(primitiveToExport);
  const highControl = visiblePrimitives.filter(p => p.controlLevel === 'high').map(primitiveToExport);
  const { primitive: lowControlBbox, bbox } = buildLowControlBoundingBoxPrimitive(
    visiblePrimitives,
    bboxMarginFraction,
  );
  return {
    all,
    highControl,
    lowControlBbox,
    bbox,
    bboxMarginFraction,
    counts: {
      all: visiblePrimitives.length,
      high: highControl.length,
      low: visiblePrimitives.filter(p => p.controlLevel === 'low').length,
    },
  };
}

export async function buildSpaceflowSqBundleBlobs(
  data: SpaceflowSqBundleData,
  metadata?: NpzSpaceflowMetadata,
): Promise<SpaceflowSqBundleBlobs> {
  const [all, highControl, lowControlBbox] = await Promise.all([
    exportNpz(data.all, { metadata }),
    exportNpz(data.highControl, { allowEmpty: true }),
    exportNpz([data.lowControlBbox]),
  ]);
  return { all, highControl, lowControlBbox };
}
