import {
  RULESETS,
  defaultColorSettings,
  makeColorState,
  cloneColorState,
  clearColorState,
  clearCellColor,
  setCellColor,
  stepColor,
  wrapHue,
  hexToHsl,
  rgbToHsl,
  hslCss,
  hueToHex,
} from "./colorRules.js";

const PHI = (1 + Math.sqrt(5)) / 2;
const SQRT2 = Math.sqrt(2);
const SQRT3 = Math.sqrt(3);

const TRI_UP_OFFSETS = [
  [-1, -1], [-1, 0], [-1, 1],
  [0, -2], [0, -1], [0, 1], [0, 2],
  [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
];

const TRI_DN_OFFSETS = [
  [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
  [0, -2], [0, -1], [0, 1], [0, 2],
  [1, -1], [1, 0], [1, 1],
];

const TILING_TYPES = new Set(["rhombus", "penrose", "hex", "trihex", "oct", "voronoi"]);

const canvas = document.querySelector("#life-canvas");
const ctx = canvas.getContext("2d");

const elements = {
  body: document.body,
  generationValue: document.querySelector("#generation-value"),
  populationValue: document.querySelector("#population-value"),
  cellsValue: document.querySelector("#cells-value"),
  playToggle: document.querySelector("#play-toggle"),
  stepOnce: document.querySelector("#step-once"),
  stepBack: document.querySelector("#step-back"),
  resetGenZero: document.querySelector("#reset-gen-zero"),
  swapColors: document.querySelector("#swap-colors"),
  clearGrid: document.querySelector("#clear-grid"),
  randomizeGrid: document.querySelector("#randomize-grid"),
  gridType: document.querySelector("#grid-type"),
  wrapToggle: document.querySelector("#wrap-toggle"),
  rowsInput: document.querySelector("#rows-input"),
  colsInput: document.querySelector("#cols-input"),
  applySize: document.querySelector("#apply-size"),
  rebuildMap: document.querySelector("#rebuild-map"),
  speedInput: document.querySelector("#speed-input"),
  densityInput: document.querySelector("#density-input"),
  thresholdInput: document.querySelector("#threshold-input"),
  speedValue: document.querySelector("#speed-value"),
  densityValue: document.querySelector("#density-value"),
  thresholdValue: document.querySelector("#threshold-value"),
  imageInput: document.querySelector("#image-input"),
  invertGrid: document.querySelector("#invert-grid"),
  snapshotImage: document.querySelector("#snapshot-image"),
  sidebarHide: document.querySelector("#sidebar-hide"),
  sidebarShow: document.querySelector("#sidebar-show"),
  voronoiOptions: document.querySelector("#voronoi-options"),
  voronoiEuclid: document.querySelector("#voronoi-euclid"),
  voronoiManhattan: document.querySelector("#voronoi-manhattan"),
  voronoiJitter: document.querySelector("#voronoi-jitter"),
  voronoiJitterValue: document.querySelector("#voronoi-jitter-value"),
  toggleColorPanel: document.querySelector("#toggle-color-panel"),
  toggleToolsPanel: document.querySelector("#toggle-tools-panel"),
  colorOptions: document.querySelector("#color-options"),
  colorRule: document.querySelector("#color-rule"),
  colorLightness: document.querySelector("#color-lightness"),
  colorLightnessValue: document.querySelector("#color-lightness-value"),
  colorGoetheanOptions: document.querySelector("#color-goethean-options"),
  goetheanSatFull: document.querySelector("#goethean-sat-full"),
  goetheanSatInherit: document.querySelector("#goethean-sat-inherit"),
  colorRotationOptions: document.querySelector("#color-rotation-options"),
  rotationHueFixed: document.querySelector("#rotation-hue-fixed"),
  rotationHueParent: document.querySelector("#rotation-hue-parent"),
  rotationFixedHueField: document.querySelector("#rotation-fixed-hue-field"),
  rotationFixedHue: document.querySelector("#rotation-fixed-hue"),
  rotationDelta: document.querySelector("#rotation-delta"),
  toolsOptions: document.querySelector("#tools-options"),
  toolPencil: document.querySelector("#tool-pencil"),
  toolEraser: document.querySelector("#tool-eraser"),
  toolSelect: document.querySelector("#tool-select"),
  toolCopy: document.querySelector("#tool-copy"),
  toolCut: document.querySelector("#tool-cut"),
  toolPaste: document.querySelector("#tool-paste"),
};

const state = {
  gridType: "penrose",
  rows: 64,
  cols: 64,
  wrap: true,
  speed: 8,
  density: 0.28,
  threshold: 140,
  running: false,
  generation: 0,
  grid: [],
  polygons: [],
  tilingStates: [],
  tilingNeighbors: [],
  tilingFaceTypes: [],
  tilingBBox: [1, 1],
  voronoiSeed: 42,
  voronoiMetric: "euclidean",
  voronoiJitter: 0.7,
  sidebarOpen: window.innerWidth > 720,
  colorsSwapped: false,
  history: [],
  gen0Snapshot: null,
  // Color rules
  colorSettings: defaultColorSettings(),
  color: makeColorState(0),
  // Editor tools
  tool: "pencil", // 'pencil' | 'eraser' | 'select'
  selection: null, // Set<flatIdx> | null
  selectionDrag: null, // in-progress rectangle in screen coords
  clipboard: null, // see editor-tools commit
  pasteGhost: null, // {dx,dy} during move/paste preview
  // Submenu visibility (UI)
  colorPanelOpen: false,
  toolsPanelOpen: false,
};
const HISTORY_LIMIT = 200;

const pointerState = {
  painting: false,
  drawValue: 1,
};

let animationFrame = 0;
let lastFrameTime = 0;
let accumulator = 0;

function isTiling() {
  return TILING_TYPES.has(state.gridType);
}

function titleCase(value) {
  return value[0].toUpperCase() + value.slice(1);
}

function makeGrid(rows, cols, fill = 0) {
  return Array.from({ length: rows }, () => Array(cols).fill(fill));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function pointInTriangle(px, py, ax, ay, bx, by, cx, cy) {
  const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
  const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
  const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
  const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(hasNeg && hasPos);
}

function pointInPolygon(px, py, poly) {
  let inside = false;
  let j = poly.length - 1;
  for (let i = 0; i < poly.length; i += 1) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
    j = i;
  }
  return inside;
}

function normalizeTiling(polys) {
  const xs = polys.flatMap((poly) => poly.map(([x]) => x));
  const ys = polys.flatMap((poly) => poly.map(([, y]) => y));
  if (!xs.length) {
    return [polys, [1, 1]];
  }
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const shifted = polys.map((poly) => poly.map(([x, y]) => [x - minX, y - minY]));
  const shiftedXs = shifted.flatMap((poly) => poly.map(([x]) => x));
  const shiftedYs = shifted.flatMap((poly) => poly.map(([, y]) => y));
  return [shifted, [Math.max(...shiftedXs), Math.max(...shiftedYs)]];
}

function computeTilingNeighbors(polys, tolerance = 1e-6) {
  const vertexMap = new Map();
  polys.forEach((poly, index) => {
    poly.forEach(([x, y]) => {
      const rx = Number((Math.round(x / tolerance) * tolerance).toFixed(5));
      const ry = Number((Math.round(y / tolerance) * tolerance).toFixed(5));
      const key = `${rx},${ry}`;
      if (!vertexMap.has(key)) {
        vertexMap.set(key, new Set());
      }
      vertexMap.get(key).add(index);
    });
  });

  return polys.map((poly, index) => {
    const neighbors = new Set();
    poly.forEach(([x, y]) => {
      const rx = Number((Math.round(x / tolerance) * tolerance).toFixed(5));
      const ry = Number((Math.round(y / tolerance) * tolerance).toFixed(5));
      const key = `${rx},${ry}`;
      (vertexMap.get(key) || []).forEach((neighbor) => {
        if (neighbor !== index) {
          neighbors.add(neighbor);
        }
      });
    });
    return [...neighbors];
  });
}

function stepGrid(grid, wrap) {
  const rows = grid.length;
  const cols = grid[0].length;
  const next = makeGrid(rows, cols);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      let total = 0;
      for (let dr = -1; dr <= 1; dr += 1) {
        for (let dc = -1; dc <= 1; dc += 1) {
          if (dr === 0 && dc === 0) {
            continue;
          }
          let nr = r + dr;
          let nc = c + dc;
          if (wrap) {
            nr = (nr + rows) % rows;
            nc = (nc + cols) % cols;
          } else if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
            continue;
          }
          total += grid[nr][nc];
        }
      }
      next[r][c] = grid[r][c] ? Number(total === 2 || total === 3) : Number(total === 3);
    }
  }
  return next;
}

function stepGridTri(grid, wrap) {
  const rows = grid.length;
  const cols = grid[0].length;
  const next = makeGrid(rows, cols);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const offsets = (r + c) % 2 === 0 ? TRI_UP_OFFSETS : TRI_DN_OFFSETS;
      let total = 0;
      offsets.forEach(([dr, dc]) => {
        let nr = r + dr;
        let nc = c + dc;
        if (wrap) {
          nr = (nr + rows) % rows;
          nc = (nc + cols) % cols;
        } else if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
          return;
        }
        total += grid[nr][nc];
      });
      next[r][c] = grid[r][c] ? Number(total === 2 || total === 3) : Number(total === 3);
    }
  }
  return next;
}

function stepTiling(states, neighbors) {
  return states.map((alive, index) => {
    const total = neighbors[index].reduce((sum, neighbor) => sum + states[neighbor], 0);
    return alive ? Number(total === 2 || total === 3) : Number(total === 3);
  });
}

function generateRhombicTiling(cubeRows, cubeCols) {
  const polys = [];
  const faceTypes = [];
  const dxCol = SQRT3;
  for (let row = 0; row < cubeRows; row += 1) {
    for (let col = 0; col < cubeCols; col += 1) {
      const cx = col * dxCol + (row % 2 ? 0.5 * dxCol : 0);
      const cy = row * 1.5;
      const h = SQRT3 / 2;
      polys.push([[cx, cy - 1], [cx + h, cy - 0.5], [cx, cy], [cx - h, cy - 0.5]]);
      faceTypes.push(0);
      polys.push([[cx - h, cy - 0.5], [cx, cy], [cx, cy + 1], [cx - h, cy + 0.5]]);
      faceTypes.push(1);
      polys.push([[cx + h, cy - 0.5], [cx, cy], [cx, cy + 1], [cx + h, cy + 0.5]]);
      faceTypes.push(2);
    }
  }
  const [normalized, bbox] = normalizeTiling(polys);
  return [normalized, faceTypes, bbox];
}

function polar(radius, angle) {
  return [radius * Math.cos(angle), radius * Math.sin(angle)];
}

function add([ax, ay], [bx, by]) {
  return [ax + bx, ay + by];
}

function sub([ax, ay], [bx, by]) {
  return [ax - bx, ay - by];
}

function scale([x, y], factor) {
  return [x * factor, y * factor];
}

function edgeKey([x1, y1], [x2, y2]) {
  const a = `${x1.toFixed(8)},${y1.toFixed(8)}`;
  const b = `${x2.toFixed(8)},${y2.toFixed(8)}`;
  return [a, b].sort().join("|");
}

function orderPolygon(poly) {
  const cx = poly.reduce((sum, [x]) => sum + x, 0) / poly.length;
  const cy = poly.reduce((sum, [, y]) => sum + y, 0) / poly.length;
  return [...poly].sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
}

function generatePenroseTiling(subdivisions = 5) {
  let triangles = [];
  for (let i = 0; i < 10; i += 1) {
    let b = polar(1, ((2 * i) - 1) * Math.PI / 10);
    let c = polar(1, ((2 * i) + 1) * Math.PI / 10);
    if (i % 2 === 0) {
      [b, c] = [c, b];
    }
    triangles.push([0, [0, 0], b, c]);
  }

  for (let step = 0; step < subdivisions; step += 1) {
    const next = [];
    triangles.forEach(([color, a, b, c]) => {
      if (color === 0) {
        const p = add(a, scale(sub(b, a), 1 / PHI));
        next.push([0, c, p, b], [1, p, c, a]);
      } else {
        const q = add(b, scale(sub(a, b), 1 / PHI));
        const r = add(b, scale(sub(c, b), 1 / PHI));
        next.push([1, r, c, a], [1, q, r, b], [0, r, q, a]);
      }
    });
    triangles = next;
  }

  const groups = new Map();
  triangles.forEach((tri) => {
    const key = edgeKey(tri[2], tri[3]);
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(tri);
  });

  const polys = [];
  const faceTypes = [];
  groups.forEach((shared) => {
    if (shared.length !== 2 || shared[0][0] !== shared[1][0]) {
      return;
    }
    const [color, a, b, c] = shared[0];
    const [, a2] = shared[1];
    polys.push(orderPolygon([a, b, a2, c]));
    faceTypes.push(color === 1 ? 0 : 1);
  });

  const [normalized, bbox] = normalizeTiling(polys);
  return [normalized, faceTypes, bbox];
}

function generateHexTiling(rows, cols) {
  const polys = [];
  const faceTypes = [];
  const s = 1;
  const h = SQRT3 * s / 2;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const cx = col * SQRT3 * s + (row % 2 ? h : 0);
      const cy = row * 1.5 * s;
      polys.push([
        [cx + h, cy + s / 2],
        [cx, cy + s],
        [cx - h, cy + s / 2],
        [cx - h, cy - s / 2],
        [cx, cy - s],
        [cx + h, cy - s / 2],
      ]);
      faceTypes.push(0);
    }
  }
  const [normalized, bbox] = normalizeTiling(polys);
  return [normalized, faceTypes, bbox];
}

function generateTrihexTiling(rows, cols) {
  const polys = [];
  const faceTypes = [];
  const centers = [];
  const s = 1;
  const h = SQRT3 * s / 2;
  let hexCount = 0;

  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      const cx = (i + j) * SQRT3 * s;
      const cy = (i - j) * s;
      centers.push([cx, cy]);
      polys.push([
        [cx + h, cy + s / 2],
        [cx, cy + s],
        [cx - h, cy + s / 2],
        [cx - h, cy - s / 2],
        [cx, cy - s],
        [cx + h, cy - s / 2],
      ]);
      faceTypes.push(0);
      hexCount += 1;
    }
  }

  const addedTriangles = new Set();
  for (let index = 0; index < hexCount; index += 1) {
    const hexVerts = polys[index];
    const [cx, cy] = centers[index];
    for (let k = 0; k < 6; k += 1) {
      const v1 = hexVerts[k];
      const v2 = hexVerts[(k + 1) % 6];
      const mx = (v1[0] + v2[0]) / 2;
      const my = (v1[1] + v2[1]) / 2;
      const dx = mx - cx;
      const dy = my - cy;
      const tri = [v1, v2, [mx + dx, my + dy]];
      const key = tri
        .map(([x, y]) => `${x.toFixed(4)},${y.toFixed(4)}`)
        .sort()
        .join("|");
      if (addedTriangles.has(key)) {
        continue;
      }
      addedTriangles.add(key);
      polys.push(tri);
      faceTypes.push(1);
    }
  }

  const [normalized, bbox] = normalizeTiling(polys);
  return [normalized, faceTypes, bbox];
}

function generateOctTiling(rows, cols) {
  const polys = [];
  const faceTypes = [];
  const s = 1;
  const k = s / 2;
  const big = k + s / Math.sqrt(2);
  const w = s * (1 + Math.sqrt(2));
  const d = s / Math.sqrt(2);

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const cx = col * w;
      const cy = row * w;
      polys.push([
        [cx - k, cy + big], [cx + k, cy + big],
        [cx + big, cy + k], [cx + big, cy - k],
        [cx + k, cy - big], [cx - k, cy - big],
        [cx - big, cy - k], [cx - big, cy + k],
      ]);
      faceTypes.push(0);
      if (row < rows - 1 && col < cols - 1) {
        const sx = cx + w / 2;
        const sy = cy + w / 2;
        polys.push([
          [sx, sy - d], [sx + d, sy], [sx, sy + d], [sx - d, sy],
        ]);
        faceTypes.push(1);
      }
    }
  }

  const [normalized, bbox] = normalizeTiling(polys);
  return [normalized, faceTypes, bbox];
}

function clipHalfplane(poly, a, b, c) {
  if (!poly.length) {
    return [];
  }
  const output = [];
  for (let i = 0; i < poly.length; i += 1) {
    const current = poly[i];
    const previous = poly[(i - 1 + poly.length) % poly.length];
    const dCurrent = a * current[0] + b * current[1] - c;
    const dPrevious = a * previous[0] + b * previous[1] - c;
    const currentIn = dCurrent <= 0;
    const previousIn = dPrevious <= 0;
    if (currentIn) {
      if (!previousIn) {
        const t = dPrevious / (dPrevious - dCurrent);
        output.push([
          previous[0] + t * (current[0] - previous[0]),
          previous[1] + t * (current[1] - previous[1]),
        ]);
      }
      output.push(current);
    } else if (previousIn) {
      const t = dPrevious / (dPrevious - dCurrent);
      output.push([
        previous[0] + t * (current[0] - previous[0]),
        previous[1] + t * (current[1] - previous[1]),
      ]);
    }
  }
  return output;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function generateVoronoiTiling(rows, cols, seed = 42, metric = "euclidean", jitter = 0.7) {
  const rng = mulberry32(seed);
  const seeds = [];
  const j = Math.max(0, Math.min(1, jitter));
  const margin = (1 - j) / 2;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      seeds.push([col + margin + j * rng(), row + margin + j * rng()]);
    }
  }
  const box = [-0.3, -0.3, cols - 0.7, rows - 0.7];

  if (metric === "manhattan") {
    return generateManhattanVoronoi(seeds, box, rows, cols);
  }

  const polys = [];
  const faceTypes = [];
  for (let i = 0; i < seeds.length; i += 1) {
    const [sx, sy] = seeds[i];
    let cell = [
      [box[0], box[1]], [box[2], box[1]],
      [box[2], box[3]], [box[0], box[3]],
    ];
    for (let k = 0; k < seeds.length; k += 1) {
      if (i === k) {
        continue;
      }
      const [tx, ty] = seeds[k];
      if ((tx - sx) ** 2 + (ty - sy) ** 2 > 25) {
        continue;
      }
      const a = 2 * (tx - sx);
      const b = 2 * (ty - sy);
      const c = (tx * tx) + (ty * ty) - (sx * sx) - (sy * sy);
      cell = clipHalfplane(cell, a, b, c);
      if (!cell.length) {
        break;
      }
    }
    if (cell.length >= 3) {
      polys.push(cell);
      const dist = Math.hypot(sx - cols / 2, sy - rows / 2);
      faceTypes.push(Math.floor(dist) % 3);
    }
  }

  const [normalized, bbox] = normalizeTiling(polys);
  return [normalized, faceTypes, bbox];
}

function generateManhattanVoronoi(seeds, box, rows, cols) {
  // Rasterize the diagram on a fine grid and trace each cell's outline.
  // Manhattan bisectors are piecewise-linear (axis-aligned + 45°) so
  // half-plane clipping doesn't apply directly. We raster-sample then
  // simplify the traced staircase back into its underlying straight lines.
  const [x0, y0, x1, y1] = box;
  const W = x1 - x0;
  const H = y1 - y0;
  // Budget total work ~80M pixel-seed comparisons. Scale raster resolution
  // with seed count: more seeds -> lower per-axis resolution per seed.
  const totalBudget = 80_000_000;
  const maxSideBySeeds = Math.floor(Math.sqrt(totalBudget / Math.max(1, seeds.length)));
  const maxSide = Math.max(120, Math.min(640, maxSideBySeeds));
  const perUnit = Math.max(6, Math.floor(maxSide / Math.max(W, H)));
  const pw = Math.min(maxSide, Math.max(16, Math.ceil(W * perUnit)));
  const ph = Math.min(maxSide, Math.max(16, Math.ceil(H * perUnit)));
  const cw = W / pw;
  const ch = H / ph;
  const grid = new Int32Array(pw * ph);
  // Each pixel: find nearest seed under Manhattan distance.
  for (let py = 0; py < ph; py += 1) {
    const y = y0 + (py + 0.5) * ch;
    for (let px = 0; px < pw; px += 1) {
      const x = x0 + (px + 0.5) * cw;
      let bestI = 0;
      let bestD = Infinity;
      for (let i = 0; i < seeds.length; i += 1) {
        const d = Math.abs(x - seeds[i][0]) + Math.abs(y - seeds[i][1]);
        if (d < bestD) {
          bestD = d;
          bestI = i;
        }
      }
      grid[py * pw + px] = bestI;
    }
  }

  // Simplification tolerance: just over half a pixel diagonal so staircase
  // approximations of a true 45° line collapse into one segment, but real
  // axis-aligned or diagonal bisector segments (length >= cell spacing)
  // stay intact.
  const eps = Math.max(cw, ch) * 0.75;

  const polys = [];
  const faceTypes = [];
  for (let i = 0; i < seeds.length; i += 1) {
    const raw = traceRegionOutline(grid, pw, ph, i, x0, y0, cw, ch);
    const poly = raw ? simplifyClosedPolygon(raw, eps) : null;
    if (poly && poly.length >= 3) {
      polys.push(poly);
      const [sx, sy] = seeds[i];
      const dist = Math.abs(sx - cols / 2) + Math.abs(sy - rows / 2);
      faceTypes.push(Math.floor(dist) % 3);
    }
  }

  const [normalized, bbox] = normalizeTiling(polys);
  return [normalized, faceTypes, bbox];
}

function traceRegionOutline(grid, pw, ph, seedIdx, x0, y0, cw, ch) {
  // For each pixel in the region, emit boundary edges on sides where the
  // neighbor is out of bounds or a different region. Edges are oriented CW
  // around their pixel (in screen-space, y-down), so the region's outer
  // boundary forms a CW loop with interior on the right.
  const edges = [];
  const keyOf = (x, y) => `${Math.round(x * 10000)}|${Math.round(y * 10000)}`;
  for (let py = 0; py < ph; py += 1) {
    for (let px = 0; px < pw; px += 1) {
      if (grid[py * pw + px] !== seedIdx) {
        continue;
      }
      const x = x0 + px * cw;
      const y = y0 + py * ch;
      const xr = x + cw;
      const yb = y + ch;
      if (py === 0 || grid[(py - 1) * pw + px] !== seedIdx) {
        edges.push([x, y, xr, y]);
      }
      if (px === pw - 1 || grid[py * pw + (px + 1)] !== seedIdx) {
        edges.push([xr, y, xr, yb]);
      }
      if (py === ph - 1 || grid[(py + 1) * pw + px] !== seedIdx) {
        edges.push([xr, yb, x, yb]);
      }
      if (px === 0 || grid[py * pw + (px - 1)] !== seedIdx) {
        edges.push([x, yb, x, y]);
      }
    }
  }
  if (edges.length === 0) {
    return null;
  }
  // Multimap from each edge's start corner to the list of edges starting there.
  // At diagonal-touch pinch points, two edges can share a start corner.
  const byStart = new Map();
  for (const e of edges) {
    const k = keyOf(e[0], e[1]);
    let arr = byStart.get(k);
    if (!arr) {
      arr = [];
      byStart.set(k, arr);
    }
    arr.push(e);
  }
  const takeEdge = (key, prevDX, prevDY) => {
    const arr = byStart.get(key);
    if (!arr || arr.length === 0) {
      return null;
    }
    // At a fork, prefer the right-most (clockwise-most) turn to stay on the
    // outer boundary of the current simply-connected piece. With y-down coords
    // a right turn corresponds to cross > 0.
    let bestIdx = 0;
    if (arr.length > 1 && prevDX !== undefined) {
      let bestScore = -Infinity;
      for (let i = 0; i < arr.length; i += 1) {
        const e = arr[i];
        const dx = e[2] - e[0];
        const dy = e[3] - e[1];
        const cross = prevDX * dy - prevDY * dx;
        const dot = prevDX * dx + prevDY * dy;
        let score;
        if (cross > 1e-9) {
          score = 3; // right turn
        } else if (cross < -1e-9) {
          score = 0; // left turn
        } else if (dot > 0) {
          score = 2; // straight
        } else {
          score = 1; // U-turn
        }
        if (score > bestScore) {
          bestScore = score;
          bestIdx = i;
        }
      }
    }
    const picked = arr.splice(bestIdx, 1)[0];
    if (arr.length === 0) {
      byStart.delete(key);
    }
    return picked;
  };

  // Walk the outer loop starting from the top-left-most edge corner. That
  // guarantees we begin on the outer boundary rather than an inner hole.
  let startIdx = 0;
  for (let i = 1; i < edges.length; i += 1) {
    const e = edges[i];
    const s = edges[startIdx];
    if (e[1] < s[1] - 1e-9 || (Math.abs(e[1] - s[1]) < 1e-9 && e[0] < s[0])) {
      startIdx = i;
    }
  }
  const start = edges[startIdx];
  // Remove start from map
  takeEdge(keyOf(start[0], start[1]));
  const loop = [[start[0], start[1]]];
  let cur = start;
  for (let iter = 0; iter < edges.length + 4; iter += 1) {
    const prevDX = cur[2] - cur[0];
    const prevDY = cur[3] - cur[1];
    const endKey = keyOf(cur[2], cur[3]);
    const startKey = keyOf(start[0], start[1]);
    if (endKey === startKey) {
      // Closed loop back to start corner
      break;
    }
    const next = takeEdge(endKey, prevDX, prevDY);
    if (!next) {
      loop.push([cur[2], cur[3]]);
      break;
    }
    loop.push([next[0], next[1]]);
    cur = next;
  }
  return simplifyPolygon(loop);
}

function simplifyPolygon(poly) {
  if (poly.length < 3) {
    return poly;
  }
  const out = [];
  const n = poly.length;
  const EPS = 1e-7;
  for (let i = 0; i < n; i += 1) {
    const prev = poly[(i - 1 + n) % n];
    const cur = poly[i];
    const next = poly[(i + 1) % n];
    // Drop consecutive duplicates
    if (Math.abs(cur[0] - prev[0]) < EPS && Math.abs(cur[1] - prev[1]) < EPS) {
      continue;
    }
    // Drop colinear midpoint
    const cross = (cur[0] - prev[0]) * (next[1] - prev[1]) - (cur[1] - prev[1]) * (next[0] - prev[0]);
    if (Math.abs(cross) < EPS) {
      continue;
    }
    out.push(cur);
  }
  return out.length >= 3 ? out : poly;
}

// Ramer-Douglas-Peucker on an open chain. Returns kept points in order.
function rdpOpen(points, epsSq) {
  if (points.length < 3) {
    return points.slice();
  }
  const keep = new Array(points.length).fill(false);
  keep[0] = true;
  keep[points.length - 1] = true;
  const stack = [[0, points.length - 1]];
  while (stack.length) {
    const [lo, hi] = stack.pop();
    if (hi - lo < 2) continue;
    const a = points[lo];
    const b = points[hi];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const denom = dx * dx + dy * dy;
    let maxD = 0;
    let idx = -1;
    for (let i = lo + 1; i < hi; i += 1) {
      const ex = points[i][0] - a[0];
      const ey = points[i][1] - a[1];
      let d;
      if (denom === 0) {
        d = ex * ex + ey * ey;
      } else {
        const num = ex * dy - ey * dx;
        d = (num * num) / denom;
      }
      if (d > maxD) {
        maxD = d;
        idx = i;
      }
    }
    if (idx >= 0 && maxD > epsSq) {
      keep[idx] = true;
      stack.push([lo, idx]);
      stack.push([idx, hi]);
    }
  }
  const result = [];
  for (let i = 0; i < points.length; i += 1) {
    if (keep[i]) result.push(points[i]);
  }
  return result;
}

// RDP on a closed polygon. Splits the loop into two open chains so RDP
// has stable anchor endpoints, then stitches the two simplified halves.
function simplifyClosedPolygon(points, eps) {
  if (!points || points.length < 4) {
    return points ? points.slice() : [];
  }
  const epsSq = eps * eps;
  const n = points.length;
  const mid = Math.floor(n / 2);
  const chain1 = points.slice(0, mid + 1);
  const chain2 = points.slice(mid).concat([points[0]]);
  const s1 = rdpOpen(chain1, epsSq);
  const s2 = rdpOpen(chain2, epsSq);
  // s1 ends at points[mid], s2 starts at points[mid] and ends at points[0];
  // drop the duplicated join points.
  const merged = s1.slice(0, -1).concat(s2.slice(0, -1));
  // Final pass: drop colinear/duplicate artifacts at the seams.
  return simplifyPolygon(merged);
}

function rebuildTopology() {
  state.generation = 0;
  state.running = false;
  elements.playToggle.textContent = "Play";
  if (isTiling()) {
    let polys;
    let faceTypes;
    let bbox;
    if (state.gridType === "rhombus") {
      [polys, faceTypes, bbox] = generateRhombicTiling(
        Math.max(2, Math.floor(state.rows / 4)),
        Math.max(2, Math.floor(state.cols / 4)),
      );
    } else if (state.gridType === "penrose") {
      const dim = Math.max(state.rows, state.cols);
      const subdivisions = dim <= 32 ? 4 : dim <= 64 ? 5 : dim <= 128 ? 6 : 7;
      [polys, faceTypes, bbox] = generatePenroseTiling(subdivisions);
    } else if (state.gridType === "hex") {
      [polys, faceTypes, bbox] = generateHexTiling(
        Math.max(4, Math.floor(state.rows / 2)),
        Math.max(4, Math.floor(state.cols / 2)),
      );
    } else if (state.gridType === "trihex") {
      [polys, faceTypes, bbox] = generateTrihexTiling(
        Math.max(3, Math.floor(state.rows / 6)),
        Math.max(3, Math.floor(state.cols / 6)),
      );
    } else if (state.gridType === "oct") {
      [polys, faceTypes, bbox] = generateOctTiling(
        Math.max(3, Math.floor(state.rows / 4)),
        Math.max(3, Math.floor(state.cols / 4)),
      );
    } else {
      [polys, faceTypes, bbox] = generateVoronoiTiling(
        Math.max(3, Math.floor(state.rows / 4)),
        Math.max(3, Math.floor(state.cols / 4)),
        state.voronoiSeed,
        state.voronoiMetric,
        state.voronoiJitter,
      );
    }
    state.polygons = polys;
    state.tilingFaceTypes = faceTypes;
    state.tilingBBox = bbox;
    state.tilingNeighbors = computeTilingNeighbors(polys);
    state.tilingStates = Array(polys.length).fill(0);
  } else {
    state.grid = makeGrid(state.rows, state.cols, 0);
  }
  // Topology changed: previous snapshots would have stale cell counts.
  state.history = [];
  state.gen0Snapshot = null;
  state.selection = null;
  resizeColorState();
  syncLabels();
}

function snapshotCurrent() {
  const base = isTiling()
    ? { type: "tiling", data: state.tilingStates.slice() }
    : { type: "grid", data: state.grid.map((row) => row.slice()) };
  if (colorRuleActive()) {
    base.color = cloneColorState(state.color);
  }
  return base;
}

function restoreSnapshot(snap) {
  if (!snap) {
    return false;
  }
  let ok = false;
  if (snap.type === "tiling" && isTiling() && snap.data.length === state.tilingStates.length) {
    state.tilingStates = snap.data.slice();
    ok = true;
  } else if (snap.type === "grid" && !isTiling()) {
    state.grid = snap.data.map((row) => row.slice());
    ok = true;
  }
  if (ok && snap.color && snap.color.hue.length === state.color.hue.length) {
    state.color = cloneColorState(snap.color);
  } else if (ok && colorRuleActive()) {
    // Snapshot predates color rule activation — re-seed alive cells.
    seedColorsForAllAlive();
  }
  return ok;
}

function captureGenZero() {
  state.gen0Snapshot = snapshotCurrent();
  state.history = [];
}

function randomizeState() {
  state.generation = 0;
  if (isTiling()) {
    state.tilingStates = state.tilingStates.map(() => (Math.random() < state.density ? 1 : 0));
  } else {
    state.grid = state.grid.map((row) => row.map(() => (Math.random() < state.density ? 1 : 0)));
  }
  if (colorRuleActive()) {
    seedColorsForAllAlive();
  }
  captureGenZero();
  syncLabels();
}

function clearState() {
  state.generation = 0;
  if (isTiling()) {
    state.tilingStates = state.tilingStates.map(() => 0);
  } else {
    state.grid = makeGrid(state.rows, state.cols, 0);
  }
  // Clearing wipes color state too (per spec: clear resets all cell color
  // state but not the ruleset choice).
  clearColorState(state.color);
  state.selection = null;
  captureGenZero();
  syncLabels();
}

function invertState() {
  if (isTiling()) {
    state.tilingStates = state.tilingStates.map((value) => 1 - value);
  } else {
    state.grid = state.grid.map((row) => row.map((value) => 1 - value));
  }
  if (colorRuleActive()) {
    seedColorsForAllAlive();
  }
  // Inverting resets the starting-point: the inverted state becomes gen 0.
  state.generation = 0;
  captureGenZero();
  syncLabels();
}

function swapColors() {
  state.colorsSwapped = !state.colorsSwapped;
  syncLabels();
}

function resetToGenZero() {
  if (!state.gen0Snapshot) {
    return;
  }
  if (restoreSnapshot(state.gen0Snapshot)) {
    state.generation = 0;
    state.history = [];
    syncLabels();
  }
}

function stepBack() {
  if (state.history.length === 0) {
    return;
  }
  const prev = state.history.pop();
  if (restoreSnapshot(prev)) {
    state.generation = Math.max(0, state.generation - 1);
    syncLabels();
  }
}

function population() {
  if (isTiling()) {
    return state.tilingStates.reduce((sum, value) => sum + value, 0);
  }
  return state.grid.reduce((sum, row) => sum + row.reduce((rowSum, value) => rowSum + value, 0), 0);
}

function cellCount() {
  return isTiling() ? state.tilingStates.length : state.rows * state.cols;
}

// ─── Flat-index helpers (for color rules + tools) ───────────────────────────
// Cells are addressed by a flat index regardless of geometry:
//   tiling: polygon index
//   grid:   r * cols + c

function flatIndexOf(target) {
  if (target == null) return -1;
  if (isTiling()) return target;
  const [r, c] = target;
  return r * state.cols + c;
}

function flatToTarget(idx) {
  if (isTiling()) return idx;
  const r = Math.floor(idx / state.cols);
  return [r, idx - r * state.cols];
}

function isAliveAt(idx) {
  if (isTiling()) return state.tilingStates[idx] === 1;
  const r = Math.floor(idx / state.cols);
  const c = idx - r * state.cols;
  return state.grid[r][c] === 1;
}

function setAliveAt(idx, value) {
  const v = value ? 1 : 0;
  if (isTiling()) {
    state.tilingStates[idx] = v;
  } else {
    const r = Math.floor(idx / state.cols);
    const c = idx - r * state.cols;
    state.grid[r][c] = v;
  }
}

function flatAlive() {
  const n = cellCount();
  const out = new Uint8Array(n);
  if (isTiling()) {
    for (let i = 0; i < n; i += 1) out[i] = state.tilingStates[i];
  } else {
    for (let r = 0; r < state.rows; r += 1) {
      const row = state.grid[r];
      const base = r * state.cols;
      for (let c = 0; c < state.cols; c += 1) {
        out[base + c] = row[c];
      }
    }
  }
  return out;
}

function applyFlatAlive(flat) {
  if (isTiling()) {
    state.tilingStates = Array.from(flat);
    return;
  }
  for (let r = 0; r < state.rows; r += 1) {
    const row = state.grid[r];
    const base = r * state.cols;
    for (let c = 0; c < state.cols; c += 1) {
      row[c] = flat[base + c];
    }
  }
}

function neighborsOfFn() {
  if (isTiling()) {
    return (i) => state.tilingNeighbors[i];
  }
  const rows = state.rows;
  const cols = state.cols;
  const wrap = state.wrap;
  if (state.gridType === "triangle") {
    return (i) => {
      const r = Math.floor(i / cols);
      const c = i - r * cols;
      const offsets = (r + c) % 2 === 0 ? TRI_UP_OFFSETS : TRI_DN_OFFSETS;
      const out = [];
      for (let k = 0; k < offsets.length; k += 1) {
        const [dr, dc] = offsets[k];
        let nr = r + dr;
        let nc = c + dc;
        if (wrap) {
          nr = ((nr % rows) + rows) % rows;
          nc = ((nc % cols) + cols) % cols;
        } else if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
          continue;
        }
        out.push(nr * cols + nc);
      }
      return out;
    };
  }
  return (i) => {
    const r = Math.floor(i / cols);
    const c = i - r * cols;
    const out = [];
    for (let dr = -1; dr <= 1; dr += 1) {
      for (let dc = -1; dc <= 1; dc += 1) {
        if (dr === 0 && dc === 0) continue;
        let nr = r + dr;
        let nc = c + dc;
        if (wrap) {
          nr = ((nr % rows) + rows) % rows;
          nc = ((nc % cols) + cols) % cols;
        } else if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
          continue;
        }
        out.push(nr * cols + nc);
      }
    }
    return out;
  };
}

// ─── Color state lifecycle ─────────────────────────────────────────────────

function colorRuleActive() {
  return state.colorSettings.rule !== RULESETS.NONE;
}

function resizeColorState() {
  state.color = makeColorState(cellCount());
}

// Seed color for a newly-alive cell at index i, given current settings.
// Used for clicks, randomize, image-without-pixel-hue, etc.
function seedColorAt(i, hue, saturation = 1.0) {
  setCellColor(state.color, i, hue, saturation);
}

function defaultSeedHue() {
  if (state.colorSettings.rule === RULESETS.ROTATION) {
    return state.colorSettings.rotationFixedHue;
  }
  // Goethean / none: random hue gives interesting starting mixtures.
  return Math.random() * 360;
}

// Re-seed color for every currently-alive cell that lacks color (sat=0).
// Called on toggling a ruleset on, after randomize, after invert, etc.
function seedColorsForAllAlive() {
  const n = cellCount();
  for (let i = 0; i < n; i += 1) {
    if (isAliveAt(i)) {
      if (state.color.sat[i] <= 0 || state.color.origSat[i] <= 0) {
        seedColorAt(i, defaultSeedHue(), 1.0);
      }
    } else {
      clearCellColor(state.color, i);
    }
  }
}

function stepOnce() {
  // Snapshot the pre-step state so Back can undo it.
  if (state.generation === 0 && !state.gen0Snapshot) {
    state.gen0Snapshot = snapshotCurrent();
  }
  state.history.push(snapshotCurrent());
  if (state.history.length > HISTORY_LIMIT) {
    state.history.shift();
  }
  if (colorRuleActive()) {
    const alive = flatAlive();
    const { nextAlive, nextColor } = stepColor(
      alive,
      state.color,
      neighborsOfFn(),
      cellCount(),
      state.colorSettings,
    );
    applyFlatAlive(nextAlive);
    state.color = nextColor;
  } else if (isTiling()) {
    state.tilingStates = stepTiling(state.tilingStates, state.tilingNeighbors);
  } else if (state.gridType === "triangle") {
    state.grid = stepGridTri(state.grid, state.wrap);
  } else {
    state.grid = stepGrid(state.grid, state.wrap);
  }
  state.generation += 1;
  syncLabels();
}

function squareMetrics(width, height) {
  const cell = Math.min(width / state.cols, height / state.rows);
  return {
    cell,
    ox: (width - cell * state.cols) / 2,
    oy: (height - cell * state.rows) / 2,
  };
}

function triMetrics(width, height) {
  const totalWidth = ((state.cols + 1) / 2);
  const baseW = width / totalWidth;
  const baseH = height / state.rows;
  const cellW = Math.min(baseW, baseH * 2 / SQRT3);
  const cellH = cellW * SQRT3 / 2;
  return {
    cellW,
    cellH,
    ox: (width - totalWidth * cellW) / 2,
    oy: (height - state.rows * cellH) / 2,
  };
}

function triCellPoints(r, c, metrics) {
  const isUp = (r + c) % 2 === 0;
  const x = metrics.ox + c * metrics.cellW / 2;
  const y = metrics.oy + r * metrics.cellH;
  if (isUp) {
    return [
      [x, y + metrics.cellH],
      [x + metrics.cellW / 2, y],
      [x + metrics.cellW, y + metrics.cellH],
    ];
  }
  return [
    [x, y],
    [x + metrics.cellW, y],
    [x + metrics.cellW / 2, y + metrics.cellH],
  ];
}

function tilingMetrics(width, height) {
  const [bboxW, bboxH] = state.tilingBBox;
  const scale = Math.min(width / bboxW, height / bboxH) * 0.92;
  return {
    scale,
    ox: (width - bboxW * scale) / 2,
    oy: (height - bboxH * scale) / 2,
  };
}

function isLightTheme() {
  return window.matchMedia("(prefers-color-scheme: light)").matches;
}

function themeCanvas() {
  const light = isLightTheme();
  return {
    canvasBg: light ? "#eceff4" : "#08121b",
    squareDead: light ? "#dbe2ec" : "#122033",
    gridStroke: light ? "rgba(30, 46, 66, 0.08)" : "rgba(181, 214, 255, 0.09)",
    tilingStroke: light ? "rgba(30, 46, 66, 0.18)" : "rgba(205, 223, 245, 0.14)",
  };
}

function gridColors() {
  const light = isLightTheme();
  const dead = light ? "#dbe2ec" : "#1b1e25";
  if (state.gridType === "rhombus") {
    return { alive: ["#b9b6cb", "#7a7b94", "#4a4c61"], dead };
  }
  if (state.gridType === "penrose") {
    return { alive: ["#5c89b8", "#d7a44d"], dead };
  }
  if (state.gridType === "trihex") {
    return { alive: ["#d8a862", "#4fb0b8"], dead };
  }
  if (state.gridType === "oct") {
    return { alive: ["#bad0ef", "#ffb55f"], dead };
  }
  if (state.gridType === "voronoi") {
    return { alive: ["#6be1af", "#d58aef", "#f5c46d"], dead };
  }
  return { alive: ["#f6d97d"], dead };
}

// Returns the fill color for a live cell at flat index `i`, honoring the
// active color ruleset and the colorsSwapped flag (which rotates hue 180°
// to its complement when a color ruleset is active).
function liveFillFor(i, fallbackHex) {
  if (!colorRuleActive()) {
    return fallbackHex;
  }
  let h = state.color.hue[i];
  const s = state.color.sat[i];
  if (state.colorsSwapped) h = wrapHue(h + 180);
  // If a cell is alive but somehow has zero saturation (e.g. just before
  // dying under Goethean), keep it visible at a low floor.
  const safeS = s > 0 ? s : 1.0;
  return hslCss(h, safeS, state.colorSettings.lightness);
}

function drawSquareGrid(width, height) {
  const theme = themeCanvas();
  const { cell, ox, oy } = squareMetrics(width, height);
  const aliveDefault = state.colorsSwapped ? theme.squareDead : "#f4d35e";
  const dead = state.colorsSwapped && !colorRuleActive() ? "#f4d35e" : theme.squareDead;
  ctx.fillStyle = theme.canvasBg;
  ctx.fillRect(0, 0, width, height);
  for (let r = 0; r < state.rows; r += 1) {
    for (let c = 0; c < state.cols; c += 1) {
      const alive = state.grid[r][c] === 1;
      ctx.fillStyle = alive ? liveFillFor(r * state.cols + c, aliveDefault) : dead;
      ctx.fillRect(ox + c * cell, oy + r * cell, cell, cell);
      if (cell > 6) {
        ctx.strokeStyle = theme.gridStroke;
        ctx.strokeRect(ox + c * cell, oy + r * cell, cell, cell);
      }
    }
  }
}

function drawTriangleGrid(width, height) {
  const theme = themeCanvas();
  const metrics = triMetrics(width, height);
  const aliveDefault = state.colorsSwapped ? theme.squareDead : "#f4d35e";
  const dead = state.colorsSwapped && !colorRuleActive() ? "#f4d35e" : theme.squareDead;
  ctx.fillStyle = theme.canvasBg;
  ctx.fillRect(0, 0, width, height);
  for (let r = 0; r < state.rows; r += 1) {
    for (let c = 0; c < state.cols; c += 1) {
      const points = triCellPoints(r, c, metrics);
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      ctx.lineTo(points[1][0], points[1][1]);
      ctx.lineTo(points[2][0], points[2][1]);
      ctx.closePath();
      const alive = state.grid[r][c] === 1;
      ctx.fillStyle = alive ? liveFillFor(r * state.cols + c, aliveDefault) : dead;
      ctx.fill();
      ctx.strokeStyle = theme.gridStroke;
      ctx.stroke();
    }
  }
}

function drawTiling(width, height) {
  const theme = themeCanvas();
  const palette = gridColors();
  const metrics = tilingMetrics(width, height);
  ctx.fillStyle = theme.canvasBg;
  ctx.fillRect(0, 0, width, height);
  state.polygons.forEach((poly, index) => {
    ctx.beginPath();
    poly.forEach(([x, y], pointIndex) => {
      const sx = metrics.ox + x * metrics.scale;
      const sy = metrics.oy + y * metrics.scale;
      if (pointIndex === 0) {
        ctx.moveTo(sx, sy);
      } else {
        ctx.lineTo(sx, sy);
      }
    });
    ctx.closePath();
    const face = state.tilingFaceTypes[index] % palette.alive.length;
    const aliveColor = palette.alive[face];
    const isAlive = state.tilingStates[index] === 1;
    let fill;
    if (colorRuleActive()) {
      // Color rules drive alive fill; dead cells stay on the theme dead color.
      fill = isAlive ? liveFillFor(index, aliveColor) : palette.dead;
    } else if (state.colorsSwapped) {
      fill = isAlive ? palette.dead : aliveColor;
    } else {
      fill = isAlive ? aliveColor : palette.dead;
    }
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = theme.tilingStroke;
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

function syncSidebar() {
  elements.body.classList.toggle("sidebar-open", state.sidebarOpen);
  elements.body.classList.toggle("sidebar-collapsed", !state.sidebarOpen);
  elements.sidebarHide.setAttribute("aria-expanded", String(state.sidebarOpen));
  elements.sidebarShow.setAttribute("aria-expanded", String(state.sidebarOpen));
}

function render() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  if (isTiling()) {
    drawTiling(rect.width, rect.height);
  } else if (state.gridType === "triangle") {
    drawTriangleGrid(rect.width, rect.height);
  } else {
    drawSquareGrid(rect.width, rect.height);
  }

  drawSelectionOverlay(rect.width, rect.height);
  drawPasteGhost(rect.width, rect.height);
}

function drawSelectionOverlay(width, height) {
  // Tinted highlight on selected cells.
  if (state.selection && state.selection.size > 0) {
    ctx.save();
    ctx.fillStyle = "rgba(110, 180, 255, 0.28)";
    ctx.strokeStyle = "rgba(180, 220, 255, 0.85)";
    ctx.lineWidth = 1.4;
    if (isTiling()) {
      const metrics = tilingMetrics(width, height);
      state.selection.forEach((idx) => {
        const poly = state.polygons[idx];
        ctx.beginPath();
        poly.forEach(([x, y], i) => {
          const sx = metrics.ox + x * metrics.scale;
          const sy = metrics.oy + y * metrics.scale;
          if (i === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        });
        ctx.closePath();
        ctx.fill();
      });
    } else if (state.gridType === "triangle") {
      const metrics = triMetrics(width, height);
      state.selection.forEach((idx) => {
        const r = Math.floor(idx / state.cols);
        const c = idx - r * state.cols;
        const points = triCellPoints(r, c, metrics);
        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        ctx.lineTo(points[1][0], points[1][1]);
        ctx.lineTo(points[2][0], points[2][1]);
        ctx.closePath();
        ctx.fill();
      });
    } else {
      const { cell, ox, oy } = squareMetrics(width, height);
      state.selection.forEach((idx) => {
        const r = Math.floor(idx / state.cols);
        const c = idx - r * state.cols;
        ctx.fillRect(ox + c * cell, oy + r * cell, cell, cell);
      });
    }
    ctx.restore();
  }
  // Drag rectangle in progress
  if (state.selectionDrag) {
    const d = state.selectionDrag;
    const x = Math.min(d.sx0, d.sx1);
    const y = Math.min(d.sy0, d.sy1);
    const w = Math.abs(d.sx1 - d.sx0);
    const h = Math.abs(d.sy1 - d.sy0);
    ctx.save();
    ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.restore();
  }
}

function drawPasteGhost(width, height) {
  if (state.tool !== "paste" || !state.clipboard || !state.pasteGhost) return;
  const pos = state.pasteGhost;
  ctx.save();
  ctx.fillStyle = "rgba(232, 192, 90, 0.35)";
  if (state.clipboard.kind === "grid" && !isTiling()) {
    const target = cellAtPointer(pos);
    if (!target) { ctx.restore(); return; }
    const [tr, tc] = target;
    if (state.gridType === "triangle") {
      const metrics = triMetrics(width, height);
      state.clipboard.items.forEach((item) => {
        const r = tr + item.dr;
        const c = tc + item.dc;
        if (r < 0 || r >= state.rows || c < 0 || c >= state.cols) return;
        const points = triCellPoints(r, c, metrics);
        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        ctx.lineTo(points[1][0], points[1][1]);
        ctx.lineTo(points[2][0], points[2][1]);
        ctx.closePath();
        ctx.fill();
      });
    } else {
      const { cell, ox, oy } = squareMetrics(width, height);
      state.clipboard.items.forEach((item) => {
        const r = tr + item.dr;
        const c = tc + item.dc;
        if (r < 0 || r >= state.rows || c < 0 || c >= state.cols) return;
        ctx.fillRect(ox + c * cell, oy + r * cell, cell, cell);
      });
    }
  } else if (state.clipboard.kind === "tiling" && isTiling()) {
    const metrics = tilingMetrics(width, height);
    const wx0 = (pos.x - metrics.ox) / metrics.scale;
    const wy0 = (pos.y - metrics.oy) / metrics.scale;
    state.clipboard.items.forEach((item) => {
      const tx = wx0 + item.dx;
      const ty = wy0 + item.dy;
      let bestI = -1;
      let bestD = Infinity;
      for (let i = 0; i < state.polygons.length; i += 1) {
        const [cx, cy] = polygonCentroid(state.polygons[i]);
        const dx = cx - tx;
        const dy = cy - ty;
        const d = dx * dx + dy * dy;
        if (d < bestD) {
          bestD = d;
          bestI = i;
        }
      }
      if (bestI < 0) return;
      const poly = state.polygons[bestI];
      ctx.beginPath();
      poly.forEach(([x, y], i) => {
        const sx = metrics.ox + x * metrics.scale;
        const sy = metrics.oy + y * metrics.scale;
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.closePath();
      ctx.fill();
    });
  }
  ctx.restore();
}

function syncLabels() {
  elements.generationValue.textContent = String(state.generation);
  elements.populationValue.textContent = String(population());
  elements.cellsValue.textContent = String(cellCount());
  elements.speedValue.textContent = String(state.speed);
  elements.densityValue.textContent = `${Math.round(state.density * 100)}%`;
  elements.thresholdValue.textContent = String(state.threshold);
  elements.stepBack.disabled = state.history.length === 0;
  elements.resetGenZero.disabled = state.generation === 0 || !state.gen0Snapshot;
  elements.swapColors.classList.toggle("active", state.colorsSwapped);
  syncVoronoiUI();
}

function syncVoronoiUI() {
  if (!elements.voronoiOptions) {
    return;
  }
  const show = state.gridType === "voronoi";
  elements.voronoiOptions.hidden = !show;
  elements.voronoiJitterValue.textContent = `${Math.round(state.voronoiJitter * 100)}%`;
  elements.voronoiEuclid.classList.toggle("active", state.voronoiMetric === "euclidean");
  elements.voronoiManhattan.classList.toggle("active", state.voronoiMetric === "manhattan");
}

function syncColorUI() {
  if (!elements.colorOptions) return;
  elements.colorOptions.hidden = !state.colorPanelOpen;
  elements.toggleColorPanel.classList.toggle("active", state.colorPanelOpen);
  elements.colorRule.value = state.colorSettings.rule;
  elements.colorLightness.value = String(Math.round(state.colorSettings.lightness * 100));
  elements.colorLightnessValue.textContent = `${Math.round(state.colorSettings.lightness * 100)}%`;
  const isGoethean = state.colorSettings.rule === RULESETS.GOETHEAN;
  const isRotation = state.colorSettings.rule === RULESETS.ROTATION;
  elements.colorGoetheanOptions.hidden = !isGoethean;
  elements.colorRotationOptions.hidden = !isRotation;
  elements.goetheanSatFull.classList.toggle("active", state.colorSettings.goetheanSatStart === "full");
  elements.goetheanSatInherit.classList.toggle("active", state.colorSettings.goetheanSatStart === "inherit");
  elements.rotationHueFixed.classList.toggle("active", state.colorSettings.rotationHueStart === "fixed");
  elements.rotationHueParent.classList.toggle("active", state.colorSettings.rotationHueStart === "parent");
  elements.rotationFixedHueField.hidden = state.colorSettings.rotationHueStart !== "fixed";
  elements.rotationFixedHue.value = hueToHex(state.colorSettings.rotationFixedHue);
  elements.rotationDelta.value = String(state.colorSettings.rotationDelta);
}

function syncToolsUI() {
  if (!elements.toolsOptions) return;
  elements.toolsOptions.hidden = !state.toolsPanelOpen;
  elements.toggleToolsPanel.classList.toggle("active", state.toolsPanelOpen);
  elements.toolPencil.classList.toggle("active", state.tool === "pencil");
  elements.toolEraser.classList.toggle("active", state.tool === "eraser");
  elements.toolSelect.classList.toggle("active", state.tool === "select");
  const hasSelection = state.selection instanceof Set && state.selection.size > 0;
  elements.toolCopy.disabled = !hasSelection;
  elements.toolCut.disabled = !hasSelection;
  elements.toolPaste.disabled = !state.clipboard;
  // Update canvas cursor class
  canvas.classList.toggle("tool-select", state.tool === "select");
  canvas.classList.toggle("tool-eraser", state.tool === "eraser");
  canvas.classList.toggle("tool-paste", state.tool === "paste");
}

function setColorRule(rule) {
  const prev = state.colorSettings.rule;
  state.colorSettings.rule = rule;
  if (rule !== RULESETS.NONE && prev === RULESETS.NONE) {
    // Activating: seed colors for currently-alive cells.
    seedColorsForAllAlive();
  } else if (rule === RULESETS.NONE) {
    // Deactivating: clear color arrays so a future re-activation starts clean.
    clearColorState(state.color);
  }
  // History snapshots become incompatible across rule changes (color presence
  // differs). Easiest correctness fix: drop history.
  state.history = [];
  state.gen0Snapshot = snapshotCurrent();
  syncLabels();
  syncColorUI();
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
    width: rect.width,
    height: rect.height,
  };
}

function cellAtPointer(pos) {
  if (isTiling()) {
    const metrics = tilingMetrics(pos.width, pos.height);
    const ux = (pos.x - metrics.ox) / metrics.scale;
    const uy = (pos.y - metrics.oy) / metrics.scale;
    for (let index = 0; index < state.polygons.length; index += 1) {
      if (pointInPolygon(ux, uy, state.polygons[index])) {
        return index;
      }
    }
    return null;
  }

  if (state.gridType === "triangle") {
    const metrics = triMetrics(pos.width, pos.height);
    const approxR = Math.floor((pos.y - metrics.oy) / metrics.cellH);
    const approxC = Math.floor((pos.x - metrics.ox) / (metrics.cellW / 2));
    for (let dr = -1; dr <= 2; dr += 1) {
      for (let dc = -2; dc <= 3; dc += 1) {
        const r = approxR + dr;
        const c = approxC + dc;
        if (r < 0 || r >= state.rows || c < 0 || c >= state.cols) {
          continue;
        }
        const points = triCellPoints(r, c, metrics);
        if (pointInTriangle(pos.x, pos.y, ...points[0], ...points[1], ...points[2])) {
          return [r, c];
        }
      }
    }
    return null;
  }

  const { cell, ox, oy } = squareMetrics(pos.width, pos.height);
  const c = Math.floor((pos.x - ox) / cell);
  const r = Math.floor((pos.y - oy) / cell);
  if (r >= 0 && r < state.rows && c >= 0 && c < state.cols) {
    return [r, c];
  }
  return null;
}

function applyPaint(target, value, hueOverride = null) {
  if (target == null) {
    return;
  }
  const flat = flatIndexOf(target);
  setAliveAt(flat, value);
  if (colorRuleActive()) {
    if (value) {
      const hue = hueOverride != null ? hueOverride : defaultSeedHue();
      seedColorAt(flat, hue, 1.0);
    } else {
      clearCellColor(state.color, flat);
    }
  }
  // Painting at gen 0 redefines the "base" state, so keep gen0 snapshot fresh.
  if (state.generation === 0) {
    state.gen0Snapshot = snapshotCurrent();
    state.history = [];
  }
  syncLabels();
}

function polygonCentroid(poly) {
  const total = poly.reduce((acc, [x, y]) => [acc[0] + x, acc[1] + y], [0, 0]);
  return [total[0] / poly.length, total[1] / poly.length];
}

function imageSampler(image) {
  const size = 240;
  const offscreen = document.createElement("canvas");
  offscreen.width = size;
  offscreen.height = size;
  const offCtx = offscreen.getContext("2d");
  offCtx.drawImage(image, 0, 0, size, size);
  const data = offCtx.getImageData(0, 0, size, size).data;
  return (normX, normY) => {
    const x = clamp(Math.floor(normX * (size - 1)), 0, size - 1);
    const y = clamp(Math.floor(normY * (size - 1)), 0, size - 1);
    const index = (y * size + x) * 4;
    const r = data[index] / 255;
    const g = data[index + 1] / 255;
    const b = data[index + 2] / 255;
    const brightness = 0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2];
    const hsl = rgbToHsl(r, g, b);
    return { brightness, hue: hsl.h, sat: Math.max(0.2, hsl.s) };
  };
}

function applyImageToCurrentGeometry(image) {
  const sample = imageSampler(image);
  state.generation = 0;
  state.running = false;
  elements.playToggle.textContent = "Play";
  const useColor = colorRuleActive();
  if (useColor) clearColorState(state.color);

  const setLive = (flatIdx, smp) => {
    if (useColor) {
      seedColorAt(flatIdx, smp.hue, smp.sat);
    }
  };

  if (isTiling()) {
    const [bboxW, bboxH] = state.tilingBBox;
    state.tilingStates = state.polygons.map((poly, index) => {
      const [cx, cy] = polygonCentroid(poly);
      const smp = sample(cx / bboxW, cy / bboxH);
      const alive = smp.brightness < state.threshold ? 1 : 0;
      if (alive) setLive(index, smp);
      return alive;
    });
  } else if (state.gridType === "triangle") {
    const metrics = {
      cellW: 1,
      cellH: SQRT3 / 2,
      ox: 0,
      oy: 0,
    };
    const totalW = ((state.cols + 1) / 2) * metrics.cellW;
    const totalH = state.rows * metrics.cellH;
    state.grid = makeGrid(state.rows, state.cols, 0);
    for (let r = 0; r < state.rows; r += 1) {
      for (let c = 0; c < state.cols; c += 1) {
        const points = triCellPoints(r, c, metrics);
        const centroid = polygonCentroid(points);
        const smp = sample(centroid[0] / totalW, centroid[1] / totalH);
        const alive = smp.brightness < state.threshold ? 1 : 0;
        state.grid[r][c] = alive;
        if (alive) setLive(r * state.cols + c, smp);
      }
    }
  } else {
    state.grid = makeGrid(state.rows, state.cols, 0);
    for (let r = 0; r < state.rows; r += 1) {
      for (let c = 0; c < state.cols; c += 1) {
        const smp = sample((c + 0.5) / state.cols, (r + 0.5) / state.rows);
        const alive = smp.brightness < state.threshold ? 1 : 0;
        state.grid[r][c] = alive;
        if (alive) setLive(r * state.cols + c, smp);
      }
    }
  }
  captureGenZero();
  syncLabels();
}

function saveCanvasSnapshot() {
  render();
  const link = document.createElement("a");
  link.href = canvas.toDataURL("image/png");
  link.download = "tnt-gol-playground.png";
  link.click();
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not load image"));
    };
    image.src = url;
  });
}

function handlePointerDown(event) {
  const pos = pointerPosition(event);

  if (state.tool === "select") {
    pointerState.selectingDrag = true;
    state.selectionDrag = { sx0: pos.x, sy0: pos.y, sx1: pos.x, sy1: pos.y };
    return;
  }
  if (state.tool === "paste") {
    applyPasteAtPointer(pos);
    state.tool = "pencil";
    syncToolsUI();
    return;
  }
  // pencil or eraser
  const target = cellAtPointer(pos);
  if (target == null) return;
  pointerState.painting = true;
  pointerState.drawValue = state.tool === "eraser" ? 0 : 1;
  applyPaint(target, pointerState.drawValue);
}

function handlePointerMove(event) {
  const pos = pointerPosition(event);
  if (state.tool === "paste") {
    state.pasteGhost = pos;
    return;
  }
  if (pointerState.selectingDrag && state.selectionDrag) {
    state.selectionDrag.sx1 = pos.x;
    state.selectionDrag.sy1 = pos.y;
    return;
  }
  if (!pointerState.painting) return;
  applyPaint(cellAtPointer(pos), pointerState.drawValue);
}

function stopPainting() {
  if (pointerState.selectingDrag) {
    finalizeSelection();
    pointerState.selectingDrag = false;
    state.selectionDrag = null;
  }
  pointerState.painting = false;
}

// ─── Selection / clipboard ─────────────────────────────────────────────────

function finalizeSelection() {
  const d = state.selectionDrag;
  if (!d) return;
  const x0 = Math.min(d.sx0, d.sx1);
  const y0 = Math.min(d.sy0, d.sy1);
  const x1 = Math.max(d.sx0, d.sx1);
  const y1 = Math.max(d.sy0, d.sy1);
  // Treat a tiny drag as a click → clear selection.
  if (x1 - x0 < 3 && y1 - y0 < 3) {
    state.selection = null;
    syncToolsUI();
    return;
  }
  const cells = new Set();
  const rect = canvas.getBoundingClientRect();
  if (isTiling()) {
    const metrics = tilingMetrics(rect.width, rect.height);
    state.polygons.forEach((poly, idx) => {
      const [cx, cy] = polygonCentroid(poly);
      const sx = metrics.ox + cx * metrics.scale;
      const sy = metrics.oy + cy * metrics.scale;
      if (sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1) cells.add(idx);
    });
  } else if (state.gridType === "triangle") {
    const metrics = triMetrics(rect.width, rect.height);
    for (let r = 0; r < state.rows; r += 1) {
      for (let c = 0; c < state.cols; c += 1) {
        const points = triCellPoints(r, c, metrics);
        const cx = (points[0][0] + points[1][0] + points[2][0]) / 3;
        const cy = (points[0][1] + points[1][1] + points[2][1]) / 3;
        if (cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1) cells.add(r * state.cols + c);
      }
    }
  } else {
    const { cell, ox, oy } = squareMetrics(rect.width, rect.height);
    const cMin = Math.max(0, Math.floor((x0 - ox) / cell));
    const cMax = Math.min(state.cols - 1, Math.floor((x1 - ox) / cell));
    const rMin = Math.max(0, Math.floor((y0 - oy) / cell));
    const rMax = Math.min(state.rows - 1, Math.floor((y1 - oy) / cell));
    for (let r = rMin; r <= rMax; r += 1) {
      for (let c = cMin; c <= cMax; c += 1) {
        cells.add(r * state.cols + c);
      }
    }
  }
  state.selection = cells.size > 0 ? cells : null;
  syncToolsUI();
}

// Snapshot the selection into the clipboard. If `cut`, also erase those cells.
function copySelection(cut) {
  if (!state.selection || state.selection.size === 0) return;
  const items = [];
  const useColor = colorRuleActive();
  if (isTiling()) {
    const cxs = [];
    const cys = [];
    state.selection.forEach((idx) => {
      const [cx, cy] = polygonCentroid(state.polygons[idx]);
      cxs.push(cx);
      cys.push(cy);
    });
    const ox = Math.min(...cxs);
    const oy = Math.min(...cys);
    state.selection.forEach((idx) => {
      const [cx, cy] = polygonCentroid(state.polygons[idx]);
      const item = { dx: cx - ox, dy: cy - oy };
      if (useColor) {
        item.hue = state.color.hue[idx];
        item.sat = state.color.sat[idx];
        item.origSat = state.color.origSat[idx];
        item.age = state.color.age[idx];
      }
      items.push(item);
    });
    state.clipboard = { kind: "tiling", items };
  } else {
    const rs = [];
    const cs = [];
    state.selection.forEach((idx) => {
      rs.push(Math.floor(idx / state.cols));
      cs.push(idx - Math.floor(idx / state.cols) * state.cols);
    });
    const r0 = Math.min(...rs);
    const c0 = Math.min(...cs);
    state.selection.forEach((idx) => {
      const r = Math.floor(idx / state.cols);
      const c = idx - r * state.cols;
      const item = { dr: r - r0, dc: c - c0 };
      if (useColor) {
        item.hue = state.color.hue[idx];
        item.sat = state.color.sat[idx];
        item.origSat = state.color.origSat[idx];
        item.age = state.color.age[idx];
      }
      items.push(item);
    });
    state.clipboard = { kind: "grid", items };
  }
  if (cut) {
    state.selection.forEach((idx) => {
      setAliveAt(idx, 0);
      if (useColor) clearCellColor(state.color, idx);
    });
    state.selection = null;
    if (state.generation === 0) {
      state.gen0Snapshot = snapshotCurrent();
      state.history = [];
    }
  }
  syncLabels();
  syncToolsUI();
}

function writeColorFromClip(idx, item) {
  if (item.hue == null) return;
  state.color.hue[idx] = item.hue;
  state.color.sat[idx] = item.sat ?? 1;
  state.color.origSat[idx] = item.origSat ?? item.sat ?? 1;
  state.color.age[idx] = item.age ?? 0;
}

function applyPasteAtPointer(pos) {
  if (!state.clipboard) return;
  const useColor = colorRuleActive();
  if (state.clipboard.kind === "tiling" && isTiling()) {
    const rect = canvas.getBoundingClientRect();
    const metrics = tilingMetrics(rect.width, rect.height);
    const wx0 = (pos.x - metrics.ox) / metrics.scale;
    const wy0 = (pos.y - metrics.oy) / metrics.scale;
    // Precompute centroids once
    const centroids = state.polygons.map(polygonCentroid);
    state.clipboard.items.forEach((item) => {
      const tx = wx0 + item.dx;
      const ty = wy0 + item.dy;
      let bestI = -1;
      let bestD = Infinity;
      for (let i = 0; i < centroids.length; i += 1) {
        const dx = centroids[i][0] - tx;
        const dy = centroids[i][1] - ty;
        const d = dx * dx + dy * dy;
        if (d < bestD) {
          bestD = d;
          bestI = i;
        }
      }
      if (bestI >= 0) {
        setAliveAt(bestI, 1);
        if (useColor) writeColorFromClip(bestI, item);
        else if (colorRuleActive()) seedColorAt(bestI, defaultSeedHue(), 1);
      }
    });
  } else if (state.clipboard.kind === "grid" && !isTiling()) {
    const target = cellAtPointer(pos);
    if (!target) return;
    const [tr, tc] = target;
    state.clipboard.items.forEach((item) => {
      const r = tr + item.dr;
      const c = tc + item.dc;
      if (r < 0 || r >= state.rows || c < 0 || c >= state.cols) return;
      const idx = r * state.cols + c;
      setAliveAt(idx, 1);
      if (useColor) writeColorFromClip(idx, item);
    });
  } else {
    // Geometry mismatch — silently no-op.
    return;
  }
  if (state.generation === 0) {
    state.gen0Snapshot = snapshotCurrent();
    state.history = [];
  }
  syncLabels();
}

function animate(timestamp) {
  if (!lastFrameTime) {
    lastFrameTime = timestamp;
  }
  const delta = timestamp - lastFrameTime;
  lastFrameTime = timestamp;
  if (state.running) {
    accumulator += delta;
    const interval = 1000 / state.speed;
    while (accumulator >= interval) {
      stepOnce();
      accumulator -= interval;
    }
  } else {
    accumulator = 0;
  }
  render();
  animationFrame = window.requestAnimationFrame(animate);
}

function bindEvents() {
  const setSidebar = (open) => {
    state.sidebarOpen = open;
    syncSidebar();
  };
  elements.sidebarHide.addEventListener("click", () => setSidebar(false));
  elements.sidebarShow.addEventListener("click", () => setSidebar(true));
  elements.playToggle.addEventListener("click", () => {
    state.running = !state.running;
    elements.playToggle.textContent = state.running ? "Pause" : "Play";
  });
  elements.stepOnce.addEventListener("click", stepOnce);
  elements.stepBack.addEventListener("click", stepBack);
  elements.resetGenZero.addEventListener("click", resetToGenZero);
  elements.swapColors.addEventListener("click", swapColors);
  elements.clearGrid.addEventListener("click", clearState);
  elements.randomizeGrid.addEventListener("click", randomizeState);
  elements.wrapToggle.addEventListener("change", (event) => {
    state.wrap = event.target.checked;
  });
  elements.gridType.addEventListener("change", (event) => {
    state.gridType = event.target.value;
    rebuildTopology();
  });
  elements.applySize.addEventListener("click", () => {
    state.rows = clamp(Number(elements.rowsInput.value) || 64, 4, 1000);
    state.cols = clamp(Number(elements.colsInput.value) || 64, 4, 1000);
    elements.rowsInput.value = String(state.rows);
    elements.colsInput.value = String(state.cols);
    rebuildTopology();
  });
  elements.rebuildMap.addEventListener("click", () => {
    if (state.gridType === "voronoi") {
      state.voronoiSeed = Math.floor(Math.random() * 1_000_000);
    }
    rebuildTopology();
  });
  elements.voronoiEuclid.addEventListener("click", () => {
    state.voronoiMetric = "euclidean";
    if (state.gridType === "voronoi") {
      rebuildTopology();
    } else {
      syncVoronoiUI();
    }
  });
  elements.voronoiManhattan.addEventListener("click", () => {
    state.voronoiMetric = "manhattan";
    if (state.gridType === "voronoi") {
      rebuildTopology();
    } else {
      syncVoronoiUI();
    }
  });
  elements.voronoiJitter.addEventListener("change", (event) => {
    state.voronoiJitter = Number(event.target.value) / 100;
    if (state.gridType === "voronoi") {
      rebuildTopology();
    } else {
      syncVoronoiUI();
    }
  });
  elements.voronoiJitter.addEventListener("input", (event) => {
    state.voronoiJitter = Number(event.target.value) / 100;
    syncVoronoiUI();
  });
  elements.speedInput.addEventListener("input", (event) => {
    state.speed = Number(event.target.value);
    syncLabels();
  });
  elements.densityInput.addEventListener("input", (event) => {
    state.density = Number(event.target.value) / 100;
    syncLabels();
  });
  elements.thresholdInput.addEventListener("input", (event) => {
    state.threshold = Number(event.target.value);
    syncLabels();
  });
  elements.invertGrid.addEventListener("click", invertState);
  elements.snapshotImage.addEventListener("click", saveCanvasSnapshot);
  elements.imageInput.addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    if (!file) {
      return;
    }
    try {
      const image = await loadImageFromFile(file);
      applyImageToCurrentGeometry(image);
      render();
    } catch (error) {
      console.error(error);
    }
  });

  // ─── Color Mode panel ───────────────────────────────────────────────────
  elements.toggleColorPanel.addEventListener("click", () => {
    state.colorPanelOpen = !state.colorPanelOpen;
    syncColorUI();
  });
  elements.colorRule.addEventListener("change", (event) => {
    setColorRule(event.target.value);
  });
  elements.colorLightness.addEventListener("input", (event) => {
    state.colorSettings.lightness = clamp(Number(event.target.value) / 100, 0.05, 0.95);
    elements.colorLightnessValue.textContent = `${Math.round(state.colorSettings.lightness * 100)}%`;
  });
  elements.goetheanSatFull.addEventListener("click", () => {
    state.colorSettings.goetheanSatStart = "full";
    syncColorUI();
  });
  elements.goetheanSatInherit.addEventListener("click", () => {
    state.colorSettings.goetheanSatStart = "inherit";
    syncColorUI();
  });
  elements.rotationHueFixed.addEventListener("click", () => {
    state.colorSettings.rotationHueStart = "fixed";
    syncColorUI();
  });
  elements.rotationHueParent.addEventListener("click", () => {
    state.colorSettings.rotationHueStart = "parent";
    syncColorUI();
  });
  elements.rotationFixedHue.addEventListener("input", (event) => {
    const { h } = hexToHsl(event.target.value);
    state.colorSettings.rotationFixedHue = h;
  });
  elements.rotationDelta.addEventListener("change", (event) => {
    const val = clamp(Number(event.target.value) || 0, -180, 180);
    state.colorSettings.rotationDelta = val;
    elements.rotationDelta.value = String(val);
  });

  // ─── Tools panel ────────────────────────────────────────────────────────
  elements.toggleToolsPanel.addEventListener("click", () => {
    state.toolsPanelOpen = !state.toolsPanelOpen;
    syncToolsUI();
  });
  elements.toolPencil.addEventListener("click", () => { state.tool = "pencil"; syncToolsUI(); });
  elements.toolEraser.addEventListener("click", () => { state.tool = "eraser"; syncToolsUI(); });
  elements.toolSelect.addEventListener("click", () => { state.tool = "select"; syncToolsUI(); });
  elements.toolCopy.addEventListener("click", () => copySelection(false));
  elements.toolCut.addEventListener("click", () => copySelection(true));
  elements.toolPaste.addEventListener("click", () => {
    if (!state.clipboard) return;
    state.tool = "paste";
    syncToolsUI();
  });

  canvas.addEventListener("pointerdown", handlePointerDown);
  canvas.addEventListener("pointermove", handlePointerMove);
  window.addEventListener("pointerup", stopPainting);
  window.addEventListener("pointercancel", stopPainting);
  window.addEventListener("resize", () => {
    render();
  });
  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", render);
}

function init() {
  elements.speedInput.value = String(state.speed);
  elements.densityInput.value = String(Math.round(state.density * 100));
  elements.thresholdInput.value = String(state.threshold);
  elements.gridType.value = state.gridType;
  elements.rowsInput.value = String(state.rows);
  elements.colsInput.value = String(state.cols);
  elements.voronoiJitter.value = String(Math.round(state.voronoiJitter * 100));
  elements.rotationFixedHue.value = hueToHex(state.colorSettings.rotationFixedHue);
  elements.rotationDelta.value = String(state.colorSettings.rotationDelta);
  elements.colorLightness.value = String(Math.round(state.colorSettings.lightness * 100));
  syncSidebar();
  bindEvents();
  rebuildTopology();
  randomizeState();
  syncColorUI();
  syncToolsUI();
  syncLabels();
  animationFrame = window.requestAnimationFrame(animate);
}

window.addEventListener("beforeunload", () => {
  window.cancelAnimationFrame(animationFrame);
});

// ─── Floating tooltip ────────────────────────────────────────────────────────
(function initTooltips() {
  const tip = document.createElement("div");
  tip.id = "floating-tooltip";
  document.body.appendChild(tip);

  function show(text, anchorRect) {
    tip.textContent = text;
    tip.classList.remove("visible");
    // Force layout so we get real dimensions before positioning
    tip.style.left = "-9999px";
    tip.style.top = "-9999px";
    tip.style.display = "block";

    const tw = tip.offsetWidth;
    const th = tip.offsetHeight;
    const margin = 8;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    // Try left of anchor; fall back to right
    let left = anchorRect.left - tw - margin;
    if (left < margin) {
      left = anchorRect.right + margin;
    }
    // Clamp right edge
    if (left + tw > vw - margin) {
      left = vw - margin - tw;
    }

    // Vertically center on anchor, clamped to viewport
    let top = anchorRect.top + (anchorRect.height - th) / 2;
    top = Math.max(margin, Math.min(top, vh - margin - th));

    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
    tip.classList.add("visible");
  }

  function hide() {
    tip.classList.remove("visible");
  }

  document.querySelectorAll("[data-tooltip]").forEach((el) => {
    el.addEventListener("mouseenter", () => show(el.dataset.tooltip, el.getBoundingClientRect()));
    el.addEventListener("mouseleave", hide);
    el.addEventListener("focus", () => show(el.dataset.tooltip, el.getBoundingClientRect()));
    el.addEventListener("blur", hide);
  });
}());

init();
