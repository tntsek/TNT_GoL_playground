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
  sidebarOpen: window.innerWidth > 720,
};

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

function generateVoronoiTiling(rows, cols, seed = 42) {
  const rng = mulberry32(seed);
  const seeds = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      seeds.push([col + 0.15 + 0.7 * rng(), row + 0.15 + 0.7 * rng()]);
    }
  }

  const polys = [];
  const faceTypes = [];
  const box = [-0.3, -0.3, cols - 0.7, rows - 0.7];

  for (let i = 0; i < seeds.length; i += 1) {
    const [sx, sy] = seeds[i];
    let cell = [
      [box[0], box[1]], [box[2], box[1]],
      [box[2], box[3]], [box[0], box[3]],
    ];
    for (let j = 0; j < seeds.length; j += 1) {
      if (i === j) {
        continue;
      }
      const [tx, ty] = seeds[j];
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
        Math.max(4, Math.floor(state.rows / 4)),
        Math.max(4, Math.floor(state.cols / 4)),
        state.voronoiSeed,
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
  syncLabels();
}

function randomizeState() {
  state.generation = 0;
  if (isTiling()) {
    state.tilingStates = state.tilingStates.map(() => (Math.random() < state.density ? 1 : 0));
  } else {
    state.grid = state.grid.map((row) => row.map(() => (Math.random() < state.density ? 1 : 0)));
  }
  syncLabels();
}

function clearState() {
  state.generation = 0;
  if (isTiling()) {
    state.tilingStates = state.tilingStates.map(() => 0);
  } else {
    state.grid = makeGrid(state.rows, state.cols, 0);
  }
  syncLabels();
}

function invertState() {
  if (isTiling()) {
    state.tilingStates = state.tilingStates.map((value) => 1 - value);
  } else {
    state.grid = state.grid.map((row) => row.map((value) => 1 - value));
  }
  syncLabels();
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

function stepOnce() {
  if (isTiling()) {
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

function drawSquareGrid(width, height) {
  const theme = themeCanvas();
  const { cell, ox, oy } = squareMetrics(width, height);
  ctx.fillStyle = theme.canvasBg;
  ctx.fillRect(0, 0, width, height);
  for (let r = 0; r < state.rows; r += 1) {
    for (let c = 0; c < state.cols; c += 1) {
      ctx.fillStyle = state.grid[r][c] ? "#f4d35e" : theme.squareDead;
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
      ctx.fillStyle = state.grid[r][c] ? "#f4d35e" : theme.squareDead;
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
    ctx.fillStyle = state.tilingStates[index] ? palette.alive[face] : palette.dead;
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
}

function syncLabels() {
  elements.generationValue.textContent = String(state.generation);
  elements.populationValue.textContent = String(population());
  elements.cellsValue.textContent = String(cellCount());
  elements.speedValue.textContent = String(state.speed);
  elements.densityValue.textContent = `${Math.round(state.density * 100)}%`;
  elements.thresholdValue.textContent = String(state.threshold);
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

function applyPaint(target, value) {
  if (target == null) {
    return;
  }
  if (isTiling()) {
    state.tilingStates[target] = value;
  } else {
    const [r, c] = target;
    state.grid[r][c] = value;
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
    return 0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2];
  };
}

function applyImageToCurrentGeometry(image) {
  const sample = imageSampler(image);
  state.generation = 0;
  state.running = false;
  elements.playToggle.textContent = "Play";

  if (isTiling()) {
    const [bboxW, bboxH] = state.tilingBBox;
    state.tilingStates = state.polygons.map((poly) => {
      const [cx, cy] = polygonCentroid(poly);
      const brightness = sample(cx / bboxW, cy / bboxH);
      return Number(brightness < state.threshold);
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
        const brightness = sample(centroid[0] / totalW, centroid[1] / totalH);
        state.grid[r][c] = Number(brightness < state.threshold);
      }
    }
  } else {
    state.grid = makeGrid(state.rows, state.cols, 0);
    for (let r = 0; r < state.rows; r += 1) {
      for (let c = 0; c < state.cols; c += 1) {
        const brightness = sample((c + 0.5) / state.cols, (r + 0.5) / state.rows);
        state.grid[r][c] = Number(brightness < state.threshold);
      }
    }
  }
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
  const target = cellAtPointer(pos);
  if (target == null) {
    return;
  }
  pointerState.painting = true;
  pointerState.drawValue = isTiling()
    ? (state.tilingStates[target] ? 0 : 1)
    : (state.grid[target[0]][target[1]] ? 0 : 1);
  applyPaint(target, pointerState.drawValue);
}

function handlePointerMove(event) {
  if (!pointerState.painting) {
    return;
  }
  applyPaint(cellAtPointer(pointerPosition(event)), pointerState.drawValue);
}

function stopPainting() {
  pointerState.painting = false;
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
    state.rows = clamp(Number(elements.rowsInput.value) || 64, 8, 180);
    state.cols = clamp(Number(elements.colsInput.value) || 64, 8, 180);
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
  syncSidebar();
  bindEvents();
  rebuildTopology();
  randomizeState();
  syncLabels();
  animationFrame = window.requestAnimationFrame(animate);
}

window.addEventListener("beforeunload", () => {
  window.cancelAnimationFrame(animationFrame);
});

init();
