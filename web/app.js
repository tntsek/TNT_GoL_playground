import {
  RULESETS,
  defaultColorSettings,
  makeColorState,
  cloneColorState,
  clearColorState,
  clearCellColor,
  setCellColor,
  stepColor,
  circularMean,
  wrapHue,
  hexToHsl,
  rgbToHsl,
  hslCss,
  hueToHex,
} from "./colorRules.js?v=6";

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

const TILING_TYPES = new Set(["rhombus", "penrose", "einstein", "hex", "trihex", "oct", "voronoi"]);

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
  deathFadeSteps: document.querySelector("#death-fade-steps"),
  freezeHueDying: document.querySelector("#freeze-hue-dying"),
  ghostFadeSteps: document.querySelector("#ghost-fade-steps"),
  blockGhostBirths: document.querySelector("#block-ghost-births"),
  deathsCreateHoles: document.querySelector("#deaths-create-holes"),
  toolsOptions: document.querySelector("#tools-options"),
  toolBrush: document.querySelector("#tool-brush"),
  toolHole: document.querySelector("#tool-hole"),
  toolSelect: document.querySelector("#tool-select"),
  brushSize: document.querySelector("#brush-size"),
  brushSizeValue: document.querySelector("#brush-size-value"),
  paintColor: document.querySelector("#paint-color"),
  paintColorField: document.querySelector("#paint-color-field"),
  toolCopy: document.querySelector("#tool-copy"),
  toolCut: document.querySelector("#tool-cut"),
  toolPaste: document.querySelector("#tool-paste"),
  fredkinToggle: document.querySelector("#fredkin-toggle"),
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
  // Holes — flat Uint8Array of cellCount(); 1 = hole/wall (no simulation here).
  holes: new Uint8Array(0),
  // Editor tools
  tool: "brush", // 'brush' | 'select' | 'paste'
  brushSize: 1, // 1..10 (cells in diameter)
  paintHue: 0, // 0..360 — hue used for brush-painted cells under a color rule
  paintSat: 1, // 0..1 — 0 makes brush-painted cells achromatic (gray/white/black)
  selection: null, // Set<flatIdx> | null
  selectionDrag: null, // in-progress rectangle in screen coords
  clipboard: null, // see editor-tools commit
  pasteGhost: null, // {dx,dy} during move/paste preview
  // Submenu visibility (UI)
  colorPanelOpen: false,
  toolsPanelOpen: false,
  // Fredkin (2nd-order reversible XOR) rule
  secondOrder: false,
  prevGrid: null,
  prevTilingStates: null,
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

function stepGrid(grid, wrap, holes) {
  const rows = grid.length;
  const cols = grid[0].length;
  const next = makeGrid(rows, cols);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      if (holes && holes[r * cols + c]) {
        next[r][c] = 0;
        continue;
      }
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
          if (holes && holes[nr * cols + nc]) continue;
          total += grid[nr][nc];
        }
      }
      next[r][c] = grid[r][c] ? Number(total === 2 || total === 3) : Number(total === 3);
    }
  }
  return next;
}

function stepGridTri(grid, wrap, holes) {
  const rows = grid.length;
  const cols = grid[0].length;
  const next = makeGrid(rows, cols);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      if (holes && holes[r * cols + c]) {
        next[r][c] = 0;
        continue;
      }
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
        if (holes && holes[nr * cols + nc]) return;
        total += grid[nr][nc];
      });
      next[r][c] = grid[r][c] ? Number(total === 2 || total === 3) : Number(total === 3);
    }
  }
  return next;
}

function stepTiling(states, neighbors, holes) {
  return states.map((alive, index) => {
    if (holes && holes[index]) return 0;
    let total = 0;
    const nbrs = neighbors[index];
    for (let k = 0; k < nbrs.length; k += 1) {
      const j = nbrs[k];
      if (holes && holes[j]) continue;
      total += states[j];
    }
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

function generateEinsteinTiling(levels = 3) {
  // Hat monotile substitution adapted from Craig S. Kaplan's BSD-licensed
  // hatviz construction. It returns only the final hat polygons.
  const r3 = Math.sqrt(3);
  const hr3 = r3 / 2;
  const ident = [1, 0, 0, 0, 1, 0];
  const add2 = ([ax, ay], [bx, by]) => [ax + bx, ay + by];
  const sub2 = ([ax, ay], [bx, by]) => [ax - bx, ay - by];
  const scl2 = ([x, y], factor) => [x * factor, y * factor];
  const hexPt = (x, y) => [x + 0.5 * y, hr3 * y];
  const inv = (T) => {
    const det = T[0] * T[4] - T[1] * T[3];
    return [
      T[4] / det,
      -T[1] / det,
      (T[1] * T[5] - T[2] * T[4]) / det,
      -T[3] / det,
      T[0] / det,
      (T[2] * T[3] - T[0] * T[5]) / det,
    ];
  };
  const mul = (A, B) => [
    A[0] * B[0] + A[1] * B[3],
    A[0] * B[1] + A[1] * B[4],
    A[0] * B[2] + A[1] * B[5] + A[2],
    A[3] * B[0] + A[4] * B[3],
    A[3] * B[1] + A[4] * B[4],
    A[3] * B[2] + A[4] * B[5] + A[5],
  ];
  const trot = (a) => [Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a), 0];
  const ttrans = ([x, y]) => [1, 0, x, 0, 1, y];
  const transPt = (M, [x, y]) => [M[0] * x + M[1] * y + M[2], M[3] * x + M[4] * y + M[5]];
  const rotAbout = (p, angle) => mul(ttrans(p), mul(trot(angle), ttrans([-p[0], -p[1]])));
  const matchSeg = (p, q) => [q[0] - p[0], p[1] - q[1], p[0], q[1] - p[1], q[0] - p[0], p[1]];
  const matchTwo = (p1, q1, p2, q2) => mul(matchSeg(p2, q2), inv(matchSeg(p1, q1)));
  const intersect = (p1, q1, p2, q2) => {
    const d = (q2[1] - p2[1]) * (q1[0] - p1[0]) - (q2[0] - p2[0]) * (q1[1] - p1[1]);
    const uA = ((q2[0] - p2[0]) * (p1[1] - p2[1]) - (q2[1] - p2[1]) * (p1[0] - p2[0])) / d;
    return [p1[0] + uA * (q1[0] - p1[0]), p1[1] + uA * (q1[1] - p1[1])];
  };

  class Geom {
    constructor(shape, fill = 0) {
      this.shape = shape;
      this.fill = fill;
      this.width = 1;
      this.children = [];
    }

    addChild(T, geom) {
      this.children.push({ T, geom });
    }

    evalChild(childIndex, pointIndex) {
      const child = this.children[childIndex];
      return transPt(child.T, child.geom.shape[pointIndex]);
    }

    recenter() {
      let tr = this.shape.reduce((acc, point) => add2(acc, point), [0, 0]);
      tr = scl2(tr, -1 / this.shape.length);
      this.shape = this.shape.map((point) => add2(point, tr));
      const M = ttrans(tr);
      this.children.forEach((child) => {
        child.T = mul(M, child.T);
      });
    }
  }

  const hatOutline = [
    hexPt(0, 0), hexPt(-1, -1), hexPt(0, -2), hexPt(2, -2),
    hexPt(2, -1), hexPt(4, -2), hexPt(5, -1), hexPt(4, 0),
    hexPt(3, 0), hexPt(2, 2), hexPt(0, 3), hexPt(0, 2),
    hexPt(-1, 2),
  ];
  const H1Hat = new Geom(hatOutline, 0);
  const HHat = new Geom(hatOutline, 1);
  const THat = new Geom(hatOutline, 2);
  const PHat = new Geom(hatOutline, 3);
  const FHat = new Geom(hatOutline, 4);

  function constructPatch(H, T, P, F) {
    const rules = [
      ["H"],
      [0, 0, "P", 2],
      [1, 0, "H", 2],
      [2, 0, "P", 2],
      [3, 0, "H", 2],
      [4, 4, "P", 2],
      [0, 4, "F", 3],
      [2, 4, "F", 3],
      [4, 1, 3, 2, "F", 0],
      [8, 3, "H", 0],
      [9, 2, "P", 0],
      [10, 2, "H", 0],
      [11, 4, "P", 2],
      [12, 0, "H", 2],
      [13, 0, "F", 3],
      [14, 2, "F", 1],
      [15, 3, "H", 4],
      [8, 2, "F", 1],
      [17, 3, "H", 0],
      [18, 2, "P", 0],
      [19, 2, "H", 2],
      [20, 4, "F", 3],
      [20, 0, "P", 2],
      [22, 0, "H", 2],
      [23, 4, "F", 3],
      [23, 0, "F", 3],
      [16, 0, "P", 2],
      [9, 4, 0, 2, "T", 2],
      [4, 0, "F", 3],
    ];
    const ret = new Geom([], null);
    ret.width = H.width;
    const shapes = { H, T, P, F };
    rules.forEach((rule) => {
      if (rule.length === 1) {
        ret.addChild(ident, shapes[rule[0]]);
      } else if (rule.length === 4) {
        const base = ret.children[rule[0]];
        const poly = base.geom.shape;
        const p = transPt(base.T, poly[(rule[1] + 1) % poly.length]);
        const q = transPt(base.T, poly[rule[1]]);
        const nextShape = shapes[rule[2]];
        const nextPoly = nextShape.shape;
        ret.addChild(matchTwo(nextPoly[rule[3]], nextPoly[(rule[3] + 1) % nextPoly.length], p, q), nextShape);
      } else {
        const childP = ret.children[rule[0]];
        const childQ = ret.children[rule[2]];
        const p = transPt(childQ.T, childQ.geom.shape[rule[3]]);
        const q = transPt(childP.T, childP.geom.shape[rule[1]]);
        const nextShape = shapes[rule[4]];
        const nextPoly = nextShape.shape;
        ret.addChild(matchTwo(nextPoly[rule[5]], nextPoly[(rule[5] + 1) % nextPoly.length], p, q), nextShape);
      }
    });
    return ret;
  }

  function constructMetatiles(patch) {
    const bps1 = patch.evalChild(8, 2);
    const bps2 = patch.evalChild(21, 2);
    const rbps = transPt(rotAbout(bps1, -2 * Math.PI / 3), bps2);
    const p72 = patch.evalChild(7, 2);
    const p252 = patch.evalChild(25, 2);
    const llc = intersect(bps1, rbps, patch.evalChild(6, 2), p72);

    let w = sub2(patch.evalChild(6, 2), llc);
    const newHOutline = [llc, bps1];
    w = transPt(trot(-Math.PI / 3), w);
    newHOutline.push(add2(newHOutline[1], w));
    newHOutline.push(patch.evalChild(14, 2));
    w = transPt(trot(-Math.PI / 3), w);
    newHOutline.push(sub2(newHOutline[3], w));
    newHOutline.push(patch.evalChild(6, 2));
    const newH = new Geom(newHOutline);
    newH.width = patch.width * 2;
    [0, 9, 16, 27, 26, 6, 1, 8, 10, 15].forEach((child) => {
      newH.addChild(patch.children[child].T, patch.children[child].geom);
    });

    const newP = new Geom([p72, add2(p72, sub2(bps1, llc)), bps1, llc]);
    newP.width = patch.width * 2;
    [7, 2, 3, 4, 28].forEach((child) => {
      newP.addChild(patch.children[child].T, patch.children[child].geom);
    });

    const newF = new Geom([
      bps2,
      patch.evalChild(24, 2),
      patch.evalChild(25, 0),
      p252,
      add2(p252, sub2(llc, bps1)),
    ]);
    newF.width = patch.width * 2;
    [21, 20, 22, 23, 24, 25].forEach((child) => {
      newF.addChild(patch.children[child].T, patch.children[child].geom);
    });

    const AAA = newHOutline[2];
    const BBB = add2(newHOutline[1], sub2(newHOutline[4], newHOutline[5]));
    const CCC = transPt(rotAbout(BBB, -Math.PI / 3), AAA);
    const newT = new Geom([BBB, CCC, AAA]);
    newT.width = patch.width * 2;
    newT.addChild(patch.children[11].T, patch.children[11].geom);

    [newH, newP, newF, newT].forEach((geom) => geom.recenter());
    return [newH, newT, newP, newF];
  }

  const HOutline = [[0, 0], [4, 0], [4.5, hr3], [2.5, 5 * hr3], [1.5, 5 * hr3], [-0.5, hr3]];
  const HInit = new Geom(HOutline);
  HInit.width = 2;
  HInit.addChild(matchTwo(hatOutline[5], hatOutline[7], HOutline[5], HOutline[0]), HHat);
  HInit.addChild(matchTwo(hatOutline[9], hatOutline[11], HOutline[1], HOutline[2]), HHat);
  HInit.addChild(matchTwo(hatOutline[5], hatOutline[7], HOutline[3], HOutline[4]), HHat);
  HInit.addChild(mul(ttrans([2.5, hr3]), mul([-0.5, -hr3, 0, hr3, -0.5, 0], [0.5, 0, 0, 0, -0.5, 0])), H1Hat);

  const TOutline = [[0, 0], [3, 0], [1.5, 3 * hr3]];
  const TInit = new Geom(TOutline);
  TInit.width = 2;
  TInit.addChild([0.5, 0, 0.5, 0, 0.5, hr3], THat);

  const POutline = [[0, 0], [4, 0], [3, 2 * hr3], [-1, 2 * hr3]];
  const PInit = new Geom(POutline);
  PInit.width = 2;
  PInit.addChild([0.5, 0, 1.5, 0, 0.5, hr3], PHat);
  PInit.addChild(mul(ttrans([0, 2 * hr3]), mul([0.5, hr3, 0, -hr3, 0.5, 0], [0.5, 0, 0, 0, 0.5, 0])), PHat);

  const FOutline = [[0, 0], [3, 0], [3.5, hr3], [3, 2 * hr3], [-1, 2 * hr3]];
  const FInit = new Geom(FOutline);
  FInit.width = 2;
  FInit.addChild([0.5, 0, 1.5, 0, 0.5, hr3], FHat);
  FInit.addChild(mul(ttrans([0, 2 * hr3]), mul([0.5, hr3, 0, -hr3, 0.5, 0], [0.5, 0, 0, 0, 0.5, 0])), FHat);

  let tiles = [HInit, TInit, PInit, FInit];
  for (let i = 0; i < levels; i += 1) {
    tiles = constructMetatiles(constructPatch(...tiles));
  }

  const polys = [];
  const faceTypes = [];
  const queue = [{ T: ident, geom: tiles[0], level: levels }];
  while (queue.length) {
    const item = queue.pop();
    if (item.level >= 0) {
      item.geom.children.forEach((child) => {
        queue.push({ T: mul(item.T, child.T), geom: child.geom, level: item.level - 1 });
      });
    } else {
      polys.push(item.geom.shape.map((point) => transPt(item.T, point)));
      faceTypes.push(item.geom.fill ?? 0);
    }
  }

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
    } else if (state.gridType === "einstein") {
      const dim = Math.max(state.rows, state.cols);
      const levels = dim <= 32 ? 2 : dim <= 128 ? 3 : 4;
      [polys, faceTypes, bbox] = generateEinsteinTiling(levels);
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
  clearPrevState();
  resizeHoles();
  syncLabels();
}

function snapshotCurrent() {
  const base = isTiling()
    ? {
        type: "tiling",
        data: state.tilingStates.slice(),
        prev: state.prevTilingStates ? state.prevTilingStates.slice() : null,
      }
    : {
        type: "grid",
        data: state.grid.map((row) => row.slice()),
        prev: state.prevGrid ? state.prevGrid.map((row) => row.slice()) : null,
      };
  if (colorRuleActive()) {
    base.color = cloneColorState(state.color);
  }
  // Always snapshot holes so step-back restores them too (deaths-create-holes
  // can grow the hole set over time).
  base.holes = new Uint8Array(state.holes);
  return base;
}

function restoreSnapshot(snap) {
  if (!snap) {
    return false;
  }
  let ok = false;
  if (snap.type === "tiling" && isTiling() && snap.data.length === state.tilingStates.length) {
    state.tilingStates = snap.data.slice();
    state.prevTilingStates = snap.prev ? snap.prev.slice() : null;
    ok = true;
  } else if (snap.type === "grid" && !isTiling()) {
    state.grid = snap.data.map((row) => row.slice());
    state.prevGrid = snap.prev ? snap.prev.map((row) => row.slice()) : null;
    ok = true;
  }
  if (ok && snap.color && snap.color.hue.length === state.color.hue.length) {
    state.color = cloneColorState(snap.color);
  } else if (ok && colorRuleActive()) {
    // Snapshot predates color rule activation — re-seed alive cells.
    seedColorsForAllAlive();
  }
  if (ok && snap.holes && snap.holes.length === state.holes.length) {
    state.holes = new Uint8Array(snap.holes);
  }
  return ok;
}

function captureGenZero() {
  state.gen0Snapshot = snapshotCurrent();
  state.history = [];
}

function clearPrevState() {
  state.prevGrid = null;
  state.prevTilingStates = null;
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
  clearPrevState();
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
  // state but not the ruleset choice). Holes and Fredkin's prev-generation
  // tracking are cleared too.
  clearColorState(state.color);
  clearHoles();
  state.selection = null;
  clearPrevState();
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
  clearPrevState();
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
    // Gen 0 has no prior step; snapshot.prev is ignored for semantics.
    clearPrevState();
    syncLabels();
  }
}

function inverseFredkinStep() {
  // Fredkin forward: (prev, cur) -> (cur, Conway(cur) XOR prev).
  // Inverse:        (prev, cur) -> (Conway(prev) XOR cur, prev).
  // Missing prev is treated as all-dead.
  if (isTiling()) {
    const cur = state.tilingStates;
    const prevArr = state.prevTilingStates || new Array(cur.length).fill(0);
    const conwayOfPrev = stepTiling(prevArr, state.tilingNeighbors, state.holes);
    const newPrev = conwayOfPrev.map((v, i) => v ^ cur[i]);
    state.tilingStates = prevArr.slice();
    state.prevTilingStates = newPrev;
  } else {
    const cur = state.grid;
    const prevArr = state.prevGrid || makeGrid(state.rows, state.cols, 0);
    const conwayOfPrev = state.gridType === "triangle"
      ? stepGridTri(prevArr, state.wrap, state.holes)
      : stepGrid(prevArr, state.wrap, state.holes);
    const newPrev = conwayOfPrev.map((row, r) => row.map((v, c) => v ^ cur[r][c]));
    state.grid = prevArr.map((row) => row.slice());
    state.prevGrid = newPrev;
  }
}

// Re-color living cells after a Fredkin step drives life/death, so an active
// color rule still "decorates" the board. Survivors keep their hue (rotating
// it under the rotation rule); newborns take the circular-mean hue of the
// neighbors that were alive a step ago, or a default seed hue if none.
function decorateColorsAfterFredkin(aliveBefore) {
  const n = cellCount();
  const aliveAfter = flatAlive();
  const neighborsOf = neighborsOfFn();
  const rotating = state.colorSettings.rule === RULESETS.ROTATION;
  const next = makeColorState(n);
  for (let i = 0; i < n; i += 1) {
    if (!aliveAfter[i]) continue; // dead cells stay cleared
    if (aliveBefore[i]) {
      const hue = rotating
        ? wrapHue(state.color.hue[i] + state.colorSettings.rotationDelta)
        : state.color.hue[i];
      next.hue[i] = hue;
      // Preserve the cell's own saturation (an achromatic start color should
      // stay achromatic) rather than forcing full saturation.
      next.sat[i] = state.color.sat[i];
      next.age[i] = Math.min(65535, state.color.age[i] + 1);
      next.origSat[i] = state.color.origSat[i];
    } else {
      const hues = [];
      const nbrs = neighborsOf(i);
      for (let k = 0; k < nbrs.length; k += 1) {
        if (aliveBefore[nbrs[k]]) hues.push(state.color.hue[nbrs[k]]);
      }
      const mean = circularMean(hues);
      const hue = mean == null ? defaultSeedHue() : mean;
      const sat = mean == null ? defaultSeedSat() : 1.0;
      next.hue[i] = wrapHue(hue);
      next.sat[i] = sat;
      next.age[i] = 0;
      next.origSat[i] = sat;
    }
  }
  state.color = next;
}

function stepBack() {
  if (state.secondOrder) {
    // Reversible: apply the inverse rule and ignore history. This keeps edits
    // intact and lets us run backward past gen 0. Fredkin drives life/death, so
    // this works with a color rule too — we just re-decorate the reconstructed
    // generation afterward (exact hues/saturation aren't recoverable, but the
    // alive/dead layer is).
    const beforeAlive = colorRuleActive() ? flatAlive() : null;
    inverseFredkinStep();
    if (beforeAlive) decorateColorsAfterFredkin(beforeAlive);
    state.generation -= 1;
    // History snapshots are tied to the forward timeline; diverging from it
    // via the inverse rule makes them stale for Back-from-here.
    state.history = [];
    syncLabels();
    return;
  }
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

function resizeHoles() {
  state.holes = new Uint8Array(cellCount());
}

function clearHoles() {
  state.holes.fill(0);
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

// Companion to defaultSeedHue(): the saturation to pair with it. Honors the
// user's chosen fixed-start saturation (0 = achromatic gray/white/black,
// depending on the Lightness setting) so Randomize/Invert/paste don't
// silently override a deliberately achromatic starting color with full sat.
function defaultSeedSat() {
  if (state.colorSettings.rule === RULESETS.ROTATION && state.colorSettings.rotationHueStart === "fixed") {
    return state.colorSettings.rotationFixedSat ?? 1.0;
  }
  return 1.0;
}

// Re-seed color for every currently-alive cell that lacks color (sat=0).
// Called on toggling a ruleset on, after randomize, after invert, etc.
function seedColorsForAllAlive() {
  const n = cellCount();
  for (let i = 0; i < n; i += 1) {
    if (isAliveAt(i)) {
      if (state.color.sat[i] <= 0 || state.color.origSat[i] <= 0) {
        seedColorAt(i, defaultSeedHue(), defaultSeedSat());
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
  // Snapshot prev-alive flat for deathsCreateHoles tracking.
  const prevFlat = state.colorSettings.deathsCreateHoles ? flatAlive() : null;

  // Track the previous generation continuously (regardless of the Fredkin
  // flag), so toggling Fredkin on mid-run immediately has a valid prior
  // generation to XOR against and to reverse from.
  const prevTilingCopy = isTiling() ? state.tilingStates.slice() : null;
  const prevGridCopy = !isTiling() ? state.grid.map((row) => row.slice()) : null;

  if (state.secondOrder) {
    // Fredkin drives life/death on the plain Conway layer (exactly like the
    // base rule), independent of any color rule. Next = Conway(cur) XOR prev.
    const aliveBefore = colorRuleActive() ? flatAlive() : null;
    if (isTiling()) {
      const nextStd = stepTiling(state.tilingStates, state.tilingNeighbors, state.holes);
      const prev = state.prevTilingStates;
      state.tilingStates = (prev && prev.length === nextStd.length)
        ? nextStd.map((v, i) => v ^ prev[i])
        : nextStd;
    } else {
      const nextStd = state.gridType === "triangle"
        ? stepGridTri(state.grid, state.wrap, state.holes)
        : stepGrid(state.grid, state.wrap, state.holes);
      const prev = state.prevGrid;
      state.grid = (prev && prev.length === nextStd.length && prev[0].length === nextStd[0].length)
        ? nextStd.map((row, r) => row.map((v, c) => v ^ prev[r][c]))
        : nextStd;
    }
    // Color rule (if any) only decorates the living cells.
    if (aliveBefore) decorateColorsAfterFredkin(aliveBefore);
  } else if (colorRuleActive()) {
    const alive = flatAlive();
    const { nextAlive, nextColor } = stepColor(
      alive,
      state.color,
      neighborsOfFn(),
      cellCount(),
      state.colorSettings,
      state.holes,
    );
    applyFlatAlive(nextAlive);
    state.color = nextColor;
  } else if (isTiling()) {
    state.tilingStates = stepTiling(state.tilingStates, state.tilingNeighbors, state.holes);
  } else if (state.gridType === "triangle") {
    state.grid = stepGridTri(state.grid, state.wrap, state.holes);
  } else {
    state.grid = stepGrid(state.grid, state.wrap, state.holes);
  }
  state.prevTilingStates = prevTilingCopy;
  state.prevGrid = prevGridCopy;
  // Apply deathsCreateHoles after the step.
  if (prevFlat) {
    const nextFlat = flatAlive();
    for (let i = 0; i < prevFlat.length; i += 1) {
      if (prevFlat[i] && !nextFlat[i]) state.holes[i] = 1;
    }
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
  // p.ew.pe/w aesthetic: always dark, regardless of OS theme.
  return false;
}

// Site palette (p.ew.pe/w): cyan-led on near-black.
function themeCanvas() {
  return {
    canvasBg: "#000000",
    squareDead: "#0e0220",
    gridStroke: "rgba(244, 244, 255, 0.07)",
    tilingStroke: "rgba(244, 244, 255, 0.16)",
    wall: "#000000",
  };
}

function gridColors() {
  const dead = "#0e0220";
  if (state.gridType === "rhombus") {
    return { alive: ["#00f0ff", "#aa55ff", "#007a8a"], dead };
  }
  if (state.gridType === "penrose") {
    return { alive: ["#00f0ff", "#aa55ff"], dead };
  }
  if (state.gridType === "einstein") {
    return { alive: ["#00f0ff", "#aa55ff", "#ff2aa0", "#00ff66"], dead };
  }
  if (state.gridType === "trihex") {
    return { alive: ["#00f0ff", "#ffaa00"], dead };
  }
  if (state.gridType === "oct") {
    return { alive: ["#00f0ff", "#aa55ff"], dead };
  }
  if (state.gridType === "voronoi") {
    return { alive: ["#00f0ff", "#00ff66", "#aa55ff"], dead };
  }
  return { alive: ["#00f0ff"], dead };
}

// Returns the fill color for a live cell at flat index `i`, honoring the
// active color ruleset and the colorsSwapped flag (which rotates hue 180°
// to its complement when a color ruleset is active).
function liveFillFor(i, fallbackHex) {
  if (!colorRuleActive()) {
    return fallbackHex;
  }
  let h = state.color.hue[i];
  // Zero saturation is a legitimate achromatic color (gray/white/black,
  // depending on the Lightness setting) — not a value to paper over.
  const s = Number.isFinite(state.color.sat[i]) ? state.color.sat[i] : 1.0;
  if (state.colorsSwapped) h = wrapHue(h + 180);
  return hslCss(h, s, state.colorSettings.lightness);
}

// Returns the fill color for any cell at flat index `i`. Decides between
// hole / live / ghost / dead automatically.
function cellFillFor(i, isAlive, theme, fallbackAliveHex, fallbackDeadHex) {
  if (state.holes[i]) return theme.wall;
  if (isAlive) return liveFillFor(i, fallbackAliveHex);
  // Dead cell — render fading ghost if any.
  if (colorRuleActive() && state.color.ghostFade && state.color.ghostFade[i] > 0) {
    let h = state.color.ghostHue[i];
    if (state.colorsSwapped) h = wrapHue(h + 180);
    const startSat = state.color.ghostStartSat[i];
    const total = state.color.ghostStart[i] || 1;
    const remaining = state.color.ghostFade[i];
    const sat = Math.max(0, Math.min(1, startSat * (remaining / total)));
    return hslCss(h, sat, state.colorSettings.lightness);
  }
  return fallbackDeadHex;
}

function drawSquareGrid(width, height) {
  const theme = themeCanvas();
  const { cell, ox, oy } = squareMetrics(width, height);
  const aliveDefault = state.colorsSwapped ? theme.squareDead : "#00f0ff";
  const dead = state.colorsSwapped && !colorRuleActive() ? "#00f0ff" : theme.squareDead;
  ctx.fillStyle = theme.canvasBg;
  ctx.fillRect(0, 0, width, height);
  for (let r = 0; r < state.rows; r += 1) {
    for (let c = 0; c < state.cols; c += 1) {
      const idx = r * state.cols + c;
      const alive = state.grid[r][c] === 1;
      ctx.fillStyle = cellFillFor(idx, alive, theme, aliveDefault, dead);
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
  const aliveDefault = state.colorsSwapped ? theme.squareDead : "#00f0ff";
  const dead = state.colorsSwapped && !colorRuleActive() ? "#00f0ff" : theme.squareDead;
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
      const idx = r * state.cols + c;
      const alive = state.grid[r][c] === 1;
      ctx.fillStyle = cellFillFor(idx, alive, theme, aliveDefault, dead);
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
    if (state.holes[index]) {
      fill = theme.wall;
    } else if (colorRuleActive()) {
      // Color rules drive alive fill; dead cells get ghost or dead color.
      fill = cellFillFor(index, isAlive, theme, aliveColor, palette.dead);
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
  elements.stepBack.disabled = !state.secondOrder && state.history.length === 0;
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
  elements.rotationFixedHue.value = hueToHex(state.colorSettings.rotationFixedHue, state.colorSettings.rotationFixedSat);
  elements.rotationDelta.value = String(state.colorSettings.rotationDelta);
  elements.deathFadeSteps.value = String(state.colorSettings.deathFadeSteps);
  elements.freezeHueDying.checked = !!state.colorSettings.freezeHueDying;
  elements.ghostFadeSteps.value = String(state.colorSettings.ghostFadeSteps);
  elements.blockGhostBirths.checked = !!state.colorSettings.blockGhostBirths;
  elements.deathsCreateHoles.checked = !!state.colorSettings.deathsCreateHoles;
}

function syncToolsUI() {
  if (!elements.toolsOptions) return;
  elements.toolsOptions.hidden = !state.toolsPanelOpen;
  elements.toggleToolsPanel.classList.toggle("active", state.toolsPanelOpen);
  elements.toolBrush.classList.toggle("active", state.tool === "brush");
  elements.toolHole.classList.toggle("active", state.tool === "hole");
  elements.toolSelect.classList.toggle("active", state.tool === "select");
  elements.brushSize.value = String(state.brushSize);
  elements.brushSizeValue.textContent = String(state.brushSize);
  const hasSelection = state.selection instanceof Set && state.selection.size > 0;
  elements.toolCopy.disabled = !hasSelection;
  elements.toolCut.disabled = !hasSelection;
  elements.toolPaste.disabled = !state.clipboard;
  // Paint-color picker only matters when a color rule is active.
  if (elements.paintColorField) {
    elements.paintColorField.hidden = !colorRuleActive();
  }
  // Update canvas cursor class
  canvas.classList.toggle("tool-select", state.tool === "select");
  canvas.classList.toggle("tool-paste", state.tool === "paste");
  canvas.classList.toggle("tool-hole", state.tool === "hole");
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
  syncToolsUI();
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

function applyPaint(target, value, hueOverride = null, satOverride = null) {
  if (target == null) {
    return;
  }
  const flat = flatIndexOf(target);
  setAliveAt(flat, value);
  if (colorRuleActive()) {
    if (value) {
      const hue = hueOverride != null ? hueOverride : defaultSeedHue();
      const sat = satOverride != null ? satOverride : defaultSeedSat();
      seedColorAt(flat, hue, sat);
    } else {
      clearCellColor(state.color, flat);
    }
  }
  // Painting at gen 0 redefines the "base" state, so keep gen0 snapshot fresh.
  if (state.generation === 0) {
    clearPrevState();
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
  // Fill white so images with alpha don't read as zero brightness.
  offCtx.fillStyle = "#ffffff";
  offCtx.fillRect(0, 0, size, size);
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
  clearPrevState();
  captureGenZero();
  syncLabels();
}

// ── Scene persistence via PNG tEXt chunk ─────────────────────────────────────

const SCENE_PNG_KEYWORD = "tntgol-scene";
const PNG_SIG = [137, 80, 78, 71, 13, 10, 26, 10];

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes, start, end) {
  let c = 0xffffffff;
  for (let i = start; i < end; i += 1) {
    c = (CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8)) >>> 0;
  }
  return (c ^ 0xffffffff) >>> 0;
}

function writeU32BE(out, offset, value) {
  out[offset] = (value >>> 24) & 0xff;
  out[offset + 1] = (value >>> 16) & 0xff;
  out[offset + 2] = (value >>> 8) & 0xff;
  out[offset + 3] = value & 0xff;
}

function readU32BE(bytes, offset) {
  return (bytes[offset] * 0x1000000)
    + ((bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3]);
}

function isPngSignature(bytes) {
  if (!bytes || bytes.length < 8) return false;
  for (let i = 0; i < 8; i += 1) {
    if (bytes[i] !== PNG_SIG[i]) return false;
  }
  return true;
}

function makeTextChunk(keyword, text) {
  const enc = new TextEncoder();
  const kw = enc.encode(keyword);
  const txt = enc.encode(text);
  const dataLen = kw.length + 1 + txt.length;
  const out = new Uint8Array(12 + dataLen);
  writeU32BE(out, 0, dataLen);
  out[4] = 0x74; out[5] = 0x45; out[6] = 0x58; out[7] = 0x74; // "tEXt"
  out.set(kw, 8);
  out[8 + kw.length] = 0;
  out.set(txt, 8 + kw.length + 1);
  writeU32BE(out, 8 + dataLen, crc32(out, 4, 8 + dataLen));
  return out;
}

function injectPngText(pngBytes, keyword, text) {
  if (!isPngSignature(pngBytes)) return pngBytes;
  // IHDR is always the first chunk: 4 length + 4 type + 13 data + 4 CRC = 25 bytes.
  const cut = 8 + 25;
  if (pngBytes.length < cut) return pngBytes;
  const chunk = makeTextChunk(keyword, text);
  const out = new Uint8Array(pngBytes.length + chunk.length);
  out.set(pngBytes.subarray(0, cut), 0);
  out.set(chunk, cut);
  out.set(pngBytes.subarray(cut), cut + chunk.length);
  return out;
}

function extractPngText(pngBytes, keyword) {
  if (!isPngSignature(pngBytes)) return null;
  const kwBytes = new TextEncoder().encode(keyword);
  const decoder = new TextDecoder();
  let offset = 8;
  while (offset + 12 <= pngBytes.length) {
    const len = readU32BE(pngBytes, offset);
    const type = String.fromCharCode(
      pngBytes[offset + 4], pngBytes[offset + 5],
      pngBytes[offset + 6], pngBytes[offset + 7],
    );
    const dataStart = offset + 8;
    if (dataStart + len + 4 > pngBytes.length) return null;
    if (type === "tEXt" && len >= kwBytes.length + 1) {
      let match = true;
      for (let k = 0; k < kwBytes.length; k += 1) {
        if (pngBytes[dataStart + k] !== kwBytes[k]) { match = false; break; }
      }
      if (match && pngBytes[dataStart + kwBytes.length] === 0) {
        return decoder.decode(pngBytes.subarray(dataStart + kwBytes.length + 1, dataStart + len));
      }
    }
    if (type === "IEND") return null;
    offset = dataStart + len + 4;
  }
  return null;
}

// ── Scene <-> JSON ───────────────────────────────────────────────────────────

function packBits(arr) {
  let s = "";
  for (let i = 0; i < arr.length; i += 1) s += arr[i] ? "1" : "0";
  return s;
}

function unpackBits(str) {
  const n = str.length;
  const a = new Array(n);
  for (let i = 0; i < n; i += 1) a[i] = str.charCodeAt(i) === 49 ? 1 : 0;
  return a;
}

function packGrid(grid) {
  return grid.map(packBits).join("");
}

function unpackGrid(str, rows, cols) {
  const g = new Array(rows);
  for (let r = 0; r < rows; r += 1) {
    const row = new Array(cols);
    const base = r * cols;
    for (let c = 0; c < cols; c += 1) row[c] = str.charCodeAt(base + c) === 49 ? 1 : 0;
    g[r] = row;
  }
  return g;
}

function captureSceneJSON() {
  const scene = {
    version: 1,
    gridType: state.gridType,
    rows: state.rows,
    cols: state.cols,
    wrap: state.wrap,
    voronoiSeed: state.voronoiSeed,
    voronoiMetric: state.voronoiMetric,
    voronoiJitter: state.voronoiJitter,
    secondOrder: state.secondOrder,
    colorsSwapped: state.colorsSwapped,
  };
  if (isTiling()) {
    scene.tilingStates = packBits(state.tilingStates);
    if (state.prevTilingStates) scene.prevTilingStates = packBits(state.prevTilingStates);
  } else {
    scene.grid = packGrid(state.grid);
    if (state.prevGrid) scene.prevGrid = packGrid(state.prevGrid);
  }
  return JSON.stringify(scene);
}

function applySceneJSON(json) {
  let scene;
  try { scene = JSON.parse(json); } catch (e) { return false; }
  if (!scene || typeof scene !== "object") return false;

  state.gridType = scene.gridType ?? state.gridType;
  state.rows = scene.rows ?? state.rows;
  state.cols = scene.cols ?? state.cols;
  state.wrap = scene.wrap ?? state.wrap;
  state.voronoiSeed = scene.voronoiSeed ?? state.voronoiSeed;
  state.voronoiMetric = scene.voronoiMetric ?? state.voronoiMetric;
  state.voronoiJitter = scene.voronoiJitter ?? state.voronoiJitter;
  state.secondOrder = !!scene.secondOrder;
  state.colorsSwapped = !!scene.colorsSwapped;

  elements.gridType.value = state.gridType;
  elements.rowsInput.value = String(state.rows);
  elements.colsInput.value = String(state.cols);
  elements.wrapToggle.checked = state.wrap;
  elements.voronoiJitter.value = String(Math.round(state.voronoiJitter * 100));
  elements.fredkinToggle.checked = state.secondOrder;

  rebuildTopology();

  if (isTiling()) {
    if (typeof scene.tilingStates === "string"
        && scene.tilingStates.length === state.tilingStates.length) {
      state.tilingStates = unpackBits(scene.tilingStates);
    }
    if (typeof scene.prevTilingStates === "string"
        && scene.prevTilingStates.length === state.tilingStates.length) {
      state.prevTilingStates = unpackBits(scene.prevTilingStates);
    }
  } else {
    const expected = state.rows * state.cols;
    if (typeof scene.grid === "string" && scene.grid.length === expected) {
      state.grid = unpackGrid(scene.grid, state.rows, state.cols);
    }
    if (typeof scene.prevGrid === "string" && scene.prevGrid.length === expected) {
      state.prevGrid = unpackGrid(scene.prevGrid, state.rows, state.cols);
    }
  }

  state.generation = 0;
  state.history = [];
  state.gen0Snapshot = snapshotCurrent();
  syncLabels();
  return true;
}

// ── Save / load ──────────────────────────────────────────────────────────────

async function saveCanvasSnapshot() {
  render();
  // Capture JSON in the same sync tick as render so the embedded scene
  // matches the pixels even if the simulation is playing.
  const sceneJson = captureSceneJSON();
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) return;
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const withScene = injectPngText(bytes, SCENE_PNG_KEYWORD, sceneJson);
  const outBlob = new Blob([withScene], { type: "image/png" });
  const url = URL.createObjectURL(outBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "tnt-gol-playground.png";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function handleImportFile(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const sceneText = extractPngText(bytes, SCENE_PNG_KEYWORD);
  if (sceneText && applySceneJSON(sceneText)) {
    return;
  }
  const image = await loadImageFromFile(file);
  applyImageToCurrentGeometry(image);
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
    state.tool = "brush";
    syncToolsUI();
    return;
  }
  if (state.tool === "hole") {
    // Hole tool — first cell's current hole state (inverted) becomes
    // the value applied for the rest of the drag.
    const target = cellAtPointer(pos);
    if (target == null) return;
    const flat = flatIndexOf(target);
    pointerState.painting = true;
    pointerState.holeDrag = true;
    pointerState.drawValue = state.holes[flat] ? 0 : 1;
    holeAtPointer(pos, pointerState.drawValue);
    return;
  }
  // brush — tap-to-toggle: first cell's current state (inverted) becomes
  // the value painted for the rest of the drag.
  const target = cellAtPointer(pos);
  if (target == null) return;
  const flat = flatIndexOf(target);
  pointerState.painting = true;
  pointerState.holeDrag = false;
  pointerState.drawValue = isAliveAt(flat) ? 0 : 1;
  brushAtPointer(pos, pointerState.drawValue);
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
  if (pointerState.holeDrag) {
    holeAtPointer(pos, pointerState.drawValue);
  } else {
    brushAtPointer(pos, pointerState.drawValue);
  }
}

// Apply the current brush at the given screen position. brushSize=1 paints
// only the cell under the cursor (preserving the original 1-cell behavior).
// Larger sizes paint a disc of cells of that diameter (in cell-units), found
// by checking each candidate cell's centroid against the cursor in screen
// coordinates and converting through the geometry's natural cell scale.
function brushAtPointer(pos, value) {
  const target = cellAtPointer(pos);
  if (target == null) return;
  // Brush paints with the user's paint color when a rule is active.
  const hue = colorRuleActive() ? state.paintHue : null;
  const sat = colorRuleActive() ? state.paintSat : null;
  if (state.brushSize <= 1) {
    const flat = flatIndexOf(target);
    if (state.holes[flat]) return;
    applyPaint(target, value, hue, sat);
    return;
  }
  const indices = brushCellIndices(pos);
  for (const idx of indices) {
    if (state.holes[idx]) continue;
    if (isAliveAt(idx) !== !!value) {
      applyPaintFlat(idx, value, hue, sat);
    }
  }
  if (state.generation === 0) {
    state.gen0Snapshot = snapshotCurrent();
    state.history = [];
  }
  syncLabels();
}

// Apply hole-toggle at the given screen position with the brush size.
// Setting a hole also kills any live cell at that index and clears its color
// state. Removing a hole leaves the cell dead.
function holeAtPointer(pos, value) {
  const target = cellAtPointer(pos);
  if (target == null) return;
  const baseFlat = flatIndexOf(target);
  const indices = state.brushSize <= 1 ? [baseFlat] : brushCellIndices(pos);
  for (const idx of indices) {
    if (state.holes[idx] === value) continue;
    state.holes[idx] = value;
    if (value) {
      // Becoming a hole — clear cell + color so the spot is truly empty.
      setAliveAt(idx, 0);
      if (state.color && state.color.hue) clearCellColor(state.color, idx);
    }
  }
  if (state.generation === 0) {
    state.gen0Snapshot = snapshotCurrent();
    state.history = [];
  }
  syncLabels();
}

// Centralized "set cell + seed/clear color" so brush + paint share a path.
// `applyPaint` keeps its own gen-0 snapshot bookkeeping for single-cell calls.
function applyPaintFlat(flat, value, hueOverride = null, satOverride = null) {
  setAliveAt(flat, value);
  if (colorRuleActive()) {
    if (value) {
      const hue = hueOverride != null ? hueOverride : defaultSeedHue();
      const sat = satOverride != null ? satOverride : defaultSeedSat();
      seedColorAt(flat, hue, sat);
    } else {
      clearCellColor(state.color, flat);
    }
  }
}

// Returns the set of flat cell indices covered by the current brush stroke
// at screen position `pos`. The brush is a disc of diameter `state.brushSize`
// in cell-units, where "cell-unit" is the natural cell size of the active
// geometry (cellSize for square; cellW for triangle; an estimate from the
// average polygon area for tilings).
function brushCellIndices(pos) {
  const rect = canvas.getBoundingClientRect();
  const out = new Set();
  const radiusCells = state.brushSize / 2;

  if (isTiling()) {
    const metrics = tilingMetrics(rect.width, rect.height);
    // Estimate average cell "radius" in world units from total bbox area / count.
    const [bw, bh] = state.tilingBBox;
    const avgArea = (bw * bh) / Math.max(1, state.polygons.length);
    const cellWorldRadius = Math.sqrt(avgArea) * 0.6;
    const radius = radiusCells * cellWorldRadius * metrics.scale;
    const r2 = radius * radius;
    state.polygons.forEach((poly, idx) => {
      const [cx, cy] = polygonCentroid(poly);
      const sx = metrics.ox + cx * metrics.scale;
      const sy = metrics.oy + cy * metrics.scale;
      const dx = sx - pos.x;
      const dy = sy - pos.y;
      if (dx * dx + dy * dy <= r2) out.add(idx);
    });
    return out;
  }

  if (state.gridType === "triangle") {
    const metrics = triMetrics(rect.width, rect.height);
    const radius = radiusCells * metrics.cellW;
    const r2 = radius * radius;
    // Bound the candidate range by approximate row/col window to avoid
    // scanning the whole grid.
    const approxR = clamp(Math.floor((pos.y - metrics.oy) / metrics.cellH), 0, state.rows - 1);
    const approxC = clamp(Math.floor((pos.x - metrics.ox) / (metrics.cellW / 2)), 0, state.cols - 1);
    const span = Math.max(2, state.brushSize + 2);
    for (let r = Math.max(0, approxR - span); r <= Math.min(state.rows - 1, approxR + span); r += 1) {
      for (let c = Math.max(0, approxC - span * 2); c <= Math.min(state.cols - 1, approxC + span * 2); c += 1) {
        const points = triCellPoints(r, c, metrics);
        const cx = (points[0][0] + points[1][0] + points[2][0]) / 3;
        const cy = (points[0][1] + points[1][1] + points[2][1]) / 3;
        const dx = cx - pos.x;
        const dy = cy - pos.y;
        if (dx * dx + dy * dy <= r2) out.add(r * state.cols + c);
      }
    }
    return out;
  }

  // square
  const { cell, ox, oy } = squareMetrics(rect.width, rect.height);
  const radius = radiusCells * cell;
  const r2 = radius * radius;
  const cCenter = (pos.x - ox) / cell;
  const rCenter = (pos.y - oy) / cell;
  const span = Math.ceil(radiusCells) + 1;
  const cMin = Math.max(0, Math.floor(cCenter - span));
  const cMax = Math.min(state.cols - 1, Math.ceil(cCenter + span));
  const rMin = Math.max(0, Math.floor(rCenter - span));
  const rMax = Math.min(state.rows - 1, Math.ceil(rCenter + span));
  for (let r = rMin; r <= rMax; r += 1) {
    for (let c = cMin; c <= cMax; c += 1) {
      const cx = ox + (c + 0.5) * cell;
      const cy = oy + (r + 0.5) * cell;
      const dx = cx - pos.x;
      const dy = cy - pos.y;
      if (dx * dx + dy * dy <= r2) out.add(r * state.cols + c);
    }
  }
  return out;
}

function stopPainting() {
  if (pointerState.selectingDrag) {
    finalizeSelection();
    pointerState.selectingDrag = false;
    state.selectionDrag = null;
  }
  pointerState.painting = false;
  pointerState.holeDrag = false;
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

// Snapshot the *live* cells inside the selection into the clipboard.
// Dead cells in the selected rectangle are ignored — pasting won't kill
// existing cells under the stamp; it only stamps the live shape.
// If `cut`, the live cells inside the selection are also erased.
function copySelection(cut) {
  if (!state.selection || state.selection.size === 0) return;
  // Filter to just the alive members of the selection.
  const liveIdxs = [];
  state.selection.forEach((idx) => {
    if (isAliveAt(idx)) liveIdxs.push(idx);
  });
  if (liveIdxs.length === 0) {
    // Nothing live to copy — clear clipboard so paste stays disabled.
    state.clipboard = null;
    syncToolsUI();
    return;
  }
  const items = [];
  const useColor = colorRuleActive();
  if (isTiling()) {
    const centroids = liveIdxs.map((idx) => polygonCentroid(state.polygons[idx]));
    const ox = Math.min(...centroids.map((p) => p[0]));
    const oy = Math.min(...centroids.map((p) => p[1]));
    liveIdxs.forEach((idx, k) => {
      const [cx, cy] = centroids[k];
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
    const rs = liveIdxs.map((idx) => Math.floor(idx / state.cols));
    const cs = liveIdxs.map((idx, k) => idx - rs[k] * state.cols);
    const r0 = Math.min(...rs);
    const c0 = Math.min(...cs);
    liveIdxs.forEach((idx, k) => {
      const item = { dr: rs[k] - r0, dc: cs[k] - c0 };
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
    liveIdxs.forEach((idx) => {
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
        else if (colorRuleActive()) seedColorAt(bestI, defaultSeedHue(), defaultSeedSat());
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
  elements.fredkinToggle.addEventListener("change", (event) => {
    state.secondOrder = event.target.checked;
    // prev is tracked continuously in stepOnce regardless of this flag, so
    // toggling just switches whether the XOR is applied — no state reset.
    syncLabels();
  });
  elements.invertGrid.addEventListener("click", invertState);
  elements.snapshotImage.addEventListener("click", saveCanvasSnapshot);
  elements.imageInput.addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    if (!file) return;
    try {
      await handleImportFile(file);
      render();
    } catch (error) {
      console.error(error);
    }
    // Clear so re-selecting the same file fires change again.
    event.target.value = "";
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
    const { h, s } = hexToHsl(event.target.value);
    state.colorSettings.rotationFixedHue = h;
    state.colorSettings.rotationFixedSat = s;
  });
  elements.rotationDelta.addEventListener("change", (event) => {
    const val = clamp(Number(event.target.value) || 0, -180, 180);
    state.colorSettings.rotationDelta = val;
    elements.rotationDelta.value = String(val);
  });
  elements.deathFadeSteps.addEventListener("change", (event) => {
    const val = clamp(Math.floor(Number(event.target.value) || 0), 0, 20);
    state.colorSettings.deathFadeSteps = val;
    elements.deathFadeSteps.value = String(val);
  });
  elements.freezeHueDying.addEventListener("change", (event) => {
    state.colorSettings.freezeHueDying = !!event.target.checked;
  });
  elements.ghostFadeSteps.addEventListener("change", (event) => {
    const val = clamp(Math.floor(Number(event.target.value) || 0), 0, 30);
    state.colorSettings.ghostFadeSteps = val;
    elements.ghostFadeSteps.value = String(val);
  });
  elements.blockGhostBirths.addEventListener("change", (event) => {
    state.colorSettings.blockGhostBirths = !!event.target.checked;
  });
  elements.deathsCreateHoles.addEventListener("change", (event) => {
    state.colorSettings.deathsCreateHoles = !!event.target.checked;
  });

  // ─── Tools panel ────────────────────────────────────────────────────────
  elements.toggleToolsPanel.addEventListener("click", () => {
    state.toolsPanelOpen = !state.toolsPanelOpen;
    syncToolsUI();
  });
  elements.toolBrush.addEventListener("click", () => { state.tool = "brush"; syncToolsUI(); });
  elements.toolHole.addEventListener("click", () => { state.tool = "hole"; syncToolsUI(); });
  elements.toolSelect.addEventListener("click", () => { state.tool = "select"; syncToolsUI(); });
  elements.brushSize.addEventListener("input", (event) => {
    state.brushSize = clamp(Number(event.target.value) || 1, 1, 10);
    elements.brushSizeValue.textContent = String(state.brushSize);
  });
  elements.paintColor.addEventListener("input", (event) => {
    const { h, s } = hexToHsl(event.target.value);
    state.paintHue = h;
    state.paintSat = s;
  });
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
  elements.rotationFixedHue.value = hueToHex(state.colorSettings.rotationFixedHue, state.colorSettings.rotationFixedSat);
  elements.rotationDelta.value = String(state.colorSettings.rotationDelta);
  elements.colorLightness.value = String(Math.round(state.colorSettings.lightness * 100));
  elements.fredkinToggle.checked = state.secondOrder;
  elements.brushSize.value = String(state.brushSize);
  elements.brushSizeValue.textContent = String(state.brushSize);
  elements.paintColor.value = hueToHex(state.paintHue, state.paintSat);
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
