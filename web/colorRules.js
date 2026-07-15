// ─── Color rulesets for Game of Life ────────────────────────────────────────
//
// Two rulesets layered on top of standard Conway B3/S23 dynamics. Both are
// geometry-agnostic — they receive a flat alive array, per-cell color state,
// and a neighbor lookup function, and return the next-generation alive +
// color state. The Conway count is computed in this module so the ruleset
// can override or gate births/survivals.
//
// Goethean polarity (RULESETS.GOETHEAN)
//   Each cell carries hue/saturation. Hue 0..180 is "warm", 180..360 is
//   "cool". A dead cell can only be born if it has at least one eligible
//   warm neighbor AND at least one eligible cool neighbor — eligible means
//   `saturation >= originalSaturation/2`. Newborn hue is the circular mean
//   of eligible parents. Each surviving cell loses `originalSaturation/10`
//   of saturation per generation, so a cell with full original saturation
//   has a soft 10-generation lifespan even if Conway would let it survive.
//
// Generational rotation (RULESETS.ROTATION)
//   Conway dynamics determine alive/dead with no extra gating. Surviving
//   cells rotate their hue by `rotationDelta` degrees per generation.
//   Newborn hue is either a fixed start hue or the circular mean of all
//   live parents, depending on settings.
//
// Hue averaging is *circular*: hues are converted to unit vectors before
// averaging, so 350° and 10° average to 0° (red), not 180° (cyan).
// ─────────────────────────────────────────────────────────────────────────

export const RULESETS = Object.freeze({
  NONE: "none",
  GOETHEAN: "goethean",
  ROTATION: "rotation",
});

export function defaultColorSettings() {
  return {
    rule: RULESETS.NONE,
    goetheanSatStart: "full", // 'full' | 'inherit'
    rotationHueStart: "fixed", // 'fixed' | 'parent'
    rotationFixedHue: 0, // degrees
    rotationDelta: 15, // degrees per generation
    lightness: 0.5, // 0..1
    // Death fade. 0 = immediate death (default Conway behavior). N>0 means a
    // cell that Conway would kill instead enters an N-step fade phase: each
    // generation it loses fadeStartSat/N of saturation. After N steps, dies.
    deathFadeSteps: 0,
    // When true, cells in the fade phase freeze their hue (they don't rotate
    // under RULESETS.ROTATION). They still lose saturation.
    freezeHueDying: false,
    // Ghosts: after a cell dies, its grid position holds a fading colored
    // "ghost" for `ghostFadeSteps` generations. Ghosts don't count toward
    // population and don't participate in Conway's neighbor count. Setting
    // 0 disables ghosting.
    ghostFadeSteps: 0,
    // When true, dead cells with an active ghost cannot birth — the spot is
    // blocked until the ghost fully fades.
    blockGhostBirths: false,
    // When true, every cell that dies turns its position into a hole
    // (permanent wall — see state.holes). Independent of color rule.
    deathsCreateHoles: false,
  };
}

export function makeColorState(n) {
  return {
    hue: new Float32Array(n),
    sat: new Float32Array(n),
    age: new Uint16Array(n),
    origSat: new Float32Array(n),
    // Fade tracking (live but dying — keeps contributing to population).
    fadeRemaining: new Uint8Array(n),
    fadeStart: new Uint8Array(n),
    fadeStartSat: new Float32Array(n),
    // Ghost tracking (cell is dead but spot still has a fading color).
    // ghostFade[i] > 0 means an active ghost; current ghost saturation is
    // ghostStartSat[i] * ghostFade[i] / ghostStart[i].
    ghostFade: new Uint8Array(n),
    ghostStart: new Uint8Array(n),
    ghostHue: new Float32Array(n),
    ghostStartSat: new Float32Array(n),
  };
}

export function cloneColorState(c) {
  const n = c.hue.length;
  return {
    hue: new Float32Array(c.hue),
    sat: new Float32Array(c.sat),
    age: new Uint16Array(c.age),
    origSat: new Float32Array(c.origSat),
    fadeRemaining: new Uint8Array(c.fadeRemaining || n),
    fadeStart: new Uint8Array(c.fadeStart || n),
    fadeStartSat: new Float32Array(c.fadeStartSat || n),
    ghostFade: new Uint8Array(c.ghostFade || n),
    ghostStart: new Uint8Array(c.ghostStart || n),
    ghostHue: new Float32Array(c.ghostHue || n),
    ghostStartSat: new Float32Array(c.ghostStartSat || n),
  };
}

export function clearColorState(c) {
  c.hue.fill(0);
  c.sat.fill(0);
  c.age.fill(0);
  c.origSat.fill(0);
  if (c.fadeRemaining) c.fadeRemaining.fill(0);
  if (c.fadeStart) c.fadeStart.fill(0);
  if (c.fadeStartSat) c.fadeStartSat.fill(0);
  if (c.ghostFade) c.ghostFade.fill(0);
  if (c.ghostStart) c.ghostStart.fill(0);
  if (c.ghostHue) c.ghostHue.fill(0);
  if (c.ghostStartSat) c.ghostStartSat.fill(0);
}

export function clearCellColor(c, i) {
  c.hue[i] = 0;
  c.sat[i] = 0;
  c.age[i] = 0;
  c.origSat[i] = 0;
  if (c.fadeRemaining) c.fadeRemaining[i] = 0;
  if (c.fadeStart) c.fadeStart[i] = 0;
  if (c.fadeStartSat) c.fadeStartSat[i] = 0;
  if (c.ghostFade) c.ghostFade[i] = 0;
  if (c.ghostStart) c.ghostStart[i] = 0;
  if (c.ghostHue) c.ghostHue[i] = 0;
  if (c.ghostStartSat) c.ghostStartSat[i] = 0;
}

export function setCellColor(c, i, hue, sat) {
  c.hue[i] = wrapHue(hue);
  c.sat[i] = sat;
  c.age[i] = 0;
  c.origSat[i] = sat;
  if (c.fadeRemaining) c.fadeRemaining[i] = 0;
  if (c.fadeStart) c.fadeStart[i] = 0;
  if (c.fadeStartSat) c.fadeStartSat[i] = 0;
  if (c.ghostFade) c.ghostFade[i] = 0;
  if (c.ghostStart) c.ghostStart[i] = 0;
  if (c.ghostHue) c.ghostHue[i] = 0;
  if (c.ghostStartSat) c.ghostStartSat[i] = 0;
}

export function wrapHue(h) {
  let v = h % 360;
  if (v < 0) v += 360;
  // Snap values that are essentially 360° to 0° for cleanliness.
  if (v > 360 - 1e-9) v = 0;
  return v;
}

export function isWarm(hue) {
  const h = wrapHue(hue);
  return h < 180;
}

// Circular mean of an iterable of hues (degrees). Returns 0..360.
// Returns null for an empty input or a perfectly-balanced antipodal pair.
export function circularMean(huesDeg) {
  let sx = 0;
  let sy = 0;
  let count = 0;
  for (const h of huesDeg) {
    const r = (h * Math.PI) / 180;
    sx += Math.cos(r);
    sy += Math.sin(r);
    count += 1;
  }
  if (count === 0) return null;
  if (Math.abs(sx) < 1e-9 && Math.abs(sy) < 1e-9) {
    // Antipodal cancellation — fall back to first hue.
    return null;
  }
  let mean = (Math.atan2(sy, sx) * 180) / Math.PI;
  return wrapHue(mean);
}

// Step under an active color ruleset. Returns { nextAlive, nextColor }.
//
//   alive       : Uint8Array length n, 0/1
//   color       : color state typed arrays length n
//   neighborsOf : function(i) -> array of neighbor flat indices
//   n           : cell count
//   settings    : color settings (see defaultColorSettings)
//   holes       : optional Uint8Array length n. holes[i] = 1 means cell is a
//                 wall — stays dead, doesn't contribute to neighbor counts.
export function stepColor(alive, color, neighborsOf, n, settings, holes) {
  const nextAlive = new Uint8Array(n);
  const nextColor = makeColorState(n);
  const rule = settings.rule;
  const fadeSteps = Math.max(0, Math.floor(settings.deathFadeSteps || 0));
  const freezeHueDying = !!settings.freezeHueDying;
  const ghostSteps = Math.max(0, Math.floor(settings.ghostFadeSteps || 0));
  const blockGhostBirths = !!settings.blockGhostBirths;

  // Spawn a fresh ghost in nextColor at index `i` based on the just-died
  // cell's prior color.
  function spawnGhost(i, hue, sat) {
    if (ghostSteps <= 0 || sat <= 0) return;
    nextColor.ghostFade[i] = ghostSteps;
    nextColor.ghostStart[i] = ghostSteps;
    nextColor.ghostHue[i] = wrapHue(hue);
    nextColor.ghostStartSat[i] = sat;
  }

  // Carry forward an existing ghost minus one step. Returns true if the
  // ghost is still active afterward.
  function decayGhost(i) {
    const remaining = (color.ghostFade ? color.ghostFade[i] : 0) - 1;
    if (remaining <= 0) return false;
    nextColor.ghostFade[i] = remaining;
    nextColor.ghostStart[i] = color.ghostStart ? color.ghostStart[i] : ghostSteps;
    nextColor.ghostHue[i] = color.ghostHue ? color.ghostHue[i] : 0;
    nextColor.ghostStartSat[i] = color.ghostStartSat ? color.ghostStartSat[i] : 0;
    return true;
  }

  // Helper to enter or continue the fade phase for cell i.
  // Reads from prev `color`, writes to `nextColor`. Returns true if cell is
  // still alive after this step (i.e., still fading), false if dead.
  function applyFade(i, hueIfAlive, freezeHue) {
    const prevRemaining = color.fadeRemaining ? color.fadeRemaining[i] : 0;
    let total;
    let startSat;
    let nextRemaining;
    if (prevRemaining === 0) {
      // Just entered fade.
      total = fadeSteps;
      startSat = color.sat[i];
      nextRemaining = total - 1;
    } else {
      total = color.fadeStart[i] || fadeSteps;
      startSat = color.fadeStartSat[i] || color.sat[i];
      nextRemaining = prevRemaining - 1;
    }
    if (nextRemaining <= 0) {
      // Last fade step → dies now.
      return false;
    }
    nextAlive[i] = 1;
    nextColor.hue[i] = wrapHue(freezeHue ? color.hue[i] : hueIfAlive);
    nextColor.sat[i] = startSat * (nextRemaining / total);
    nextColor.age[i] = Math.min(65535, color.age[i] + 1);
    nextColor.origSat[i] = color.origSat[i];
    nextColor.fadeRemaining[i] = nextRemaining;
    nextColor.fadeStart[i] = total;
    nextColor.fadeStartSat[i] = startSat;
    return true;
  }

  for (let i = 0; i < n; i += 1) {
    if (holes && holes[i]) {
      // Holes are walls — stay dead, no color, no ghost.
      continue;
    }
    const wasAlive = alive[i] === 1;
    const isDying = wasAlive && color.fadeRemaining && color.fadeRemaining[i] > 0;
    const hasGhost = color.ghostFade && color.ghostFade[i] > 0;
    const nbrs = neighborsOf(i);
    let liveCount = 0;
    for (let k = 0; k < nbrs.length; k += 1) {
      const j = nbrs[k];
      if (holes && holes[j]) continue;
      if (alive[j]) liveCount += 1;
    }
    const standardSurvive = wasAlive && (liveCount === 2 || liveCount === 3);
    let standardBirth = !wasAlive && liveCount === 3;
    if (standardBirth && blockGhostBirths && hasGhost) {
      standardBirth = false;
    }

    if (rule === RULESETS.GOETHEAN) {
      if (wasAlive) {
        // Already-fading cells continue fading regardless of Conway's mood.
        if (isDying) {
          if (!applyFade(i, color.hue[i], freezeHueDying)) {
            // Fade completed → cell dies. Spawn ghost from current state.
            spawnGhost(i, color.hue[i], color.sat[i]);
          }
          continue;
        }
        const decayed = color.sat[i] - color.origSat[i] / 10;
        if (standardSurvive && decayed > 0) {
          nextAlive[i] = 1;
          nextColor.hue[i] = color.hue[i];
          nextColor.sat[i] = decayed;
          nextColor.age[i] = Math.min(65535, color.age[i] + 1);
          nextColor.origSat[i] = color.origSat[i];
        } else if (fadeSteps > 0) {
          // Conway-death OR Goethean sat-depletion → enter fade.
          if (!applyFade(i, color.hue[i], freezeHueDying)) {
            spawnGhost(i, color.hue[i], color.sat[i]);
          }
        } else {
          // Immediate death → spawn ghost (if enabled) from current color.
          spawnGhost(i, color.hue[i], color.sat[i]);
        }
      } else if (standardBirth) {
        // Births follow standard Conway B3 — any 3 live neighbors → birth.
        // Color of newborn is the circular mean of *eligible* parents
        // (sat ≥ origSat/2). If no parents are eligible (all dying), fall
        // back to the mean of all live neighbors so the cell still gets a
        // sensible hue.
        const eligibleHues = [];
        const eligibleSats = [];
        const fallbackHues = [];
        const fallbackSats = [];
        for (let k = 0; k < nbrs.length; k += 1) {
          const j = nbrs[k];
          if (!alive[j]) continue;
          fallbackHues.push(color.hue[j]);
          fallbackSats.push(color.sat[j]);
          if (color.sat[j] < color.origSat[j] / 2) continue;
          eligibleHues.push(color.hue[j]);
          eligibleSats.push(color.sat[j]);
        }
        const parentHues = eligibleHues.length > 0 ? eligibleHues : fallbackHues;
        const parentSats = eligibleSats.length > 0 ? eligibleSats : fallbackSats;
        const mean = circularMean(parentHues);
        const newHue = mean == null ? (parentHues[0] || 0) : mean;
        const startSat = settings.goetheanSatStart === "inherit"
          ? parentSats.reduce((s, v) => s + v, 0) / parentSats.length
          : 1.0;
        nextAlive[i] = 1;
        nextColor.hue[i] = wrapHue(newHue);
        nextColor.sat[i] = startSat;
        nextColor.age[i] = 0;
        nextColor.origSat[i] = startSat;
      }
    } else if (rule === RULESETS.ROTATION) {
      if (wasAlive) {
        if (isDying) {
          if (!applyFade(i, color.hue[i] + settings.rotationDelta, freezeHueDying)) {
            spawnGhost(i, color.hue[i], color.sat[i]);
          }
          continue;
        }
        if (standardSurvive) {
          nextAlive[i] = 1;
          nextColor.hue[i] = wrapHue(color.hue[i] + settings.rotationDelta);
          nextColor.sat[i] = 1.0;
          nextColor.age[i] = Math.min(65535, color.age[i] + 1);
          nextColor.origSat[i] = 1.0;
        } else if (fadeSteps > 0) {
          if (!applyFade(i, color.hue[i] + settings.rotationDelta, freezeHueDying)) {
            spawnGhost(i, color.hue[i], color.sat[i]);
          }
        } else {
          spawnGhost(i, color.hue[i], color.sat[i]);
        }
      } else if (standardBirth) {
        let h;
        if (settings.rotationHueStart === "parent") {
          const liveHues = [];
          for (let k = 0; k < nbrs.length; k += 1) {
            const j = nbrs[k];
            if (alive[j]) liveHues.push(color.hue[j]);
          }
          const mean = circularMean(liveHues);
          h = mean == null ? settings.rotationFixedHue : mean;
        } else {
          h = settings.rotationFixedHue;
        }
        nextAlive[i] = 1;
        nextColor.hue[i] = wrapHue(h);
        nextColor.sat[i] = 1.0;
        nextColor.age[i] = 0;
        nextColor.origSat[i] = 1.0;
      }
    }
  }

  // Ghost-decay finalization: any cell still dead at this point that has
  // an active ghost from the previous generation gets its ghost ticked
  // down. Cells that just died this step already had spawnGhost called.
  for (let i = 0; i < n; i += 1) {
    if (holes && holes[i]) continue;
    if (nextAlive[i]) continue;
    if (nextColor.ghostFade[i] !== 0) continue; // freshly spawned this step
    decayGhost(i);
  }

  return { nextAlive, nextColor };
}

// Convert a CSS hex color string ("#rrggbb") to {h, s, l} (degrees / 0..1).
export function hexToHsl(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return { h: 0, s: 1, l: 0.5 };
  const r = parseInt(m[1].slice(0, 2), 16) / 255;
  const g = parseInt(m[1].slice(2, 4), 16) / 255;
  const b = parseInt(m[1].slice(4, 6), 16) / 255;
  return rgbToHsl(r, g, b);
}

export function rgbToHsl(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h = 0;
  let s = 0;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) * 60;
    else if (max === g) h = ((b - r) / d + 2) * 60;
    else h = ((r - g) / d + 4) * 60;
  }
  return { h, s, l };
}

export function hslCss(h, s, l) {
  return `hsl(${h.toFixed(1)}, ${(s * 100).toFixed(1)}%, ${(l * 100).toFixed(1)}%)`;
}

// Convert HSL hue to a 6-digit hex string at full saturation, 50% lightness.
// Used for the rotation start-hue color picker.
export function hueToHex(h) {
  const c = 1;
  const x = 1 - Math.abs(((h / 60) % 2) - 1);
  let r = 0;
  let g = 0;
  let b = 0;
  const seg = Math.floor(((h % 360) + 360) % 360 / 60);
  if (seg === 0) [r, g, b] = [c, x, 0];
  else if (seg === 1) [r, g, b] = [x, c, 0];
  else if (seg === 2) [r, g, b] = [0, c, x];
  else if (seg === 3) [r, g, b] = [0, x, c];
  else if (seg === 4) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  // 50% lightness: scale by 0.5 then add 0
  const m = 0.5 - c / 2;
  const toHex = (v) => Math.round((v + m) * 255).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}
