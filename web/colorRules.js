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
  };
}

export function makeColorState(n) {
  return {
    hue: new Float32Array(n),
    sat: new Float32Array(n),
    age: new Uint16Array(n),
    origSat: new Float32Array(n),
  };
}

export function cloneColorState(c) {
  return {
    hue: new Float32Array(c.hue),
    sat: new Float32Array(c.sat),
    age: new Uint16Array(c.age),
    origSat: new Float32Array(c.origSat),
  };
}

export function clearColorState(c) {
  c.hue.fill(0);
  c.sat.fill(0);
  c.age.fill(0);
  c.origSat.fill(0);
}

export function clearCellColor(c, i) {
  c.hue[i] = 0;
  c.sat[i] = 0;
  c.age[i] = 0;
  c.origSat[i] = 0;
}

export function setCellColor(c, i, hue, sat) {
  c.hue[i] = wrapHue(hue);
  c.sat[i] = sat;
  c.age[i] = 0;
  c.origSat[i] = sat;
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
//   color       : { hue, sat, age, origSat } typed arrays length n
//   neighborsOf : function(i) -> array of neighbor flat indices
//   n           : cell count
//   settings    : color settings (see defaultColorSettings)
export function stepColor(alive, color, neighborsOf, n, settings) {
  const nextAlive = new Uint8Array(n);
  const nextColor = makeColorState(n);
  const rule = settings.rule;

  for (let i = 0; i < n; i += 1) {
    const wasAlive = alive[i] === 1;
    const nbrs = neighborsOf(i);
    let liveCount = 0;
    for (let k = 0; k < nbrs.length; k += 1) {
      if (alive[nbrs[k]]) liveCount += 1;
    }
    const standardSurvive = wasAlive && (liveCount === 2 || liveCount === 3);
    const standardBirth = !wasAlive && liveCount === 3;

    if (rule === RULESETS.GOETHEAN) {
      if (wasAlive) {
        const decayed = color.sat[i] - color.origSat[i] / 10;
        if (standardSurvive && decayed > 0) {
          nextAlive[i] = 1;
          nextColor.hue[i] = color.hue[i];
          nextColor.sat[i] = decayed;
          nextColor.age[i] = Math.min(65535, color.age[i] + 1);
          nextColor.origSat[i] = color.origSat[i];
        }
      } else if (standardBirth) {
        // Eligible parents: live + sat >= origSat/2
        const eligibleHues = [];
        const eligibleSats = [];
        let warmCount = 0;
        let coolCount = 0;
        for (let k = 0; k < nbrs.length; k += 1) {
          const j = nbrs[k];
          if (!alive[j]) continue;
          if (color.sat[j] < color.origSat[j] / 2) continue;
          eligibleHues.push(color.hue[j]);
          eligibleSats.push(color.sat[j]);
          if (isWarm(color.hue[j])) warmCount += 1;
          else coolCount += 1;
        }
        if (warmCount >= 1 && coolCount >= 1) {
          const mean = circularMean(eligibleHues);
          const newHue = mean == null ? eligibleHues[0] : mean;
          const startSat = settings.goetheanSatStart === "inherit"
            ? eligibleSats.reduce((s, v) => s + v, 0) / eligibleSats.length
            : 1.0;
          nextAlive[i] = 1;
          nextColor.hue[i] = wrapHue(newHue);
          nextColor.sat[i] = startSat;
          nextColor.age[i] = 0;
          nextColor.origSat[i] = startSat;
        }
      }
    } else if (rule === RULESETS.ROTATION) {
      if (standardSurvive) {
        nextAlive[i] = 1;
        nextColor.hue[i] = wrapHue(color.hue[i] + settings.rotationDelta);
        nextColor.sat[i] = 1.0;
        nextColor.age[i] = Math.min(65535, color.age[i] + 1);
        nextColor.origSat[i] = 1.0;
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
