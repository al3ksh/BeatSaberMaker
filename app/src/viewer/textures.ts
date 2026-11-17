import * as THREE from "three";

/** Generate a smooth radial Gaussian glow texture in memory (no external asset files). */
export function createSoftGlowTexture(size = 64): THREE.DataTexture {
  const data = new Uint8Array(size * size * 4);
  const half = (size - 1) * 0.5;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const dx = (x - half) / half;
      const dy = (y - half) / half;
      const r = Math.hypot(dx, dy);
      const falloff = Math.max(0, 1 - r);
      const intensity = falloff * falloff * falloff;
      const idx = (y * size + x) * 4;
      data[idx] = 255;
      data[idx + 1] = 255;
      data[idx + 2] = 255;
      data[idx + 3] = Math.round(intensity * 255);
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.needsUpdate = true;
  return tex;
}

/**
 * 64x1 alpha profile for camera-facing light ribbons (arcs): narrow bright core + soft falloff to
 * transparent edges. Sample with U across the ribbon width.
 */
export function createRibbonProfileTexture(width = 64): THREE.DataTexture {
  const data = new Uint8Array(width * 4);
  for (let i = 0; i < width; i += 1) {
    const x = (i / (width - 1)) * 2 - 1;
    const core = Math.exp(-((x / 0.2) ** 2));
    const halo = 0.32 * Math.exp(-((x / 0.62) ** 2));
    const edge = Math.max(0, 1 - x * x);
    const a = Math.min(1, (core + halo) * edge);
    data[i * 4] = 255;
    data[i * 4 + 1] = 255;
    data[i * 4 + 2] = 255;
    data[i * 4 + 3] = Math.round(a * 255);
  }
  const tex = new THREE.DataTexture(data, width, 1, THREE.RGBAFormat);
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}
