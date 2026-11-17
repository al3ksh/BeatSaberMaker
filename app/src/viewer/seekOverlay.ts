export const SEEK_STEP_SEC = 5;
export const SEEK_OVERLAY_DURATION_MS = 600;

export interface SeekOverlayState {
  direction: -1 | 1;
  amountSec: number;
  seq: number;
  expiresAtMs: number;
}

/** Accumulate repeated ±5s seeks in the same direction within the 0.6s overlay window. */
export function accumulateSeekOverlay(
  prev: SeekOverlayState | null,
  direction: -1 | 1,
  nowMs: number,
): SeekOverlayState {
  const sameActive = prev !== null && prev.direction === direction && nowMs < prev.expiresAtMs;
  const amountSec = sameActive ? prev.amountSec + SEEK_STEP_SEC : SEEK_STEP_SEC;
  return {
    direction,
    amountSec,
    seq: (prev?.seq ?? 0) + 1,
    expiresAtMs: nowMs + SEEK_OVERLAY_DURATION_MS,
  };
}

export function formatSeekOverlayText(direction: -1 | 1, amountSec: number): string {
  return direction < 0 ? `« ${amountSec} seconds` : `${amountSec} seconds »`;
}
