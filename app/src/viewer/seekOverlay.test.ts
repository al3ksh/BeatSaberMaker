import { describe, expect, it } from "vitest";
import {
  SEEK_OVERLAY_DURATION_MS,
  accumulateSeekOverlay,
  formatSeekOverlayText,
} from "./seekOverlay";

describe("YouTube-style seek overlay accumulation", () => {
  it("accumulates repeated same-direction seeks within window and resets on expiry or direction flip", () => {
    const first = accumulateSeekOverlay(null, 1, 1000);
    expect(first.direction).toBe(1);
    expect(first.amountSec).toBe(5);
    expect(first.expiresAtMs).toBe(1000 + SEEK_OVERLAY_DURATION_MS);
    expect(formatSeekOverlayText(first.direction, first.amountSec)).toBe("5 seconds »");

    // Second forward press 250ms later -> 10 seconds »
    const second = accumulateSeekOverlay(first, 1, 1250);
    expect(second.amountSec).toBe(10);
    expect(second.expiresAtMs).toBe(1250 + SEEK_OVERLAY_DURATION_MS);
    expect(second.seq).toBeGreaterThan(first.seq);
    expect(formatSeekOverlayText(second.direction, second.amountSec)).toBe("10 seconds »");

    // Opposite direction while active resets to 5 in the new direction
    const flipped = accumulateSeekOverlay(second, -1, 1400);
    expect(flipped.direction).toBe(-1);
    expect(flipped.amountSec).toBe(5);
    expect(formatSeekOverlayText(flipped.direction, flipped.amountSec)).toBe("« 5 seconds");

    // Press after expiry resets back to 5
    const expired = accumulateSeekOverlay(flipped, -1, flipped.expiresAtMs + 10);
    expect(expired.amountSec).toBe(5);
    expect(expired.direction).toBe(-1);
  });
});
