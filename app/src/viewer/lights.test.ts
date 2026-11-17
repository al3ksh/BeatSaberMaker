import { describe, expect, it } from "vitest";
import { buildLightTimeline, evaluateLightsAtTime } from "./lights";
import type { ViewerLightEvent } from "./types";

describe("viewer lights", () => {
  it("evaluates on/flash/fade/transition/off and restores state when scrubbing backward", () => {
    const events: ViewerLightEvent[] = [
      { beat: 1, timeS: 1.0, type: 0, value: 1, floatValue: 0.8 }, // et0 blue on (0.8)
      { beat: 2, timeS: 2.0, type: 0, value: 7, floatValue: 1.0 }, // et0 red fade
      { beat: 6, timeS: 6.0, type: 0, value: 0, floatValue: 1.0 }, // et0 off
      { beat: 2, timeS: 2.0, type: 4, value: 10, floatValue: 0.5 }, // et4 white flash
      { beat: 4, timeS: 4.0, type: 1, value: 4, floatValue: 1.0 }, // et1 blue transition
    ];

    const timeline = buildLightTimeline(events);

    // Before any event
    const t0 = evaluateLightsAtTime(timeline, 0.5);
    expect(t0.groups[0].intensity).toBe(0);

    // At t=1.2 -> et0 is blue on at 0.8
    const t1 = evaluateLightsAtTime(timeline, 1.2);
    expect(t1.groups[0].color).toBe("blue");
    expect(t1.groups[0].intensity).toBeCloseTo(0.8, 5);

    // At t=2.05 -> et0 red fade is bright; et4 white flash is boosted
    const t2 = evaluateLightsAtTime(timeline, 2.05);
    expect(t2.groups[0].color).toBe("red");
    expect(t2.groups[0].intensity).toBeGreaterThan(0.9);
    expect(t2.groups[4].color).toBe("white");
    expect(t2.groups[4].intensity).toBeGreaterThan(0.5);

    // At t=3.5 -> et0 red fade has decayed to 0; et4 white flash has settled to 0.5
    const t3 = evaluateLightsAtTime(timeline, 3.5);
    expect(t3.groups[0].intensity).toBeCloseTo(0, 5);
    expect(t3.groups[4].intensity).toBeCloseTo(0.5, 5);

    // At t=4.25 -> et1 transition is halfway from 0 to 1.0
    const t4 = evaluateLightsAtTime(timeline, 4.25);
    expect(t4.groups[1].color).toBe("blue");
    expect(t4.groups[1].intensity).toBeGreaterThan(0.3);
    expect(t4.groups[1].intensity).toBeLessThan(0.8);

    // Scrub forward to t=7 (off), then scrub backward to t=1.2 -> exact match!
    const t7 = evaluateLightsAtTime(timeline, 7.0);
    expect(t7.groups[0].intensity).toBe(0);
    const tBack = evaluateLightsAtTime(timeline, 1.2);
    expect(tBack.groups[0]).toEqual(t1.groups[0]);
  });

  it("computes ring spin/zoom and laser angle from events 8, 9, 12, 13", () => {
    const timeline = buildLightTimeline([
      { beat: 1, timeS: 1.0, type: 8, value: 1, floatValue: 1 },
      { beat: 2, timeS: 2.0, type: 9, value: 1, floatValue: 1 },
      { beat: 1, timeS: 1.0, type: 12, value: 4, floatValue: 1 },
      { beat: 1, timeS: 1.0, type: 13, value: 2, floatValue: 1 },
    ]);

    const s = evaluateLightsAtTime(timeline, 3.0);
    expect(s.ringRotation).not.toBe(0);
    expect(s.ringSpacing).not.toBe(0);
    expect(s.leftLaserPhase).toBeGreaterThan(s.rightLaserPhase);
  });
});
