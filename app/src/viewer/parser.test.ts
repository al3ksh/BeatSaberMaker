import { describe, expect, it } from "vitest";
import { computeLocalNps, parseDifficultyData, queryWindow } from "./parser";
import { buildTempoMap } from "./timing";

describe("viewer parser", () => {
  it("parses v3 difficulty objects sorted by time and queries window efficiently", () => {
    const tempo = buildTempoMap(120, 0, []);
    const parsed = parseDifficultyData(
      {
        version: "3.3.0",
        colorNotes: [
          { b: 4, x: 2, y: 1, c: 1, d: 0, a: 15 },
          { b: 2, x: 1, y: 0, c: 0, d: 1, a: 0 },
        ],
        bombNotes: [{ b: 3, x: 0, y: 0 }],
        obstacles: [{ b: 1, x: 0, y: 0, d: 2, w: 1, h: 5 }],
        sliders: [
          { b: 2, c: 0, x: 1, y: 0, d: 1, mu: 1, tb: 4, tx: 2, ty: 1, tc: 0, tmu: 1, m: 0 },
        ],
        burstSliders: [
          { b: 5, x: 1, y: 0, c: 0, d: 2, tb: 5.5, tx: 0, ty: 0, sc: 5, s: 0.8 },
        ],
        basicBeatmapEvents: [
          { b: 2, et: 1, i: 3, f: 1 },
          { b: 0, et: 0, i: 5, f: 0.75 },
        ],
      },
      tempo,
    );

    expect(parsed.notes).toHaveLength(2);
    expect(parsed.notes[0].beat).toBe(2);
    expect(parsed.notes[0].timeS).toBeCloseTo(1.0, 6);
    expect(parsed.notes[1].beat).toBe(4);
    expect(parsed.notes[1].timeS).toBeCloseTo(2.0, 6);
    expect(parsed.notes[1].angleOffset).toBe(15);

    expect(parsed.bombs[0].timeS).toBeCloseTo(1.5, 6);
    expect(parsed.obstacles[0].endTimeS).toBeCloseTo(1.5, 6);
    expect(parsed.arcs[0].tailTimeS).toBeCloseTo(2.0, 6);
    expect(parsed.chains[0].segments).toBe(5);
    expect(parsed.lightEvents[0].type).toBe(0);

    const win = queryWindow(parsed.notes, 0.8, 1.2);
    expect(win).toHaveLength(1);
    expect(win[0].beat).toBe(2);

    expect(computeLocalNps(parsed.notes, 1.0)).toBe(1);
    expect(computeLocalNps(parsed.notes, 1.5)).toBe(2);
  });

  it("also handles v2 difficulty schemas gracefully", () => {
    const tempo = buildTempoMap(120, 0, []);
    const parsed = parseDifficultyData(
      {
        _version: "2.2.0",
        _notes: [
          { _time: 2, _lineIndex: 1, _lineLayer: 0, _type: 0, _cutDirection: 1 },
          { _time: 3, _lineIndex: 2, _lineLayer: 0, _type: 3, _cutDirection: 0 },
        ],
        _obstacles: [{ _time: 4, _lineIndex: 0, _type: 0, _duration: 2, _width: 1 }],
        _events: [{ _time: 1, _type: 0, _value: 5 }],
      },
      tempo,
    );

    expect(parsed.notes).toHaveLength(1);
    expect(parsed.bombs).toHaveLength(1);
    expect(parsed.obstacles).toHaveLength(1);
    expect(parsed.obstacles[0].h).toBe(5);
    expect(parsed.lightEvents).toHaveLength(1);
  });
});
