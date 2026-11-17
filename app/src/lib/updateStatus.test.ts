import { describe, expect, it } from "vitest";
import { mergeUpdateStatus } from "./updateStatus";

describe("mergeUpdateStatus", () => {
  it("keeps version from available into downloading, with pct", () => {
    const prev = mergeUpdateStatus({ state: "idle" }, { state: "available", version: "0.1.1" });
    expect(mergeUpdateStatus(prev, { state: "downloading", pct: 10 })).toEqual({
      state: "downloading",
      version: "0.1.1",
      pct: 10,
    });
  });
  it("drops pct outside downloading and prefers the new version", () => {
    const prev = { state: "downloading" as const, version: "0.1.1", pct: 99 };
    expect(mergeUpdateStatus(prev, { state: "downloaded", version: "0.1.2" })).toEqual({
      state: "downloaded",
      version: "0.1.2",
    });
  });
});
