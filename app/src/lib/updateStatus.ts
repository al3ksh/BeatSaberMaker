import type { UpdateStatus } from "../types";

/** Keep the last known version across events that omit it; pct only for downloading. */
export function mergeUpdateStatus(prev: UpdateStatus, next: UpdateStatus): UpdateStatus {
  const version = next.version ?? prev.version;
  const merged: UpdateStatus = { state: next.state };
  if (version !== undefined) merged.version = version;
  if (next.state === "downloading" && next.pct !== undefined) merged.pct = next.pct;
  return merged;
}
