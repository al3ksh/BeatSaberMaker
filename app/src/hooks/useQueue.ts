import { useEffect, useMemo, useRef, useState } from "react";
import type { QueueEnqueueRequest, QueueItem } from "../types";

export interface QueueActions {
  enqueue: (req: QueueEnqueueRequest) => Promise<QueueItem>;
  cancel: (id: string) => Promise<QueueItem[]>;
  retry: (id: string) => Promise<QueueItem[]>;
  move: (id: string, direction: "up" | "down") => Promise<QueueItem[]>;
  clearFinished: () => Promise<QueueItem[]>;
}

export interface UseQueueResult {
  items: QueueItem[];
  activeItem: QueueItem | null;
  waitingCount: number;
  failedCount: number;
  actions: QueueActions;
}

/**
 * Subscribe to the main-process generation queue and expose derived counts and actions.
 */
export function useQueue(): UseQueueResult {
  const [items, setItems] = useState<QueueItem[]>([]);
  const receivedSnapshotRef = useRef(false);

  useEffect(() => {
    let active = true;

    const unsubChanged = window.bsm?.onQueueChanged?.((nextItems) => {
      if (!active) return;
      receivedSnapshotRef.current = true;
      setItems(nextItems);
    });

    const unsubProgress = window.bsm?.onQueueProgress?.((ev) => {
      if (!active) return;
      setItems((prev) =>
        prev.map((item) =>
          item.id === ev.id
            ? { ...item, progress: { stage: ev.stage, pct: ev.pct } }
            : item,
        ),
      );
    });

    window.bsm
      ?.getQueue?.()
      .then((initial) => {
        if (active && !receivedSnapshotRef.current) {
          setItems(initial);
        }
      })
      .catch(() => {});

    return () => {
      active = false;
      unsubChanged?.();
      unsubProgress?.();
    };
  }, []);

  const activeItem = useMemo(
    () =>
      items.find(
        (item) => item.status === "downloading" || item.status === "running",
      ) ?? null,
    [items],
  );

  const waitingCount = useMemo(
    () => items.filter((item) => item.status === "waiting").length,
    [items],
  );

  const failedCount = useMemo(
    () => items.filter((item) => item.status === "failed").length,
    [items],
  );

  const actions = useMemo<QueueActions>(
    () => ({
      enqueue: (req) => window.bsm.enqueueGeneration(req),
      cancel: async (id) => {
        const next = await window.bsm.cancelQueueItem(id);
        if (Array.isArray(next)) setItems(next);
        return next;
      },
      retry: async (id) => {
        const next = await window.bsm.retryQueueItem(id);
        if (Array.isArray(next)) setItems(next);
        return next;
      },
      move: async (id, direction) => {
        const next = await window.bsm.moveQueueItem(id, direction);
        if (Array.isArray(next)) setItems(next);
        return next;
      },
      clearFinished: async () => {
        const next = await window.bsm.clearFinishedQueue();
        if (Array.isArray(next)) setItems(next);
        return next;
      },
    }),
    [],
  );

  return { items, activeItem, waitingCount, failedCount, actions };
}
