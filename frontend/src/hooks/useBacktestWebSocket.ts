import { useEffect, useRef, useCallback } from "react";
import { wsManager } from "@/api/websocket";
import { useJobStore } from "@/stores/useJobStore";
import type { WsEvent } from "@/api/schemas";

/**
 * Connects WebSocket to ALL running job IDs.
 * Uses a ref to diff old/new lists — only connects new IDs
 * and disconnects removed IDs. Avoids reconnect churn on re-renders.
 */
export function useBacktestWebSocket(jobIds: string[]) {
  const handleWsEvent = useJobStore((s) => s.handleWsEvent);
  const handlerRef = useRef(handleWsEvent);
  const connectedRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    handlerRef.current = handleWsEvent;
  }, [handleWsEvent]);

  useEffect(() => {
    if (jobIds.length === 0) {
      for (const id of connectedRef.current) {
        wsManager.disconnect(id);
      }
      connectedRef.current.clear();
      return;
    }

    const newSet = new Set(jobIds);
    const prevSet = connectedRef.current;

    // Disconnect stale IDs
    for (const id of prevSet) {
      if (!newSet.has(id)) {
        if (import.meta.env.DEV) console.log("[WS-HOOK] disconnecting stale:", id.slice(0, 8));
        wsManager.disconnect(id);
      }
    }

    // Connect new IDs
    const unsubs: (() => void)[] = [];
    for (const id of newSet) {
      if (!prevSet.has(id)) {
        if (import.meta.env.DEV) console.log("[WS-HOOK] connecting new:", id.slice(0, 8));
        wsManager.connect(id);
        const unsub = wsManager.subscribe(id, (event: unknown) => {
          handlerRef.current(event as WsEvent);
          const ev = event as { event?: string };
          if (ev.event === "job_failed" || ev.event === "job_complete") {
            wsManager.markTerminal(id);
          }
        });
        unsubs.push(unsub);
      }
    }

    connectedRef.current = newSet;

    return () => {
      for (const u of unsubs) u();
    };
  }, [jobIds]);

  const disconnect = useCallback(() => {
    for (const id of connectedRef.current) {
      wsManager.disconnect(id);
    }
    connectedRef.current.clear();
  }, []);

  return { disconnect };
}
