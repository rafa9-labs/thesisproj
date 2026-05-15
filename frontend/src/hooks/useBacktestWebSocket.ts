import { useEffect, useRef, useCallback } from "react";
import { wsManager } from "@/api/websocket";
import { useJobStore } from "@/stores/useJobStore";
import type { WsEvent } from "@/api/schemas";

export function useBacktestWebSocket(jobId: string | null) {
  const handleWsEvent = useJobStore((s) => s.handleWsEvent);
  const handlerRef = useRef(handleWsEvent);
  handlerRef.current = handleWsEvent;

  useEffect(() => {
    if (!jobId) return;
    if (import.meta.env.DEV) console.log("[WS-HOOK] connecting for job:", jobId.slice(0, 8));

    const timer = setTimeout(() => {
      wsManager.connect(jobId);
    }, 0);

    const unsub = wsManager.subscribe((event: unknown) => {
      if (import.meta.env.DEV) {
        const e = event as { event?: string; job_id?: string };
        console.log("[WS] event:", e.event, "job:", e.job_id?.slice(0, 8));
      }
      handlerRef.current(event as WsEvent);
    });

    return () => {
      if (import.meta.env.DEV) console.log("[WS-HOOK] cleaning up for job:", jobId.slice(0, 8));
      clearTimeout(timer);
      unsub();
      wsManager.disconnect();
    };
  }, [jobId]);

  const disconnect = useCallback(() => {
    wsManager.disconnect();
  }, []);

  return { disconnect };
}
