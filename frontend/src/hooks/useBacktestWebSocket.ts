import { useEffect, useRef, useCallback } from "react";
import { wsManager } from "@/api/websocket";
import { useJobStore } from "@/stores/useJobStore";
import type { WsEvent } from "@/api/schemas";

export function useBacktestWebSocket(jobId: string | null) {
  const handleWsEvent = useJobStore((s) => s.handleWsEvent);
  const handlerRef = useRef(handleWsEvent);

  useEffect(() => {
    handlerRef.current = handleWsEvent;
  }, [handleWsEvent]);

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
      if (import.meta.env.DEV) console.log("[WS-HOOK] unsubscribing for job:", jobId.slice(0, 8));
      clearTimeout(timer);
      unsub();
      // Keep wsManager connected across tab switches so the backend polling loop
      // and progress updates survive component unmount/remount.
    };
  }, [jobId]);

  const disconnect = useCallback(() => {
    wsManager.disconnect();
  }, []);

  return { disconnect };
}
