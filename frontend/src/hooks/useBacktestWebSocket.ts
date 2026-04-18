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

    wsManager.connect(jobId);

    const unsub = wsManager.subscribe((event: unknown) => {
      handlerRef.current(event as WsEvent);
    });

    return () => {
      unsub();
    };
  }, [jobId]);

  const disconnect = useCallback(() => {
    wsManager.disconnect();
  }, []);

  return { disconnect };
}
