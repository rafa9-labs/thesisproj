import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useLiveCandles } from "@/api/queries";
import apiClient from "@/api/client";
import type { CandleBar } from "@/api/schemas";

export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

const _TF_SECONDS: Record<string, number> = {
  M15: 900,
  M30: 1800,
  H1: 3600,
  H2: 7200,
  H4: 14400,
};

interface PriceTickEvent {
  event: "price_tick";
  pair: string;
  timeframe: string;
  live_price: number;
  forming_candle: CandleBar;
  bid: number;
  ask: number;
}

interface NewBarSavedEvent {
  event: "new_bar_saved";
  pair: string;
  timeframe: string;
  candle: CandleBar;
}

type ChartWSEvent =
  | PriceTickEvent
  | NewBarSavedEvent
  | { event: "heartbeat"; time: number }
  | { event: "error"; message: string };

interface CandlesResponse {
  pair: string;
  timeframe: string;
  candles: Array<{ t: number; o: number; h: number; l: number; c: number; volume: number }>;
}

interface UseChartStreamResult {
  historical: CandleData[];
  liveBar: CandleData | null;
  isLoading: boolean;
  isStreaming: boolean;
  loadOlder: () => Promise<boolean>;
  hasOlder: boolean;
  isLoadingOlder: boolean;
  setChartReady: (ready: boolean) => void;
}

export function useChartStream(
  pair: string,
  timeframe: string,
  limit = 500,
): UseChartStreamResult {
  const queryClient = useQueryClient();
  const { data, isLoading } = useLiveCandles(pair, timeframe, limit);
  const [isStreaming, setIsStreaming] = useState(false);
  const [liveBar, setLiveBar] = useState<CandleData | null>(null);
  const [olderBars, setOlderBars] = useState<CandleData[]>([]);
  const [hasOlder, setHasOlder] = useState(true);
  const [isLoadingOlder, setIsLoadingOlder] = useState(false);
  const [chartReady, setChartReady] = useState(false);
  const chartReadyRef = useRef(false);
  const lastHistTimeRef = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);
  const prevPairTfRef = useRef(`${pair}:${timeframe}`);

  useEffect(() => {
    const key = `${pair}:${timeframe}`;
    if (key === prevPairTfRef.current) return;
    prevPairTfRef.current = key;
    setOlderBars([]);
    setHasOlder(true);
    setChartReady(false);
    chartReadyRef.current = false;
    setLiveBar(null);
    lastHistTimeRef.current = 0;
  }, [pair, timeframe]);

  const setReady = useCallback((ready: boolean) => {
    chartReadyRef.current = ready;
    setChartReady(ready);
  }, []);

  const historical: CandleData[] = useMemo(() => {
    const candles = data?.candles;
    if (!candles || !Array.isArray(candles)) return [];
    return candles.map((c) => ({
      time: c.t,
      open: c.o,
      high: c.h,
      low: c.l,
      close: c.c,
      volume: c.volume,
    }));
  }, [data]);

  useEffect(() => {
    if (historical.length > 0) {
      lastHistTimeRef.current = historical[historical.length - 1].time;
    }
  }, [historical]);

  const allHistorical = useMemo(() => {
    const combined = [...olderBars, ...historical];
    const seen = new Set<number>();
    return combined
      .filter((c) => {
        if (seen.has(c.time)) return false;
        seen.add(c.time);
        return true;
      })
      .sort((a, b) => a.time - b.time);
  }, [olderBars, historical]);

  const loadOlder = useCallback(async (): Promise<boolean> => {
    if (isLoadingOlder || !hasOlder) return false;

    const allBars = [...olderBars, ...historical];
    if (allBars.length === 0) return false;

    const oldestTime = allBars[0].time;
    setIsLoadingOlder(true);

    try {
      const { data: resp } = await apiClient.get<CandlesResponse>(
        `/candles/${pair}/${timeframe}`,
        { params: { limit: 200, end: oldestTime - 1 } },
      );
      const fetched: CandleData[] = (resp.candles || []).map((c) => ({
        time: c.t,
        open: c.o,
        high: c.h,
        low: c.l,
        close: c.c,
        volume: c.volume,
      }));

      if (fetched.length === 0) {
        setHasOlder(false);
        return false;
      }

      setOlderBars((prev) => {
        const existingTimes = new Set([...prev, ...historical].map((b) => b.time));
        const unique = fetched.filter((b) => !existingTimes.has(b.time));
        return [...unique, ...prev].sort((a, b) => a.time - b.time);
      });

      return true;
    } catch {
      return false;
    } finally {
      setIsLoadingOlder(false);
    }
  }, [olderBars, historical, pair, timeframe, isLoadingOlder, hasOlder]);

  useEffect(() => {
    if (isLoading || !pair || !timeframe) return;

    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsHost = window.location.host || "localhost:8001";
    const wsUrl = `${wsProtocol}//${wsHost}/api/v1/chart/${pair}/${timeframe}/ws`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => setIsStreaming(true);

    ws.onmessage = (evt) => {
      try {
        const msg: ChartWSEvent = JSON.parse(evt.data);

        if (msg.event === "price_tick" && msg.forming_candle) {
          if (!chartReadyRef.current) return;
          setLiveBar((prev) => {
            const fc = msg.forming_candle as CandleBar;
            if (!prev || prev.time !== fc.time) {
              return { ...fc, volume: 0 };
            }
            return {
              ...fc,
              open: prev.open,
              volume: prev.volume ?? 0,
            };
          });
        } else if (msg.event === "new_bar_saved" && msg.candle) {
          if (!chartReadyRef.current) return;
          const nc = msg.candle as CandleBar;

          const period = _TF_SECONDS[timeframe] ?? 1800;
          if (
            lastHistTimeRef.current > 0 &&
            nc.time - lastHistTimeRef.current > period * 1.5
          ) {
            queryClient.invalidateQueries({
              queryKey: ["live-candles", pair, timeframe],
            });
          }

          setLiveBar((prev) => {
            return { ...nc, volume: prev?.volume ?? 0 };
          });
        }
      } catch {
        /* ignore parse errors */
      }
    };

    ws.onerror = () => setIsStreaming(false);
    ws.onclose = () => {
      setIsStreaming(false);
      wsRef.current = null;
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [pair, timeframe, isLoading]);

  return {
    historical: allHistorical,
    liveBar: chartReady ? liveBar : null,
    isLoading,
    isStreaming: chartReady && isStreaming,
    loadOlder,
    hasOlder,
    isLoadingOlder,
    setChartReady: setReady,
  };
}
