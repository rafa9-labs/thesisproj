import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { newsWsManager } from "@/api/newsWebSocket";

export function useNewsWebSocket() {
  const queryClient = useQueryClient();

  useEffect(() => {
    newsWsManager.connect();

    const unsub = newsWsManager.subscribe((event: unknown) => {
      const data = event as { event?: string };
      if (data.event === "news_sync") {
        queryClient.invalidateQueries({ queryKey: ["live-sentiment"] });
        queryClient.invalidateQueries({ queryKey: ["news-articles"] });
        queryClient.invalidateQueries({ queryKey: ["news-status"] });
      }
    });

    return () => {
      unsub();
    };
  }, [queryClient]);
}
