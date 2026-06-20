import { useEffect, useRef } from "react";
import { useJobStore } from "@/stores/useJobStore";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { playChime } from "./useNotificationSound";

export function useJobNotifications() {
  const settings = useSettingsStore();
  const lastCountRef = useRef(useJobStore.getState().unreadCompletedCount);

  useEffect(() => {
    const unsub = useJobStore.subscribe((state) => {
      const newCount = state.unreadCompletedCount;
      if (newCount <= lastCountRef.current) {
        lastCountRef.current = newCount;
        return;
      }
      lastCountRef.current = newCount;

      if (!settings.notificationsEnabled) return;
      if (settings.notificationSound) {
        playChime();
      }

      // Show desktop notification if tab is backgrounded
      if (document.visibilityState !== "visible" && "Notification" in window) {
        try {
          if (Notification.permission === "granted") {
            const latest = state.completedJobs[state.completedJobs.length - 1];
            if (latest) {
              const modelStr = latest.models.length > 2
                ? `${latest.models.length} models`
                : latest.models.join(", ");
              new Notification(
                latest.status === "completed" ? "Backtest Complete" : "Backtest Failed",
                {
                  body: `${modelStr} on ${latest.pair} · ${latest.jobId.slice(0, 8)}`,
                  icon: "/favicon.ico",
                },
              );
            }
          }
        } catch {
          // Notifications not supported
        }
      }

      // Update document title
      if (document.visibilityState !== "visible") {
        document.title = `KodaQuant (${newCount})`;
      }
    });

    return () => unsub();
  }, [settings.notificationsEnabled, settings.notificationSound]);

  // Clear badge on tab focus
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === "visible") {
        useJobStore.getState().clearUnreadCount();
        document.title = "KodaQuant";
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, []);
}
