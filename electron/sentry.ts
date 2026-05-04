/**
 * Sentry crash reporting — opt-in, production builds only.
 *
 * Initialized with a DSN from the SENTRY_DSN environment variable.
 * If no DSN is set or running in dev mode, Sentry is not started.
 * No personal data (PII) is sent.
 */

import * as Sentry from "@sentry/electron/main";
import { app } from "electron";

let sentryInitialized = false;

export function initSentry(dsn: string | undefined): void {
  if (!dsn || !app.isPackaged) {
    console.log("[Sentry] Not initialized — no DSN or dev mode");
    return;
  }

  try {
    Sentry.init({
      dsn,
      environment: "production",
      release: app.getVersion(),
      sendDefaultPii: false,
      maxBreadcrumbs: 20,
      beforeSend(event) {
        if (event.request) {
          delete event.request.cookies;
          delete event.request.headers;
        }
        return event;
      },
    });
    sentryInitialized = true;
    console.log("[Sentry] Initialized successfully");
  } catch (err) {
    console.error("[Sentry] Failed to initialize:", err);
  }
}

export function isSentryInitialized(): boolean {
  return sentryInitialized;
}

export function captureException(error: Error): void {
  if (sentryInitialized) {
    Sentry.captureException(error);
  }
}

export function captureMessage(message: string, level: Sentry.SeverityLevel = "info"): void {
  if (sentryInitialized) {
    Sentry.captureMessage(message, level);
  }
}