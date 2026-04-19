/**
 * System tray icon with status indicators.
 */

import { Tray, Menu, BrowserWindow, nativeImage, app } from "electron";
import path from "path";

export function createTray(mainWindow: BrowserWindow): Tray {
  const iconPath = path.resolve(__dirname, "..", "frontend", "public", "favicon.svg");
  const icon = nativeImage.createFromPath(iconPath);
  const tray = new Tray(icon.isEmpty() ? nativeImage.createEmpty() : icon);

  const contextMenu = Menu.buildFromTemplate([
    { label: "FX ML Backtester", enabled: false },
    { type: "separator" },
    {
      label: "Show Window",
      click: () => {
        mainWindow.show();
        mainWindow.focus();
      },
    },
    {
      label: "New Backtest",
      click: () => {
        mainWindow.show();
        mainWindow.webContents.send("navigate", "/backtest");
      },
    },
    { type: "separator" },
    { label: "Quit", click: () => app.quit() },
  ]);

  tray.setToolTip("FX ML Backtester");
  tray.setContextMenu(contextMenu);

  tray.on("double-click", () => {
    mainWindow.show();
    mainWindow.focus();
  });

  return tray;
}
