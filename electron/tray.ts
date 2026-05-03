/**
 * System tray icon with status indicators.
 */

import { Tray, Menu, BrowserWindow, nativeImage, app } from "electron";
import path from "path";

const TRAY_ICON_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAAAARnQU1BACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAGISURBVFhH7ZY9TsNAEIVnpa9yE6QoG0FRlrCAosQUQbIIrhSxAfIH0EfkDRSO4gZ5A22K3E1Od7p7p+x2p9HO9p1z5ps+DKOk0XS+TxJ4zyXpHknSOv0XSc9fArBf0nN3y+EYq0l6fw7wlKT3Z4G1cB/YAa5h9s4A27kKzIN1qPUIkNYhE1gB1qMcwk1gM8px3AXmA9+AdfAYuAo8BObDbXAHWICskPU+q0i7gX3ICHAGnAVWga3IPnDPXn8H5kFrYR9YB7bBbbASZJ5x4BiYD3fgbpBzLAW24zy4A1yH2+Z1fAEex/l4Ai6BP+ISuArOgDvIN9xG1iINXIYqcBU4B7fCfXD3Tbj7M2gCHoA0sR+sgQ2wAE6BmfAdrIMb4C3YARfABrATHoAN8B6kz+8B8mEd2Aa2wV6Q5fk2MA++xGnZ/x/4C1SZ/ceAR1UK3AAAAABJRU5ErkJggg==",
  "base64",
);

export function createTray(mainWindow: BrowserWindow): Tray {
  let icon: Electron.NativeImage;
  if (app.isPackaged) {
    icon = nativeImage.createFromBuffer(TRAY_ICON_PNG);
  } else {
    const iconPath = path.resolve(__dirname, "..", "frontend", "public", "favicon.svg");
    const fromFile = nativeImage.createFromPath(iconPath);
    icon = fromFile.isEmpty() ? nativeImage.createFromBuffer(TRAY_ICON_PNG) : fromFile;
  }
  const tray = new Tray(icon);

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
