/**
 * Native application menu.
 */

import { Menu, BrowserWindow, app, shell } from "electron";

export function buildMenu(isDev: boolean): Menu {
  const template: Electron.MenuItemConstructorOptions[] = [
    {
      label: "File",
      submenu: [
        { label: "New Backtest", accelerator: "CmdOrCtrl+N", click: () => sendNav("backtest") },
        { type: "separator" },
        { label: "Settings", accelerator: "CmdOrCtrl+,", click: () => sendNav("settings") },
        { type: "separator" },
        { role: "quit" },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { role: "toggleDevTools", visible: isDev },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    {
      label: "Window",
      submenu: [{ role: "minimize" }, { role: "zoom" }, { role: "close" }],
    },
    {
      label: "Help",
      submenu: [
        { label: "Check for Updates...", click: () => triggerUpdateCheck() },
        { type: "separator" },
        {
          label: "Documentation",
          click: () => shell.openExternal("https://github.com/rafa9-labs/thesisproj"),
        },
        {
          label: "Report Issue",
          click: () => shell.openExternal("https://github.com/rafa9-labs/thesisproj/issues"),
        },
        { type: "separator" },
        { label: `Version ${app.getVersion()}`, enabled: false },
      ],
    },
  ];

  return Menu.buildFromTemplate(template);
}

function sendNav(path: string) {
  const win = BrowserWindow.getAllWindows()[0];
  if (win) {
    win.webContents.send("navigate", path);
  }
}

function triggerUpdateCheck() {
  const win = BrowserWindow.getAllWindows()[0];
  if (win) {
    win.webContents.send("trigger-update-check");
  }
}
