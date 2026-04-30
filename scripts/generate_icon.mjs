/**
 * Generate placeholder app icons for Electron Builder.
 *
 * Run: node scripts/generate_icon.mjs
 * Output: build/icon.ico (Windows), build/icon.png (Linux fallback)
 *
 * Creates a simple SVG-based icon, then converts to PNG and ICO.
 * Uses only built-in Node.js + sharp (install if needed).
 *
 * Replace with proper branding assets before commercial release.
 */

import { writeFileSync, mkdirSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BUILD_DIR = join(__dirname, "..", "build");

mkdirSync(BUILD_DIR, { recursive: true });

const SVG_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <rect width="512" height="512" rx="80" fill="#131722"/>
  <rect x="16" y="16" width="480" height="480" rx="64" fill="#1E222D" stroke="#2A2E39" stroke-width="2"/>
  <polyline points="76,340 128,310 179,290 230,250 282,270 333,200 385,180 436,112"
    fill="none" stroke="#089981" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="380" y="420" font-family="Consolas, monospace" font-size="80" font-weight="bold" fill="#E0E3EB">FX</text>
  <circle cx="333" cy="200" r="6" fill="#089981"/>
  <circle cx="436" cy="112" r="6" fill="#2962FF"/>
  <line x1="76" y1="380" x2="436" y2="380" stroke="#363A45" stroke-width="1"/>
</svg>`;

const ICON_PNG_PATH = join(BUILD_DIR, "icon.png");
const ICON_ICO_PATH = join(BUILD_DIR, "icon.ico");
const ICON_SVG_PATH = join(BUILD_DIR, "icon.svg");

writeFileSync(ICON_SVG_PATH, SVG_ICON);
console.log("Created build/icon.svg");

async function convertToPng() {
  let sharp;
  try {
    sharp = (await import("../frontend/node_modules/sharp/src/index.js")).default;
  } catch {
    try {
      sharp = (await import("sharp")).default;
    } catch {
      console.warn("sharp not available. Install it: cd frontend && npm install --save-dev sharp");
      console.warn("Creating placeholder icons...");

      const minimalPng = Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "base64"
      );
      writeFileSync(ICON_PNG_PATH, minimalPng);
      console.log("Created build/icon.png (1x1 placeholder)");

      const minimalIco = Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "base64"
      );
      writeFileSync(ICON_ICO_PATH, minimalIco);
      console.log("Created build/icon.ico (placeholder - replace before release!)");
      return;
    }
  }
    await sharp(Buffer.from(SVG_ICON))
      .resize(512, 512)
      .png()
      .toFile(ICON_PNG_PATH);
    console.log("Created build/icon.png (512x512)");

    // Generate ICO (multi-size PNGs packed in ICO format)
    const sizes = [16, 32, 48, 64, 128, 256];
    const pngBuffers = [];
    for (const size of sizes) {
      const buf = await sharp(Buffer.from(SVG_ICON))
        .resize(size, size)
        .png()
        .toBuffer();
      pngBuffers.push({ size, buf });
    }

    // Build ICO binary
    const headerSize = 6;
    const dirEntrySize = 16;
    const dirSize = dirEntrySize * pngBuffers.length;
    let dataOffset = headerSize + dirSize;

    const parts = [];
    // ICO header
    const header = Buffer.alloc(headerSize);
    header.writeUInt16LE(0, 0); // Reserved
    header.writeUInt16LE(1, 2); // Type: ICO
    header.writeUInt16LE(pngBuffers.length, 4); // Count
    parts.push(header);

    // Directory entries
    let currentOffset = dataOffset;
    for (let i = 0; i < pngBuffers.length; i++) {
      const entry = Buffer.alloc(dirEntrySize);
      const { size, buf } = pngBuffers[i];
      entry.writeUInt8(size === 256 ? 0 : size, 0);  // Width (0 = 256)
      entry.writeUInt8(size === 256 ? 0 : size, 1);  // Height
      entry.writeUInt8(0, 2);   // Color palette
      entry.writeUInt8(0, 3);   // Reserved
      entry.writeUInt16LE(1, 4); // Color planes
      entry.writeUInt16LE(32, 6); // Bits per pixel
      entry.writeUInt32LE(buf.length, 8); // Image size
      entry.writeUInt32LE(currentOffset, 12); // Image offset
      parts.push(entry);
      currentOffset += buf.length;
    }

    // Image data
    for (const { buf } of pngBuffers) {
      parts.push(buf);
    }

    const ico = Buffer.concat(parts);
    writeFileSync(ICON_ICO_PATH, ico);
    console.log("Created build/icon.ico (16,32,48,64,128,256)");

  } catch (err) {
    console.warn("sharp not available, creating PNG fallback from SVG...");
    console.warn("Install sharp for proper icon generation: npm install --save-dev sharp");

    // Fallback: just copy the SVG as the icon (electron-builder can use SVG on some platforms)
    console.log("SVG icon saved at build/icon.svg");
    console.log("For Windows .ico, install sharp: cd frontend && npm install --save-dev sharp && cd .. && node scripts/generate_icon.mjs");

    // Create a minimal 1x1 PNG as placeholder so build doesn't fail
    const minimalPng = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      "base64"
    );
    writeFileSync(ICON_PNG_PATH, minimalPng);
    console.log("Created build/icon.png (placeholder 1x1)");

    // Create minimal ICO placeholder
    const minimalIco = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      "base64"
    );
    writeFileSync(ICON_ICO_PATH, minimalIco);
    console.log("Created build/icon.ico (placeholder - replace before release!)");
  }
}

convertToPng();