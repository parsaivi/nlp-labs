// Writes an offline copy of a built page and prints its path.
//   node tools/offline.js <lab-id>     -> build/offline-<lab-id>.html
//   node tools/offline.js ""           -> build/offline-gallery.html
// The copy loads React from tools/vendor/ (downloaded once, gitignored) and drops the web fonts, so headless
// Chrome never touches the network. On some machines headless Chrome can't reach the CDN and just hangs.
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const VENDOR = path.join(__dirname, "vendor");
const LIBS = {
  "react.js": "https://cdnjs.cloudflare.com/ajax/libs/react/18.3.1/umd/react.production.min.js",
  "react-dom.js": "https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.3.1/umd/react-dom.production.min.js",
};

async function main() {
  const lab = process.argv[2] || "";
  fs.mkdirSync(VENDOR, { recursive: true });
  for (const [file, url] of Object.entries(LIBS)) {
    const p = path.join(VENDOR, file);
    if (!fs.existsSync(p)) {
      const r = await fetch(url);
      if (!r.ok) throw new Error("download failed: " + url);
      fs.writeFileSync(p, Buffer.from(await r.arrayBuffer()));
    }
  }
  const src = path.join(ROOT, "docs", lab, "index.html");
  let html = fs.readFileSync(src, "utf8");
  for (const [file, url] of Object.entries(LIBS)) html = html.split(url).join("file://" + path.join(VENDOR, file));
  html = html.replace(/^.*fonts\.(googleapis|gstatic)\.com.*$/gm, "");
  if (/https:\/\/(unpkg|cdn)/.test(html)) console.error("note: " + (lab || "gallery") + " also loads other CDN scripts; those still need the network");
  fs.mkdirSync(path.join(ROOT, "build"), { recursive: true });
  const out = path.join(ROOT, "build", "offline-" + (lab || "gallery") + ".html");
  fs.writeFileSync(out, html);
  process.stdout.write(out);
}
main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
