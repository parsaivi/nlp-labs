// Interaction smoke test: opens a built lab in headless Chrome (DevTools protocol), then on every screen of
// every lesson clicks each button and clickable cell, sweeps each slider to min / max / middle, types into
// text inputs and drags each draggable point. Reports any lesson crash or console error.
//   node tools/fuzz.js <lab-id>
// Exit code 0 = clean, 2 = something broke. Uses the offline copy from tools/offline.js.
const { spawn, execFileSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const lab = process.argv[2];
if (!lab) {
  console.error("usage: node tools/fuzz.js <lab-id>");
  process.exit(1);
}
const file = execFileSync("node", [path.join(__dirname, "offline.js"), lab]).toString();
const PORT = 9333 + Math.floor(Math.random() * 500);
const prof = fs.mkdtempSync(path.join(os.tmpdir(), "fuzz-"));
const chrome = spawn("google-chrome", ["--headless=new", "--disable-gpu", "--no-first-run", "--user-data-dir=" + prof, "--remote-debugging-port=" + PORT, "--window-size=1400,1600", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PAGE_SCRIPT = fs.readFileSync(path.join(__dirname, "fuzz-page.js"), "utf8");

(async () => {
  let targets = [];
  for (let i = 0; i < 40 && !targets.length; i++) {
    try {
      targets = await (await fetch("http://127.0.0.1:" + PORT + "/json")).json();
    } catch (e) {
      await sleep(250);
    }
  }
  const page = targets.find((t) => t.type === "page");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0;
  const pending = {};
  const logs = [];
  const send = (method, params) =>
    new Promise((res) => {
      const i = ++id;
      pending[i] = res;
      ws.send(JSON.stringify({ id: i, method, params }));
    });
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending[d.id]) {
      pending[d.id](d);
      delete pending[d.id];
    }
    if (d.method === "Runtime.exceptionThrown") {
      const x = d.params.exceptionDetails;
      logs.push("exception: " + String((x.exception && x.exception.description) || x.text).slice(0, 400));
    }
    if (d.method === "Runtime.consoleAPICalled" && d.params.type === "error") logs.push("console.error: " + d.params.args.map((a) => a.value || a.description || "").join(" ").slice(0, 400));
  };
  await new Promise((r) => (ws.onopen = r));
  await send("Runtime.enable", {});
  await send("Page.enable", {});
  await send("Page.navigate", { url: "file://" + file });
  await sleep(2500);
  const r = await send("Runtime.evaluate", { expression: PAGE_SCRIPT + "\nfuzz()", awaitPromise: true, returnByValue: true, timeout: 280000 });
  const out = r.result && r.result.result ? r.result.result.value : null;
  if (!out) console.log("no result: " + JSON.stringify(r).slice(0, 500));
  else {
    console.log(lab + ": " + out.stats.length + " screens exercised");
    console.log(out.errs.length ? "CRASHES:\n" + out.errs.join("\n") : "no crashes");
  }
  console.log(logs.length ? "console / exceptions:\n" + logs.slice(0, 15).join("\n") : "no console errors");
  ws.close();
  chrome.kill();
  // Chrome may still be writing its profile right after kill(); retry, and never let cleanup fail the run
  try { fs.rmSync(prof, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }); } catch (e) {}
  process.exit(out && !out.errs.length && !logs.length ? 0 : 2);
})().catch((e) => {
  console.error("harness error", e);
  chrome.kill();
  process.exit(1);
});
