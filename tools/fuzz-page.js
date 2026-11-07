// Runs inside the lab page; loaded by tools/fuzz.js.
async function fuzz() {
  const errs = [];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
  const navN = document.querySelectorAll("nav button").length;
  const stats = [];
  for (let li = 0; li < navN; li++) {
    document.querySelectorAll("nav button")[li].click();
    await sleep(200);
    for (let sc = 0; sc < 4; sc++) {
      const box = [...document.querySelectorAll("#main > div")].find((d) => !d.hidden);
      const tabs = [...box.querySelectorAll("button")].filter((b) => /^\d (Puzzle|Explore|Name|Challenge)$/.test(b.textContent.trim()));
      if (tabs.length < 4) { errs.push("lesson " + li + ": no screen tabs"); break; }
      tabs[sc].click();
      await sleep(120);
      let n = { btn: 0, range: 0, text: 0, cell: 0, drag: 0 };
      const skip = /^(\d (Puzzle|Explore|Name|Challenge)|← Back|Next: .*|Last screen)$/;
      for (let pass = 0; pass < 2; pass++) {
        const btns = [...box.querySelectorAll("button")].filter((b) => !skip.test(b.textContent.trim()));
        for (const b of btns) if (b.isConnected && !b.disabled) { b.click(); n.btn++; await sleep(15); }
      }
      for (const r of [...box.querySelectorAll("input[type=range]")]) for (const v of [r.min, r.max, (Number(r.min) + Number(r.max)) / 2]) { if (!r.isConnected) continue; setter.call(r, v); r.dispatchEvent(new Event("input", { bubbles: true })); n.range++; await sleep(15); }
      for (const t of [...box.querySelectorAll("input:not([type=range])")]) for (const v of ["", "xyzzy", "کتابخانه", "a", "playing"]) { setter.call(t, v); t.dispatchEvent(new Event("input", { bubbles: true })); n.text++; await sleep(15); }
      const cells = [...box.querySelectorAll("svg g")].filter((g) => g.style.cursor === "pointer");
      for (const g of cells.slice(0, 40)) { if (!g.isConnected) continue; g.dispatchEvent(new MouseEvent("click", { bubbles: true })); n.cell++; await sleep(8); }
      for (const c of [...box.querySelectorAll("circle[data-id]")]) {
        if (!c.isConnected) continue;
        const svg = c.ownerSVGElement; const rb = svg.getBoundingClientRect(); const cb = c.getBoundingClientRect();
        const o = (x, y) => ({ bubbles: true, clientX: x, clientY: y, pointerId: 1, isPrimary: true, pointerType: "mouse" });
        c.dispatchEvent(new PointerEvent("pointerdown", o(cb.x + cb.width / 2, cb.y + cb.height / 2)));
        svg.dispatchEvent(new PointerEvent("pointermove", o(rb.x + rb.width * 0.2, rb.y + rb.height * 0.8)));
        svg.dispatchEvent(new PointerEvent("pointermove", o(rb.x + rb.width * 0.95, rb.y + rb.height * 0.05)));
        svg.dispatchEvent(new PointerEvent("pointerup", o(rb.x + rb.width * 0.95, rb.y + rb.height * 0.05)));
        n.drag++; await sleep(20);
      }
      await sleep(400);
      const crash = document.querySelector(".err");
      stats.push(li + ":" + sc + " " + JSON.stringify(n));
      if (crash) { errs.push("CRASH lesson " + li + " screen " + sc + ": " + crash.textContent.slice(0, 400)); crash.remove(); }
    }
  }
  return { errs, stats };
}
