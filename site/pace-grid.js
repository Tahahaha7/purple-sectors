/* pace-grid.js — where would each driver qualify in each car?
 *
 * Drop-in: <div data-pace-grid data-src="data/pace.json"></div> <script src="pace-grid.js" defer></script>
 * Reads site/data/pace.json written by `python -m purple_sectors.pace.export`.
 * Inherits the page's font and text colour; override the tokens below from your own CSS if you like:
 *   --pg-p1 (the purple for P1), --pg-line (hairlines), --pg-muted (secondary text).
 * Add data-no-heading when the page already has a heading for the section.
 * No dependencies, no build step. pace.json carries seconds on a 90-second lap.
 */
(function () {
  "use strict";

  const CSS = `
  .pg { max-width: 72rem; line-height: 1.45; }
  .pg h2 { font-size: 1.35rem; font-weight: 600; margin: 0 0 .25rem; }
  .pg p { margin: 0 0 1rem; color: var(--pg-muted, color-mix(in srgb, currentColor 62%, transparent)); max-width: 60ch; }
  .pg-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .pg table { border-collapse: separate; border-spacing: 3px; font-variant-numeric: tabular-nums; }
  .pg th, .pg td { padding: 0; text-align: center; font-weight: 400; }
  .pg thead th { font-size: .78rem; padding-bottom: .35rem;
                 color: var(--pg-muted, color-mix(in srgb, currentColor 62%, transparent)); }
  .pg thead th.pg-fast { color: inherit; }
  .pg tbody th { text-align: left; padding-right: .6rem; position: sticky; left: 0; background: inherit; }
  .pg .pg-name { all: unset; cursor: pointer; display: inline-flex; gap: .5rem; align-items: baseline;
                 padding: .15rem .35rem; border-radius: 3px; white-space: nowrap; }
  .pg .pg-name:hover, .pg .pg-name:focus-visible { outline: 2px solid var(--pg-p1, #7c3aed); outline-offset: 1px; }
  .pg .pg-name[aria-pressed="true"] { background: var(--pg-p1, #7c3aed); color: #fff; }
  .pg .pg-name small { font-size: .74rem; opacity: .7; }
  .pg .pg-cell { all: unset; cursor: pointer; display: grid; place-items: center; width: 2.6rem; height: 2.1rem;
                 border-radius: 4px; font-size: .9rem; box-sizing: border-box;
                 border: 1px solid var(--pg-line, color-mix(in srgb, currentColor 14%, transparent)); }
  .pg .pg-cell:focus-visible { outline: 2px solid currentColor; outline-offset: 2px; }
  .pg .pg-cell.pg-seat { border: 2px solid currentColor; }
  .pg .pg-cell.pg-guess { border-style: dashed; }
  .pg .pg-cell[aria-pressed="true"] { outline: 3px solid var(--pg-p1, #7c3aed); outline-offset: 1px; }
  .pg-key { display: flex; flex-wrap: wrap; gap: 1.1rem; margin: .8rem 0 0;
            font-size: .82rem; color: var(--pg-muted, color-mix(in srgb, currentColor 62%, transparent)); }
  .pg-key span { display: inline-flex; align-items: center; gap: .4rem; }
  .pg-key i { display: inline-block; width: 1rem; height: .9rem; border-radius: 3px; box-sizing: border-box;
              border: 1px solid var(--pg-line, color-mix(in srgb, currentColor 14%, transparent)); }
  .pg-detail { min-height: 3.2rem; margin: 1rem 0 0; padding: .75rem 0 0;
               border-top: 1px solid var(--pg-line, color-mix(in srgb, currentColor 14%, transparent)); }
  .pg-detail b { font-weight: 600; }
  @media (max-width: 640px) { .pg .pg-cell { width: 2.2rem; height: 1.9rem; font-size: .8rem; } }
  `;

  const fmtS = (sec) => sec.toFixed(2);
  const pos = (p) => `P${p}`;

  // Fill: one purple, strongest at the front of the grid. Text flips to white on strong fills.
  function fill(p, n, p1) {
    const share = Math.max(0.07, 1 - (p - 1) / Math.max(1, n - 1));
    return { background: `color-mix(in srgb, ${p1} ${Math.round(share * 100)}%, transparent)`,
             color: share > 0.55 ? "#fff" : "inherit" };
  }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "text") node.textContent = v;
      else if (k === "style") Object.assign(node.style, v);
      else node.setAttribute(k, v);
    }
    for (const c of children || []) node.append(c);
    return node;
  }

  function sentenceFor(data, i, j) {
    const d = data.drivers[i], team = data.grid.teams[j], c = data.grid.cells[i][j];
    const who = d.name || d.code;
    const best = Math.min(...data.grid.cells.map((row) => row.find((x) => x.seat).pace[1]));
    const behind = c.pace[1] - best;
    const where = c.seat ? `in the ${team}, their current seat` : `in the ${team}`;
    const guess = c.seen ? `` : ` ${who} has never driven this car, so the fit between them is a guess from the prior.`;
    return `${who} ${where} would qualify around ${pos(c.pos[1])} (${pos(c.pos[0])} to ${pos(c.pos[2])}), `
         + `${Math.abs(behind).toFixed(2)} s ${behind >= 0 ? "behind" : "ahead of"} the quickest seat on the grid.${guess}`;
  }

  function swapSentence(data, a, b) {
    const A = data.drivers[a], B = data.drivers[b];
    const ja = data.grid.cars.indexOf(A.short), jb = data.grid.cars.indexOf(B.short);
    if (ja === jb) return `${A.name} and ${B.name} are teammates already. Pick a driver from another team to swap seats.`;
    const aNow = data.grid.cells[a][ja], aThere = data.grid.cells[a][jb];
    const bNow = data.grid.cells[b][jb], bThere = data.grid.cells[b][ja];
    return `Swap seats: ${A.name} goes from ${pos(aNow.pos[1])} in the ${A.team} to about ${pos(aThere.pos[1])} `
         + `in the ${B.team} (${pos(aThere.pos[0])} to ${pos(aThere.pos[2])}); ${B.name} goes from ${pos(bNow.pos[1])} `
         + `to about ${pos(bThere.pos[1])} in the ${A.team} (${pos(bThere.pos[0])} to ${pos(bThere.pos[2])}).`;
  }

  function render(root, data) {
    const p1 = getComputedStyle(root).getPropertyValue("--pg-p1").trim() || "#7c3aed";
    const n = data.drivers.length;
    root.classList.add("pg");
    root.replaceChildren();

    if (!root.hasAttribute("data-no-heading")) {
      root.append(el("h2", { text: "Where would each driver qualify in each car?" }));
      root.append(el("p", { text: `Drivers are ranked with the car taken out, cars with the driver taken out, `
        + `and every square puts one driver into one car against the rest of today's grid. `
        + `Click a square for the detail, or two names to swap their seats. `
        + `After ${data.event || "round " + data.round} ${data.season}.` }));
    }

    const detail = el("div", { class: "pg-detail", "aria-live": "polite",
      text: "Pick a square. Purple means the front of the grid; a bold border marks where the driver actually sits." });
    const picked = [];
    let pressedCell = null;

    const head = el("tr", {}, [el("th", { text: "" })]);
    data.grid.cars.forEach((code, j) => head.append(el("th", {
      text: code, title: data.grid.teams[j], class: j === 0 ? "pg-fast" : "" })));

    const body = el("tbody");
    data.drivers.forEach((d, i) => {
      const name = el("button", { class: "pg-name", type: "button", "aria-pressed": "false",
        title: `${d.name}: ${fmtS(d.skill[1])} s against the average driver, `
             + `${d.moved ? "has driven more than one team" : "one team only, so this rests partly on the prior"}` },
        [document.createTextNode(d.code), el("small", { text: d.short })]);
      name.addEventListener("click", () => {
        const k = picked.indexOf(i);
        if (k >= 0) picked.splice(k, 1); else { picked.push(i); if (picked.length > 2) picked.shift(); }
        body.querySelectorAll(".pg-name").forEach((b, idx) => b.setAttribute("aria-pressed", String(picked.includes(idx))));
        detail.textContent = picked.length === 2 ? swapSentence(data, picked[0], picked[1])
          : picked.length === 1 ? `${data.drivers[picked[0]].name} selected. Pick a second driver to swap seats.`
          : "Pick a square, or two names to swap seats.";
      });
      const tr = el("tr", {}, [el("th", { scope: "row" }, [name])]);
      data.grid.cells[i].forEach((c, j) => {
        const btn = el("button", { class: "pg-cell" + (c.seat ? " pg-seat" : "") + (c.seen ? "" : " pg-guess"),
          type: "button", "aria-pressed": "false", style: fill(c.pos[1], n, p1), text: String(c.pos[1]),
          "aria-label": `${d.name} in the ${data.grid.teams[j]}: ${pos(c.pos[1])}, ${pos(c.pos[0])} to ${pos(c.pos[2])}` });
        btn.addEventListener("click", () => {
          if (pressedCell) pressedCell.setAttribute("aria-pressed", "false");
          pressedCell = btn; btn.setAttribute("aria-pressed", "true");
          detail.textContent = sentenceFor(data, i, j);
        });
        tr.append(el("td", {}, [btn]));
      });
      body.append(tr);
    });

    const table = el("table", { "aria-label": "Predicted qualifying position of every driver in every car" },
      [el("thead", {}, [head]), body]);
    root.append(el("div", { class: "pg-scroll" }, [table]));
    root.append(el("div", { class: "pg-key" }, [
      el("span", {}, [el("i", { style: fill(1, n, p1) }), document.createTextNode("front of the grid")]),
      el("span", {}, [el("i", { style: fill(n, n, p1) }), document.createTextNode("back of the grid")]),
      el("span", {}, [el("i", { style: { border: "2px solid currentColor" } }), document.createTextNode("actual seat")]),
      el("span", {}, [el("i", { style: { borderStyle: "dashed" } }), document.createTextNode("never driven that car")]),
      el("span", {}, [document.createTextNode(`fastest car on the left, fastest driver at the top, `
        + `${data.units.interval} intervals in the detail`)]),
    ]));
    root.append(detail);
  }

  function mount() {
    if (!document.getElementById("pace-grid-css")) {
      document.head.append(el("style", { id: "pace-grid-css", text: CSS }));
    }
    document.querySelectorAll("[data-pace-grid]").forEach(async (root) => {
      const src = root.getAttribute("data-src") || "data/pace.json";
      try {
        const res = await fetch(src, { cache: "no-store" });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        render(root, await res.json());
      } catch (err) {
        root.textContent = /404/.test(err.message) ? "This grid arrives with the next model update."
          : `The pace data did not load (${src}: ${err.message}).`;
      }
    });
  }

  window.PaceGrid = { render };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount); else mount();
})();
