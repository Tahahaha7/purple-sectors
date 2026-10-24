/* pace-grid.js — where would each driver qualify in each car?
 *
 * Drop-in: <div data-pace-grid data-src="data/pace.json"></div> <script src="pace-grid.js" defer></script>
 * Reads site/data/pace.json written by `python -m purple_sectors.pace.export`.
 * Rows are cars (fastest at the top), columns are drivers (fastest on the left).
 * Inherits the page's font and text colour; override the tokens from your own CSS if you like:
 *   --pg-front (front of the grid), --pg-back (back of the grid), --pg-accent (selection),
 *   --pg-line (hairlines), --pg-muted (secondary text).
 * Add data-no-heading when the page already has a heading for the section.
 * No dependencies, no build step. pace.json carries seconds on a 90-second lap.
 */
(function () {
  "use strict";

  const CSS = `
  .pg { font-family: var(--sans, sans-serif); line-height: 1.55; --front: var(--pg-front, #5b9dff); --back: var(--pg-back, #c98b4a); }
  .pg h2 { font-size: 1.35rem; font-weight: 600; margin: 0 0 .25rem; }
  .pg > p { font-size: 13px; margin: 0 0 1rem; color: var(--pg-muted, color-mix(in srgb, currentColor 62%, transparent)); max-width: 60ch; }
  .pg-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .pg table { border-collapse: separate; border-spacing: 3px; font-variant-numeric: tabular-nums; }
  .pg th, .pg td { padding: 0; text-align: center; font-weight: 400; }
  .pg thead th { padding-bottom: .3rem; vertical-align: bottom; }
  .pg tbody th { text-align: left; padding-right: .7rem; position: sticky; left: 0; z-index: 2; background: var(--bg, #17161a); white-space: nowrap;
                 font-size: .82rem; color: var(--pg-muted, color-mix(in srgb, currentColor 62%, transparent)); }
  .pg tbody th b { font-weight: 600; color: inherit; }
  .pg tbody tr:first-child th b { color: var(--front); }
  .pg .pg-name { all: unset; cursor: pointer; display: block; padding: .5rem .2rem; border-radius: 0; font-size: .75rem;
                 font-weight: 600; letter-spacing: .02em; }
  .pg .pg-name:hover, .pg .pg-name:focus-visible { outline: 2px solid var(--pg-accent, #7c3aed); outline-offset: 1px; }
  .pg .pg-name[aria-pressed="true"] { background: var(--pg-accent, #c5a1ef); color: var(--bg, #17161a); }
  .pg .pg-cell { all: unset; cursor: pointer; display: grid; place-items: center; width: 2.45rem; height: 2.3rem; border-radius: 0;
                 font-family: var(--mono, monospace); font-size: .75rem; box-sizing: border-box;
                 border: 1px solid var(--pg-line, color-mix(in srgb, currentColor 14%, transparent)); }
  .pg .pg-cell:focus-visible { outline: 2px solid currentColor; outline-offset: 2px; }
  .pg .pg-cell.pg-seat { border: 2px solid currentColor; }
  .pg .pg-cell.pg-guess { border-style: dashed; }
  .pg .pg-cell[aria-pressed="true"] { outline: 3px solid var(--pg-accent, #7c3aed); outline-offset: 1px; }
  .pg-key { display: flex; flex-wrap: wrap; gap: 1.1rem; margin: .8rem 0 0; font-size: 11px;
            color: var(--pg-muted, color-mix(in srgb, currentColor 62%, transparent)); }
  .pg-key span { display: inline-flex; align-items: center; gap: .4rem; }
  .pg-key i { display: inline-block; width: 1rem; height: .9rem; border-radius: 3px; box-sizing: border-box;
              border: 1px solid var(--pg-line, color-mix(in srgb, currentColor 14%, transparent)); }
  .pg-detail { min-height: 5.2rem; margin: 0 0 1.2rem; padding: 1rem 0; font-size: 13px;
               border-bottom: 1px solid var(--pg-line, color-mix(in srgb, currentColor 14%, transparent)); }
  @media (max-width: 640px) { .pg .pg-cell { width: 2.5rem; height: 2.75rem; font-size: .75rem; } .pg .pg-name { min-height: 44px; display: grid; place-items: center; } .pg tbody th { max-width: 125px; font-size: 11px; white-space: normal; } }
  `;

  const fmtS = (sec) => sec.toFixed(2);
  const pos = (p) => `P${p}`;

  // Two quiet hues: the front of the grid in blue, the back in a warm tone, the midfield almost bare.
  function fill(p, n) {
    const share = 1 - (p - 1) / Math.max(1, n - 1);           // 1 at pole, 0 at the back
    const k = Math.abs(share - 0.5) * 2;                       // 0 in the midfield, 1 at either end
    const hue = share >= 0.5 ? "var(--front)" : "var(--back)";
    const alpha = Math.round(6 + k * (share >= 0.5 ? 54 : 48));
    return { background: `color-mix(in srgb, ${hue} ${alpha}%, transparent)`, color: "inherit" };
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
    const where = c.seat ? `in the ${team} (their current seat)` : `in the ${team}`;
    const guess = c.seen ? `` : ` ${who} has never driven this car, so the fit between them is a guess from the prior.`;
    return `${who} ${where} would qualify around ${pos(c.pos[1])} (${pos(c.pos[0])} to ${pos(c.pos[2])}), `
         + `${Math.abs(behind).toFixed(2)} s ${behind >= 0 ? "behind" : "ahead of"} the quickest seat on the grid.${guess}`;
  }

  function swapSentence(data, a, b) {
    const A = data.drivers[a], B = data.drivers[b];
    const ja = data.grid.cars.indexOf(A.short), jb = data.grid.cars.indexOf(B.short);
    if (ja === jb) return `${A.name} and ${B.name} are teammates already. Pick a driver from another team to compare placements.`;
    const aNow = data.grid.cells[a][ja], aThere = data.grid.cells[a][jb];
    const bNow = data.grid.cells[b][jb], bThere = data.grid.cells[b][ja];
    return `Compare placements: ${A.name} goes from ${pos(aNow.pos[1])} in the ${A.team} to about ${pos(aThere.pos[1])} `
         + `in the ${B.team} (${pos(aThere.pos[0])} to ${pos(aThere.pos[2])}); ${B.name} goes from ${pos(bNow.pos[1])} `
         + `to about ${pos(bThere.pos[1])} in the ${A.team} (${pos(bThere.pos[0])} to ${pos(bThere.pos[2])}). Each estimate keeps the rest of the grid in its current seats.`;
  }

  function render(root, data) {
    const n = data.drivers.length;
    root.classList.add("pg");
    root.replaceChildren();

    if (!root.hasAttribute("data-no-heading")) {
      root.append(el("h2", { text: "Where would each driver qualify in each car?" }));
      root.append(el("p", { text: `Cars down the side, fastest at the top; drivers across, fastest on the left. `
        + `Every square puts one driver into one car against the rest of today's grid. `
        + `Click a square for its estimate, or two driver headings to compare placements in each other's cars. `
        + `After ${data.event || "round " + data.round} ${data.season}.` }));
    }

    const detail = el("div", { class: "pg-detail", "aria-live": "polite",
      text: "Pick a square. Blue is the front of the grid, warm is the back; a bold border marks where the driver actually sits." });
    const picked = [];
    let pressedCell = null;
    const nameButtons = [];

    const head = el("tr", {}, [el("th", { text: "" })]);
    data.drivers.forEach((d, i) => {
      const name = el("button", { class: "pg-name", type: "button", "aria-pressed": "false", text: d.code,
        title: `${d.name}, ${d.team}: ${fmtS(d.skill[1])} s against the average driver, `
             + `${d.moved ? "has driven more than one team" : "one team only, so this rests partly on the prior"}` });
      name.addEventListener("click", () => {
        const k = picked.indexOf(i);
        if (k >= 0) picked.splice(k, 1); else { picked.push(i); if (picked.length > 2) picked.shift(); }
        nameButtons.forEach((b, idx) => b.setAttribute("aria-pressed", String(picked.includes(idx))));
        detail.textContent = picked.length === 2 ? swapSentence(data, picked[0], picked[1])
          : picked.length === 1 ? `${data.drivers[picked[0]].name} selected. Pick a second driver to compare estimates in each other's cars.`
          : "Pick a square, or two driver headings to compare placements.";
      });
      nameButtons.push(name);
      head.append(el("th", { scope: "col" }, [name]));
    });

    const body = el("tbody");
    data.grid.cars.forEach((code, j) => {
      const tr = el("tr", {}, [el("th", { scope: "row", title: data.grid.teams[j] },
        [el("b", { text: code }), document.createTextNode(" " + data.grid.teams[j])])]);
      data.drivers.forEach((d, i) => {
        const c = data.grid.cells[i][j];
        const btn = el("button", { class: "pg-cell" + (c.seat ? " pg-seat" : "") + (c.seen ? "" : " pg-guess"),
          type: "button", "aria-pressed": "false", style: fill(c.pos[1], n), text: String(c.pos[1]),
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
    root.append(detail);
    root.append(el("div", { class: "pg-scroll" }, [table]));
    root.append(el("div", { class: "pg-key" }, [
      el("span", {}, [el("i", { style: fill(1, n) }), document.createTextNode("front of the grid")]),
      el("span", {}, [el("i", { style: fill(n, n) }), document.createTextNode("back of the grid")]),
      el("span", {}, [el("i", { style: { border: "2px solid currentColor" } }), document.createTextNode("actual seat")]),
      el("span", {}, [el("i", { style: { borderStyle: "dashed" } }), document.createTextNode("never driven that car")]),
      el("span", {}, [document.createTextNode(`${data.units.interval} intervals in the detail`)]),
    ]));
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
