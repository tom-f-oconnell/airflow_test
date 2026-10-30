// Dashboard behaviour: node polling, run polling, launch helpers, trace chart.
"use strict";

const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const seriesColor = (roiId) => css(SERIES[(roiId - 1) % SERIES.length]);

function fmtBytes(n) {
  if (n == null) return "–";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return (i === 0 ? n.toFixed(0) : n.toFixed(2)) + " " + units[i];
}

function memLevel(pct) {
  if (pct == null) return ["", "n/a"];
  if (pct >= 90) return ["critical", "HIGH"];
  if (pct >= 75) return ["serious", "WARN"];
  return ["good", "OK"];
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v; else if (k === "html") node.innerHTML = v; else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

function renderNodes(snapshot) {
  const body = document.querySelector("#nodes tbody");
  if (!body) return;
  body.innerHTML = "";
  if (snapshot.error) {
    body.append(el("tr", {}, el("td", { colspan: 8, class: "bad" }, "Docker socket unavailable: " + snapshot.error)));
  }
  for (const n of snapshot.nodes) {
    const limit = n.mem_cap_bytes || n.mem_limit_bytes;
    const pct = n.mem_used_bytes != null && limit ? (100 * n.mem_used_bytes / limit) : null;
    const [level, word] = memLevel(pct);
    const meter = el("div", { class: "meter" },
      el("div", { class: "bar" }, el("div", { class: "fill " + level, style: `width:${Math.min(100, pct || 0).toFixed(1)}%` })),
      el("span", { class: "txt" }, n.mem_used_bytes != null
        ? `${word} ${fmtBytes(n.mem_used_bytes)} / ${fmtBytes(limit)}${pct != null ? ` (${pct.toFixed(0)}%)` : ""}`
        : "–"));
    const stateLevel = n.status !== "running" ? (n.status === "exited" && n.service === "airflow-init" ? "" : "critical")
      : n.health === "unhealthy" ? "critical" : n.health === "starting" ? "warning" : n.health === "healthy" ? "good" : "";
    const state = el("span", { class: "badge " + stateLevel },
      n.status + (n.health ? " · " + n.health : "") + (n.oom_killed ? " · OOM-killed" : ""));
    body.append(el("tr", {},
      el("td", {}, el("strong", {}, n.service), el("div", { class: "muted" }, n.role)),
      el("td", { class: "mono" }, n.name),
      el("td", {}, state),
      el("td", { class: "num" }, n.cpu_percent != null ? n.cpu_percent.toFixed(1) + " %" : "–"),
      el("td", {}, meter),
      el("td", { class: "num" }, n.mem_cap_bytes ? fmtBytes(n.mem_cap_bytes) : "none"),
      el("td", { class: "num" }, n.cpus ? n.cpus.toFixed(1) : "–"),
      el("td", { class: "num" }, String(n.restart_count || 0))));
  }
  const stamp = document.getElementById("nodes-updated");
  if (stamp && snapshot.updated) stamp.textContent = "updated " + new Date(snapshot.updated * 1000).toLocaleTimeString();
  const project = document.getElementById("nodes-project");
  if (project && snapshot.project) project.textContent = snapshot.project;
}

async function pollNodes() {
  try {
    const resp = await fetch("/api/nodes");
    renderNodes(await resp.json());
  } catch (err) {
    renderNodes({ nodes: [], error: String(err) });
  }
}

function startNodePolling(ms) {
  pollNodes();
  setInterval(pollNodes, Math.max(2000, ms));
}

function renderRuns(data) {
  const body = document.querySelector("#runs tbody");
  if (!body || data.error) return;
  body.innerHTML = "";
  if (!data.runs.length) body.append(el("tr", {}, el("td", { colspan: 8, class: "muted" }, "no runs yet")));
  for (const r of data.runs) {
    const tasks = el("td", { class: "tasks", "data-dag": r.dag_id, "data-run": r.dag_run_id });
    body.append(el("tr", {},
      el("td", {}, r.dag_id),
      el("td", {}, el("a", { href: r.airflow_url, target: "_blank", rel: "noopener", class: "mono" }, r.dag_run_id)),
      el("td", {}, r.dataset ? el("a", { href: `/datasets/${r.dataset}?run=${r.result_dir}` }, r.dataset) : ""),
      el("td", {}, el("span", { class: "badge state-" + r.state }, r.state)),
      el("td", { class: "mono" }, (r.start_date || "").slice(0, 19)),
      el("td", { class: "num" }, r.duration ? r.duration.toFixed(0) + " s" : ""),
      tasks,
      el("td", {}, el("details", { class: "inline" }, el("summary", {}, "conf"), el("code", {}, r.conf_json)))));
  }
  fillTaskCells();
  const stamp = document.getElementById("runs-updated");
  if (stamp) stamp.textContent = "updated " + new Date().toLocaleTimeString();
}

async function fillTaskCells() {
  const cells = Array.from(document.querySelectorAll("td.tasks")).slice(0, 8);
  await Promise.all(cells.map(async (cell) => {
    try {
      const resp = await fetch(`/api/runs/${cell.dataset.dag}/${encodeURIComponent(cell.dataset.run)}/tasks`);
      const data = await resp.json();
      cell.innerHTML = "";
      for (const t of data.tasks || []) {
        cell.append(el("span", { class: "badge state-" + (t.state || "none"), title: `${t.task_id} on ${t.hostname || "?"} (${t.queue})` },
          `${t.task_id}: ${t.state || "–"}`), " ");
      }
    } catch (err) { /* leave the cell empty */ }
  }));
}

async function pollRuns() {
  try {
    const resp = await fetch("/api/runs");
    renderRuns(await resp.json());
  } catch (err) { /* keep the server-rendered table */ }
}

function startRunPolling(ms) {
  fillTaskCells();
  setInterval(pollRuns, ms);
}

function prefillFactorization(dataset) {
  const box = document.getElementById("launch-roi_factorization");
  if (!box) return;
  box.open = true;
  const input = box.querySelector("input[name=dataset]");
  if (input) input.value = dataset;
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------- traces

function initTraceChart() {
  const chart = document.getElementById("trace-chart");
  const svg = document.getElementById("trace-svg");
  const legend = document.getElementById("trace-legend");
  const tooltip = document.getElementById("trace-tooltip");
  const picks = Array.from(document.querySelectorAll(".roi-pick"));
  for (const sw of document.querySelectorAll(".swatch")) sw.style.background = seriesColor(Number(sw.dataset.roi));
  let data = null;

  async function load() {
    const ids = picks.filter((p) => p.checked).map((p) => p.value).slice(0, 8);
    picks.forEach((p) => { if (!p.checked && ids.length >= 8) p.disabled = true; else p.disabled = false; });
    if (!ids.length) { svg.innerHTML = ""; legend.innerHTML = ""; return; }
    const resp = await fetch(`${chart.dataset.url}?ids=${ids.join(",")}`);
    data = await resp.json();
    draw();
  }

  function draw() {
    if (!data) return;
    const W = svg.clientWidth || 800, H = 260, m = { l: 48, r: 12, t: 10, b: 26 };
    const ns = "http://www.w3.org/2000/svg";
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = "";
    const t = data.t, series = data.ids.map((id) => ({ id, v: data.traces[String(id)], color: seriesColor(id) }));
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const v of s.v) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!(hi > lo)) { hi = lo + 1; }
    const pad = (hi - lo) * 0.05; lo -= pad; hi += pad;
    const x = (i) => m.l + (t[i] / Math.max(1, t[t.length - 1])) * (W - m.l - m.r);
    const y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
    const add = (tag, attrs, parent = svg) => {
      const node = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
      parent.append(node); return node;
    };
    if (data.stimuli && data.stimuli.onsets) {
      const dur = data.stimuli.duration_frames || 1;
      for (const onsets of Object.values(data.stimuli.onsets)) for (const on of onsets) {
        const i0 = t.findIndex((f) => f >= on), i1 = t.findIndex((f) => f >= on + dur);
        if (i0 < 0) continue;
        add("rect", { class: "stim", x: x(i0), y: m.t, width: Math.max(1, x(i1 < 0 ? t.length - 1 : i1) - x(i0)), height: H - m.t - m.b });
      }
    }
    const ticks = 4;
    for (let k = 0; k <= ticks; k++) {
      const v = lo + (hi - lo) * k / ticks, yy = y(v);
      add("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: yy, y2: yy });
      const label = add("text", { class: "tick", x: m.l - 6, y: yy + 4, "text-anchor": "end" });
      label.textContent = v.toFixed(2);
    }
    add("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b });
    const fr = data.frame_rate_hz;
    for (let k = 0; k <= 5; k++) {
      const i = Math.round((t.length - 1) * k / 5), label = add("text", { class: "tick", x: x(i), y: H - m.b + 16, "text-anchor": "middle" });
      label.textContent = fr ? (t[i] / fr).toFixed(0) + " s" : "frame " + t[i];
    }
    for (const s of series) {
      const d = s.v.map((v, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1)).join(" ");
      add("path", { class: "line", d, stroke: s.color, "data-roi": s.id });
    }
    const cross = add("line", { class: "crosshair", x1: 0, x2: 0, y1: m.t, y2: H - m.b, visibility: "hidden" });
    legend.innerHTML = "";
    for (const s of series) {
      const key = el("span", { class: "key" }); key.style.background = s.color;
      legend.append(el("span", {}, key, `ROI ${s.id}`));
    }
    svg.onmousemove = (ev) => {
      const rect = svg.getBoundingClientRect();
      const px = (ev.clientX - rect.left) * (W / rect.width);
      const frac = Math.min(1, Math.max(0, (px - m.l) / (W - m.l - m.r)));
      const i = Math.round(frac * (t.length - 1));
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
      tooltip.hidden = false;
      tooltip.innerHTML = `<div class="row"><span>${fr ? (t[i] / fr).toFixed(1) + " s" : ""}</span><span>frame ${t[i]}</span></div>` +
        series.map((s) => `<div class="row"><span><span class="key" style="background:${s.color}"></span>ROI ${s.id}</span><span>${s.v[i].toFixed(3)}</span></div>`).join("");
      const left = Math.min(ev.clientX - rect.left + 14, rect.width - tooltip.offsetWidth - 8);
      tooltip.style.left = left + "px"; tooltip.style.top = (ev.clientY - rect.top + 14) + "px";
    };
    svg.onmouseleave = () => { tooltip.hidden = true; cross.setAttribute("visibility", "hidden"); };
  }

  picks.forEach((p) => p.addEventListener("change", load));
  window.addEventListener("resize", draw);
  load();
}
