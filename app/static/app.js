const $ = (id) => document.getElementById(id);
const MODELS = {
  features_lr: ["Признаци + Logistic Regression", "19 стилови признака, линеен модел"],
  features_lr_no_genre: ["Признаци без жанр + LR", "без скоби, числа и кавички"],
  features_rf: ["Признаци + Random Forest", "същите признаци, нелинеен модел"],
  tfidf_lr: ["TF-IDF + Logistic Regression", "символни n-грами от суровия текст (baseline)"],
};
const MIN_WORDS = 80;
const pct = (p) => `${Math.round(p * 100)}%`;
const fmt = (x) => (Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(2));
const wordCount = (s) => s.trim().split(/\s+/).filter(Boolean).length;

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Дивергентна скала: човек (синьо) ← сиво → ИИ (червено).
function tint(p, strength = 45) {
  const d = Math.abs(p - 0.5) * 2;
  const pole = p >= 0.5 ? "var(--ai)" : "var(--human)";
  return `color-mix(in oklab, ${pole} ${Math.round(d * strength)}%, var(--neutral))`;
}
const sideColor = (p) => (p >= 0.5 ? "var(--ai)" : "var(--human)");

/* --- тема --- */
const THEME_KEY = "detector-theme";
try {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved) document.documentElement.dataset.theme = saved;
} catch {}
$("theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  const next = dark ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem(THEME_KEY, next); } catch {}
});

/* --- вход --- */
function updateCount() {
  const n = wordCount($("text").value);
  $("count-n").textContent = n;
  $("count-fill").style.width = `${Math.min(100, (n / MIN_WORDS) * 100)}%`;
  $("count-fill").classList.toggle("ok", n >= MIN_WORDS);
}
$("text").addEventListener("input", updateCount);
$("clear").addEventListener("click", () => { $("text").value = ""; updateCount(); $("text").focus(); });

fetch("/examples").then((r) => r.json()).then((items) => {
  $("examples").innerHTML = items.map((e, i) =>
    `<button type="button" class="chip-btn" data-i="${i}"
      aria-label="Пример: ${e.label === "ai" ? "ИИ" : "човек"} (${escapeHtml(e.topic)})">
      <span class="dot" style="background:var(--${e.label === "ai" ? "ai" : "human"})"></span>Пример: ${e.label === "ai" ? "ИИ" : "човек"}</button>`).join("");
  $("examples").onclick = (ev) => {
    const btn = ev.target.closest("button");
    if (!btn) return;
    $("text").value = items[Number(btn.dataset.i)].text;
    updateCount();
    $("form").requestSubmit();
  };
}).catch(() => {});

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("text").value.trim();
  $("error").hidden = true;
  if (!text) { $("text").focus(); return; }
  const btn = $("submit");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.querySelector(".btn-label").textContent = "Анализирам…";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Текстът не може да бъде анализиран.");
    render(data);
  } catch (err) {
    $("error").textContent = err.message;
    $("error").hidden = false;
    $("empty").hidden = !$("result").hidden;
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.querySelector(".btn-label").textContent = "Анализирай";
  }
});

/* --- табове --- */
document.querySelector(".tabs").addEventListener("click", (e) => {
  const tab = e.target.closest("[role=tab]");
  if (!tab) return;
  selectTab(tab.dataset.tab);
});
function selectTab(name) {
  document.querySelectorAll("[role=tab]").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === name)));
  document.querySelectorAll("[data-panel]").forEach((p) => { p.hidden = p.dataset.panel !== name; });
  if (name === "sentences" && lastSegments) drawChart(lastSegments);
}

/* --- резултат --- */
let lastSegments = null;

function render(d) {
  $("empty").hidden = true;
  $("result").hidden = false;
  const p = d.ai_probability;

  const value = $("gauge-value");
  value.style.stroke = sideColor(p);
  value.style.strokeDasharray = `${Math.max(0.5, p * 100)} 100`;
  $("gauge-num").textContent = pct(p);
  $("gauge").setAttribute("aria-label", `Вероятност текстът да е от ИИ: ${pct(p)}`);

  const chip = $("verdict-chip");
  const short = d.word_count < MIN_WORDS / 2;
  chip.className = `verdict-chip ${short ? "" : p >= 0.6 ? "ai" : p <= 0.4 ? "human" : ""}`;
  chip.textContent = short ? "ненадеждно" : p >= 0.6 ? "по-скоро ИИ" : p <= 0.4 ? "по-скоро човек" : "неясно";
  $("verdict").textContent = d.verdict;
  $("summary-meta").textContent = `${d.word_count} думи · ${d.calibrated ? "калибрирана вероятност" : "некалибрирана вероятност"} · основен модел: признаци + LR`;
  $("warnings").innerHTML = d.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("");

  renderReasons(d);
  renderFeatures(d.features);
  renderModels(d);
  lastSegments = d.segments;
  renderSentences(d.segments);
  selectTab("why");
}

function divergeBar(contrib, max) {
  const w = Math.min(50, (Math.abs(contrib) / max) * 50);
  const side = contrib > 0 ? `left:50%;width:${w}%;background:var(--ai)` : `right:50%;width:${w}%;background:var(--human)`;
  return `<div class="diverge" aria-hidden="true"><span style="${side}"></span></div>`;
}

function renderReasons(d) {
  const max = Math.max(...d.features.map((f) => Math.abs(f.contribution)), 0.01);
  $("reasons").innerHTML = d.top_features.map((f) => `
    <li>
      <div class="reason-head">
        <span class="reason-title">${escapeHtml(f.label)}</span>
        <span class="reason-dir ${f.toward === "ai" ? "ai-ink" : "human-ink"}">${f.toward === "ai" ? "към ИИ" : "към човек"}</span>
      </div>
      <p class="reason-text">${escapeHtml(f.text)}</p>
      ${divergeBar(f.contribution, max)}
    </li>`).join("");
}

function renderFeatures(features) {
  const max = Math.max(...features.map((f) => Math.abs(f.contribution)), 0.01);
  const rows = [...features].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  $("features").innerHTML = `<thead><tr><th>Признак</th><th class="num">Текст</th>
      <th class="num hide-sm">Човек ср.</th><th class="num hide-sm">ИИ ср.</th><th>Принос</th></tr></thead><tbody>` +
    rows.map((f) => `<tr><td>${escapeHtml(f.label)}</td><td class="num"><strong>${fmt(f.value)}</strong></td>
      <td class="num hide-sm human-ink">${fmt(f.human_mean)}</td><td class="num hide-sm ai-ink">${fmt(f.ai_mean)}</td>
      <td class="bar-cell" title="${f.contribution > 0 ? "+" : ""}${f.contribution.toFixed(2)}">${divergeBar(f.contribution, max)}</td></tr>`).join("") +
    `</tbody>`;
}

function renderModels(d) {
  $("models").innerHTML = Object.entries(d.model_probabilities).map(([k, p]) => {
    const [name, desc] = MODELS[k] || [k, ""];
    const shown = k === d.model ? d.ai_probability : p;
    return `<li>
      <div class="model-head"><span>${name}${k === d.model ? '<span class="main-tag">основен</span>' : ""}</span>
        <span class="num" style="color:${sideColor(shown)}">${pct(shown)}</span></div>
      <div class="prob-track"><span style="width:${shown * 100}%;background:${tint(shown, 90)}"></span></div>
      <div class="muted small" style="margin-top:6px">${desc}</div></li>`;
  }).join("");
}

function renderSentences(segs) {
  const tab = document.querySelector('[data-tab="sentences"]');
  tab.disabled = segs.length === 0;
  tab.title = segs.length ? "" : "Нужни са поне 4 изречения";
  $("sentences").innerHTML = segs.map((s, i) =>
    `<span data-i="${i}" style="background:${tint(s.ai_probability)}">${escapeHtml(s.sentence)}</span>`).join(" ");
}

function drawChart(segs) {
  const svg = $("chart");
  if (!segs.length) { svg.innerHTML = ""; return; }
  const W = svg.clientWidth || 600, H = 160, pad = { l: 38, r: 12, t: 12, b: 24 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const x = (i) => pad.l + (segs.length === 1 ? 0 : (i / (segs.length - 1)) * (W - pad.l - pad.r));
  const y = (p) => pad.t + (1 - p) * (H - pad.t - pad.b);
  const step = segs.length > 1 ? x(1) - x(0) : W;

  let out = "";
  for (const p of [0, 1]) {
    out += `<line class="grid" x1="${pad.l}" x2="${W - pad.r}" y1="${y(p)}" y2="${y(p)}"/>`;
  }
  out += `<line class="mid" x1="${pad.l}" x2="${W - pad.r}" y1="${y(0.5)}" y2="${y(0.5)}"/>`;
  for (const [p, label] of [[1, "ИИ"], [0.5, "50%"], [0, "човек"]]) {
    out += `<text class="axis-label" x="${pad.l - 6}" y="${y(p) + 4}" text-anchor="end">${label}</text>`;
  }
  out += `<text class="axis-label" x="${pad.l}" y="${H - 4}">изречение 1</text>`;
  out += `<text class="axis-label" x="${W - pad.r}" y="${H - 4}" text-anchor="end">изречение ${segs.length}</text>`;
  out += `<polyline class="line" points="${segs.map((s, i) => `${x(i)},${y(s.ai_probability)}`).join(" ")}"/>`;
  segs.forEach((s, i) => {
    out += `<rect class="hit" data-i="${i}" x="${x(i) - step / 2}" y="0" width="${step}" height="${H}"/>`;
    out += `<circle class="dot" data-i="${i}" cx="${x(i)}" cy="${y(s.ai_probability)}" r="4.5" fill="${sideColor(s.ai_probability)}"/>`;
  });
  svg.innerHTML = out;

  const tip = $("tooltip");
  const spans = $("sentences").children;
  const dots = svg.querySelectorAll(".dot");
  const highlight = (i) => {
    [...spans].forEach((el, j) => el.classList.toggle("active", j === i));
    dots.forEach((el, j) => el.classList.toggle("active", j === i));
  };
  svg.onmousemove = (e) => {
    if (!e.target.classList.contains("hit")) return;
    const i = Number(e.target.dataset.i);
    const s = segs[i];
    tip.innerHTML = `<strong>Изречение ${i + 1} · ${pct(s.ai_probability)} ИИ</strong><br>${escapeHtml(s.sentence.slice(0, 140))}${s.sentence.length > 140 ? "…" : ""}`;
    tip.hidden = false;
    tip.style.left = `${Math.min(e.clientX + 14, window.innerWidth - 336)}px`;
    tip.style.top = `${e.clientY + 14}px`;
    highlight(i);
  };
  svg.onmouseleave = () => { tip.hidden = true; highlight(-1); };
}

window.addEventListener("resize", () => {
  if (lastSegments && !document.querySelector('[data-panel="sentences"]').hidden) drawChart(lastSegments);
});
