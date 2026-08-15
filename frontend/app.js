const btn = document.getElementById("analyze-btn");
const input = document.getElementById("essay-input");
const status = document.getElementById("status");
const summaryEl = document.getElementById("doc-summary");
const resultsSection = document.getElementById("results-section");
const essayEl = document.getElementById("highlighted-essay");
const flaggedEl = document.getElementById("flagged-list");

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------------- opening splash ----------------
   Icon assembles, holds for exactly 2s, then cracks along its middle
   seam while the two full-bleed panels slide apart to reveal the page.
   Plays on every load (no once-per-session suppression). */
function initSplash() {
  const splash = document.getElementById("splash");
  if (!splash) return;

  if (reducedMotion) {
    splash.remove();
    return;
  }

  const HOLD_MS = 2000;
  const prevOverflow = document.body.style.overflow;
  document.body.style.overflow = "hidden";

  requestAnimationFrame(() => splash.classList.add("show-icon"));
  setTimeout(() => splash.classList.add("breaking"), HOLD_MS);
  setTimeout(() => {
    splash.classList.add("fade-out");
    document.body.style.overflow = prevOverflow;
  }, HOLD_MS + 1500);
  setTimeout(() => splash.remove(), HOLD_MS + 2050);
}

initSplash();

/* ---------------- color mapping ---------------- */
function probToColor(p) {
  // Low-probability sentences stay near-invisible so ordinary human prose
  // reads cleanly; only elevated AI-likelihood visibly tints, and climbs
  // toward a saturated red as it approaches certainty. Continuous curve,
  // not a binary flag color -- the highlight reflects a likelihood.
  const t = Math.max(0, Math.min(1, p));
  const r = Math.round(90 + t * (239 - 90));
  const g = Math.round(100 + t * (68 - 100));
  const b = Math.round(140 + t * (68 - 140));
  const alpha = Math.pow(t, 1.6) * 0.62;
  return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(2)})`;
}

function probGlow(p) {
  if (p < 0.75) return "none";
  const t = (p - 0.75) / 0.25;
  return `0 0 ${(6 + t * 10).toFixed(0)}px rgba(239, 68, 68, ${(0.15 + t * 0.25).toFixed(2)})`;
}

/* ---------------- analyze flow ---------------- */
btn.addEventListener("click", async () => {
  const text = input.value.trim();
  if (!text) {
    setStatus("Paste an essay first.", true);
    return;
  }
  setStatus("Analyzing… (first request loads the model, can take a bit)", false);
  setLoading(true);
  resultsSection.hidden = true;
  essayEl.innerHTML = "";
  flaggedEl.innerHTML = "";
  summaryEl.innerHTML = "";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    render(data);
    setStatus("", false);
  } catch (e) {
    setStatus("Error: " + e.message, true);
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  btn.disabled = isLoading;
  btn.classList.toggle("loading", isLoading);
}

function setStatus(msg, isError) {
  status.textContent = msg;
  status.classList.toggle("error", !!isError);
}

/* ---------------- rendering ---------------- */
function render(data) {
  const { sentences, doc_summary } = data;

  renderSummary(doc_summary);

  essayEl.innerHTML = "";
  sentences.forEach((s, i) => {
    const span = document.createElement("span");
    span.className = "sent";
    span.style.backgroundColor = probToColor(s.prob_ai);
    span.style.boxShadow = probGlow(s.prob_ai);
    span.style.setProperty("--d", reducedMotion ? "0ms" : `${Math.min(i * 18, 600)}ms`);
    span.title = `AI-likelihood: ${(s.prob_ai * 100).toFixed(0)}%`;
    span.textContent = s.text + " ";
    essayEl.appendChild(span);
  });

  flaggedEl.innerHTML = "";
  const flagged = sentences.filter((s) => s.predicted_ai);
  if (flagged.length === 0) {
    const note = document.createElement("div");
    note.className = "empty-note";
    note.textContent = "No sentences were flagged as likely AI-written.";
    flaggedEl.appendChild(note);
  } else {
    flagged.forEach((s, i) => {
      const card = document.createElement("div");
      card.className = "flag-card tilt";
      card.style.setProperty("--d", reducedMotion ? "0ms" : `${Math.min(i * 60, 500)}ms`);
      const pct = (s.prob_ai * 100).toFixed(0);
      card.innerHTML = `
        <span class="prob">${pct}% AI-likelihood</span>
        <div class="sent-text">"${escapeHtml(s.text)}"</div>
        <ul>${s.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>
      `;
      attachTilt(card);
      flaggedEl.appendChild(card);
    });
  }

  resultsSection.hidden = false;
}

function renderSummary(summary) {
  summaryEl.innerHTML = "";
  if (!summary || !summary.n_sentences) return;

  const stats = [
    { label: "Sentences analyzed", value: summary.n_sentences },
    { label: "Flagged as likely AI", value: summary.n_flagged },
    { label: "Mean AI-likelihood", value: Math.round(summary.mean_prob_ai * 100), suffix: "%" },
  ];

  stats.forEach((s, i) => {
    const el = document.createElement("div");
    el.className = "stat";
    el.style.animationDelay = reducedMotion ? "0ms" : `${i * 90}ms`;
    el.innerHTML = `<div class="num">0</div><div class="label">${escapeHtml(s.label)}</div>`;
    summaryEl.appendChild(el);
    animateNumber(el.querySelector(".num"), s.value, s.suffix || "");
  });

  const note = document.createElement("div");
  note.className = "summary-note";
  note.textContent = "Supporting context, not a single verdict — see the sentence-level detail below.";
  summaryEl.appendChild(note);
}

function animateNumber(el, target, suffix) {
  if (reducedMotion || target === 0) {
    el.textContent = target + suffix;
    return;
  }
  const duration = 700;
  const start = performance.now();
  function tick(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(eased * target) + suffix;
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/* ---------------- 3D tilt-on-hover ---------------- */
function attachTilt(el) {
  if (!el || reducedMotion) return;
  const maxTilt = 5;
  el.addEventListener("mousemove", (e) => {
    el.classList.remove("tilt-reset");
    const rect = el.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width;
    const py = (e.clientY - rect.top) / rect.height;
    el.style.setProperty("--rx", ((0.5 - py) * maxTilt * 2).toFixed(2) + "deg");
    el.style.setProperty("--ry", ((px - 0.5) * maxTilt * 2).toFixed(2) + "deg");
  });
  el.addEventListener("mouseleave", () => {
    // snap back with a touch of elastic overshoot rather than a linear ease
    el.classList.add("tilt-reset");
    el.style.setProperty("--rx", "0deg");
    el.style.setProperty("--ry", "0deg");
  });
}

document.querySelectorAll(".tilt").forEach(attachTilt);

// Digits, Latin letters, Greek letters, and common math symbols -- the
// pool the drifting-character background draws from.
const CHAR_POOL =
  "0123456789" +
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" +
  "αβγδεζηθικλμνξοπρστυφχψω" +
  "ΓΔΘΛΞΠΣΦΨΩ" +
  "∑∫√∞≈≠±×÷∂∇∈∉⊂⊃≤≥∀∃∅∪∩";

/* ---------------- animated particle backdrop ----------------
   A field of drifting digits, letters, Greek letters, and math symbols
   at varying sizes -- reacts to the cursor (nearby characters drift away
   and brighten). Plus a handful of thin line fragments and soft dots
   underneath for extra depth. */
function initParticles() {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");
  let w, h, dpr, fragments, dots, chars;
  const mouse = { x: null, y: null };

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function initShapes() {
    const fragCount = Math.min(16, Math.max(8, Math.floor((w * h) / 110000)));
    fragments = Array.from({ length: fragCount }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      len: 26 + Math.random() * 46,
      angle: Math.random() * Math.PI,
      spin: (Math.random() - 0.5) * 0.0006,
      vx: (Math.random() - 0.5) * 0.045,
      vy: (Math.random() - 0.5) * 0.045,
      alpha: 0.06 + Math.random() * 0.1,
    }));

    const dotCount = Math.min(28, Math.floor((w * h) / 70000));
    dots = Array.from({ length: dotCount }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 1.1 + 0.5,
      vx: (Math.random() - 0.5) * 0.05,
      vy: (Math.random() - 0.5) * 0.05,
      alpha: 0.15 + Math.random() * 0.25,
    }));
  }

  // The character field: digits/letters/Greek/math symbols drifting
  // slowly in random directions, at a wide range of sizes, occasionally
  // swapping to a different random character. Nearby ones get pushed
  // away from the cursor and brighten as it approaches.
  function initChars() {
    const count = Math.min(160, Math.max(70, Math.floor((w * h) / 8500)));
    chars = Array.from({ length: count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.14,
      vy: (Math.random() - 0.5) * 0.14,
      char: CHAR_POOL[Math.floor(Math.random() * CHAR_POOL.length)],
      size: 11 + Math.pow(Math.random(), 1.6) * 35,
      baseAlpha: 0.07 + Math.random() * 0.2,
      tint: Math.random(),
      rotation: (Math.random() - 0.5) * 0.5,
      swap: 200 + Math.random() * 500,
    }));
  }

  function charTint(c) {
    if (c.tint < 0.6) return "210, 215, 235";
    if (c.tint < 0.82) return "167, 139, 250";
    return "94, 210, 235";
  }

  function stepChars() {
    for (const c of chars) {
      c.x += c.vx;
      c.y += c.vy;
      if (c.x < -30) c.x = w + 30; else if (c.x > w + 30) c.x = -30;
      if (c.y < -30) c.y = h + 30; else if (c.y > h + 30) c.y = -30;

      if (mouse.x != null) {
        const dx = c.x - mouse.x, dy = c.y - mouse.y;
        const dist = Math.hypot(dx, dy) || 1;
        const radius = 140;
        if (dist < radius) {
          const force = ((radius - dist) / radius) * 0.05;
          c.vx += (dx / dist) * force;
          c.vy += (dy / dist) * force;
        }
      }
      c.vx *= 0.985;
      c.vy *= 0.985;

      c.swap -= 1;
      if (c.swap <= 0) {
        c.char = CHAR_POOL[Math.floor(Math.random() * CHAR_POOL.length)];
        c.swap = 250 + Math.random() * 550;
      }
    }
  }

  function drawChars() {
    for (const c of chars) {
      let alpha = c.baseAlpha;
      if (mouse.x != null) {
        const dist = Math.hypot(c.x - mouse.x, c.y - mouse.y);
        if (dist < 160) alpha = Math.min(0.95, alpha + ((160 - dist) / 160) * 0.55);
      }
      ctx.save();
      ctx.translate(c.x, c.y);
      ctx.rotate(c.rotation);
      ctx.font = `${c.size.toFixed(0)}px "SFMono-Regular", Consolas, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = `rgba(${charTint(c)}, ${alpha.toFixed(2)})`;
      ctx.fillText(c.char, 0, 0);
      ctx.restore();
    }
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(210, 215, 235, 1)";
    ctx.lineCap = "round";
    for (const f of fragments) {
      ctx.globalAlpha = f.alpha;
      ctx.lineWidth = 1;
      const dx = Math.cos(f.angle) * f.len * 0.5;
      const dy = Math.sin(f.angle) * f.len * 0.5;
      ctx.beginPath();
      ctx.moveTo(f.x - dx, f.y - dy);
      ctx.lineTo(f.x + dx, f.y + dy);
      ctx.stroke();
    }

    ctx.fillStyle = "rgba(210, 215, 235, 1)";
    for (const d of dots) {
      ctx.globalAlpha = d.alpha;
      ctx.beginPath();
      ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    drawChars();
  }

  function step() {
    for (const f of fragments) {
      f.x += f.vx;
      f.y += f.vy;
      f.angle += f.spin;
      if (f.x < -60) f.x = w + 60; else if (f.x > w + 60) f.x = -60;
      if (f.y < -60) f.y = h + 60; else if (f.y > h + 60) f.y = -60;
    }
    for (const d of dots) {
      d.x += d.vx;
      d.y += d.vy;
      if (d.x < 0) d.x = w; else if (d.x > w) d.x = 0;
      if (d.y < 0) d.y = h; else if (d.y > h) d.y = 0;
    }
    stepChars();
    draw();
    requestAnimationFrame(step);
  }

  resize();
  initShapes();
  initChars();
  window.addEventListener("resize", () => { resize(); initShapes(); initChars(); });

  if (reducedMotion) {
    draw();
    return;
  }

  window.addEventListener("mousemove", (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });
  window.addEventListener("mouseleave", () => { mouse.x = null; mouse.y = null; });
  step();
}

initParticles();

/* ---------------- parallax for floating 3D shards ---------------- */
function initParallax() {
  if (reducedMotion) return;
  const wraps = Array.from(document.querySelectorAll(".shard-wrap"));
  if (!wraps.length) return;

  let targetX = 0, targetY = 0, curX = 0, curY = 0;

  window.addEventListener("mousemove", (e) => {
    targetX = (e.clientX / window.innerWidth - 0.5) * 2;
    targetY = (e.clientY / window.innerHeight - 0.5) * 2;
  });

  function tick() {
    curX += (targetX - curX) * 0.05;
    curY += (targetY - curY) * 0.05;
    for (const wrap of wraps) {
      const depth = parseFloat(wrap.dataset.depth || "0.03");
      const dx = curX * depth * 400;
      const dy = curY * depth * 260;
      wrap.style.transform = `translate3d(${dx.toFixed(1)}px, ${dy.toFixed(1)}px, 0)`;
    }
    requestAnimationFrame(tick);
  }
  tick();
}

initParallax();

/* ==================================================================
   Auth (login/signup/logout) + essay history.
   A tiny hash router drives four views: home, login, signup, history.
   #/history/:id loads a saved analysis into the SAME render() used for
   a live analyze() call, rather than duplicating the results markup.
   ================================================================== */

const authArea = document.getElementById("auth-area");
const views = Array.from(document.querySelectorAll("[data-view]"));
let currentUser = null;

async function apiJson(url, options) {
  const res = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options && options.headers) },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

async function refreshAuthState() {
  try {
    const res = await fetch("/api/auth/me");
    currentUser = res.ok ? await res.json() : null;
  } catch {
    currentUser = null;
  }
  renderAuthArea();
}

function renderAuthArea() {
  if (currentUser) {
    authArea.innerHTML = `
      <a href="#/history">History</a>
      <span class="username-pill">${escapeHtml(currentUser.username)}</span>
      <button type="button" class="ghost-btn" id="logout-btn">Log out</button>
    `;
    document.getElementById("logout-btn").addEventListener("click", async () => {
      await fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
      currentUser = null;
      renderAuthArea();
      location.hash = "#/";
    });
  } else {
    authArea.innerHTML = `
      <a href="#/login">Log in</a>
      <a href="#/signup" class="signup-link">Sign up</a>
    `;
  }
}

function showView(name) {
  for (const v of views) {
    v.hidden = v.dataset.view !== name;
  }
}

function setupAuthForm(formId, errorId, endpoint, onSuccess) {
  const form = document.getElementById(formId);
  if (!form) return;
  const errorEl = document.getElementById(errorId);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorEl.textContent = "";
    const submitBtn = form.querySelector("button[type=submit]");
    const username = form.querySelector('input[type="text"]').value.trim();
    const password = form.querySelector('input[type="password"]').value;
    submitBtn.disabled = true;
    submitBtn.classList.add("loading");
    try {
      const data = await apiJson(endpoint, { method: "POST", body: JSON.stringify({ username, password }) });
      currentUser = { username: data.username };
      renderAuthArea();
      onSuccess();
    } catch (err) {
      errorEl.textContent = err.message;
    } finally {
      submitBtn.disabled = false;
      submitBtn.classList.remove("loading");
    }
  });
}

setupAuthForm("login-form", "login-error", "/api/auth/login", () => { location.hash = "#/"; });
setupAuthForm("signup-form", "signup-error", "/api/auth/signup", () => { location.hash = "#/"; });

function historyItemMarkup(item, index) {
  const pct = item.mean_prob_ai != null ? Math.round(item.mean_prob_ai * 100) : null;
  const date = new Date(item.created_at).toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
  });
  return `
    <button type="button" class="history-item" data-id="${item.id}" style="--d:${reducedMotion ? 0 : index * 45}ms">
      <div class="hi-text">
        <div class="hi-preview">${escapeHtml(item.preview || "(empty)")}</div>
        <div class="hi-meta">${date} · ${item.n_sentences} sentences</div>
      </div>
      <div class="hi-stat">
        <strong>${item.n_flagged}</strong>
        flagged${pct != null ? ` · ${pct}%` : ""}
      </div>
    </button>
  `;
}

async function loadHistoryList() {
  const listEl = document.getElementById("history-list");
  listEl.innerHTML = `<div class="empty-note">Loading…</div>`;
  try {
    const data = await apiJson("/api/history");
    if (!data.analyses.length) {
      listEl.innerHTML = `<div class="empty-note">No essays analyzed yet. Once you're logged in, everything you analyze is saved here.</div>`;
      return;
    }
    listEl.innerHTML = data.analyses.map(historyItemMarkup).join("");
    listEl.querySelectorAll(".history-item").forEach((el) => {
      el.addEventListener("click", () => { location.hash = `#/history/${el.dataset.id}`; });
    });
  } catch (err) {
    listEl.innerHTML = `<div class="empty-note">Couldn't load history: ${escapeHtml(err.message)}</div>`;
  }
}

async function loadHistoryDetail(id) {
  showView("home");
  setStatus("Loading saved analysis…", false);
  try {
    const data = await apiJson(`/api/history/${id}`);
    input.value = data.essay_text || "";
    render(data);
    setStatus("", false);
  } catch (err) {
    setStatus("Error: " + err.message, true);
  }
}

async function router() {
  const hash = location.hash || "#/";
  const historyMatch = hash.match(/^#\/history\/(\d+)$/);

  if (hash === "#/login") {
    if (currentUser) { location.hash = "#/"; return; }
    showView("login");
  } else if (hash === "#/signup") {
    if (currentUser) { location.hash = "#/"; return; }
    showView("signup");
  } else if (hash === "#/history") {
    if (!currentUser) { location.hash = "#/login"; return; }
    showView("history");
    loadHistoryList();
  } else if (historyMatch) {
    if (!currentUser) { location.hash = "#/login"; return; }
    loadHistoryDetail(historyMatch[1]);
  } else {
    showView("home");
  }
}

window.addEventListener("hashchange", router);

(async () => {
  await refreshAuthState();
  router();
})();
