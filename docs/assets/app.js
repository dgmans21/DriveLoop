// DriveLoop 결과 페이지: 조건 탭 · 영상 위 오버레이 · 2D 지도 · 타임라인 (의존성 없음)
(() => {
  const $ = (s) => document.querySelector(s);
  const CONDS = ["clear_noon", "rain_noon", "clear_night", "rain_night"];
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const STATE_VAR = { RED: "--tl-red", YELLOW: "--tl-yellow", GREEN: "--tl-green", UNKNOWN: "--tl-unknown" };
  const CLS_VAR = { vehicle: "--vehicle", tl_red: "--tl-red", tl_yellow: "--tl-yellow", tl_green: "--tl-green" };
  const SHORT = { vehicle: "car", tl_red: "red", tl_yellow: "yellow", tl_green: "green" };

  const store = {
    get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* 저장 불가 환경 */ } },
  };

  // ---------- i18n ----------
  let lang = store.get("dl-lang", (navigator.language || "ko").startsWith("ko") ? "ko" : "en");
  const t = (k) => (window.I18N[lang] || {})[k] ?? k;

  function applyI18n() {
    document.documentElement.lang = lang;
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    $("#lang").textContent = lang === "ko" ? "EN" : "한국어";
    renderStatic();
    renderTabs();
    renderCondStats();
    renderLegends();
    $("#play").textContent = front.paused ? t("ctl.play") : t("ctl.pause");
    $("#map-mode").textContent = mapFollow ? t("map.full") : t("map.follow");
    drawAll();
  }
  $("#lang").addEventListener("click", () => { lang = lang === "ko" ? "en" : "ko"; store.set("dl-lang", lang); applyI18n(); });

  // ---------- 정적 섹션 ----------
  function renderStatic() {
    const best = index.length ? index : [];
    const redRun = best.reduce((a, c) => a + c.red_run, 0);
    const color = best.length ? Math.min(...best.map((c) => c.color)) : 1;
    const infer = best.length ? (best.reduce((a, c) => a + c.infer_ms, 0) / best.length).toFixed(0) : "12";
    $("#hero-stats").innerHTML = [
      [String(redRun), t("stat.redrun")],
      ["+5.1%p", t("stat.recall")],
      [`${(color * 100).toFixed(1)}%`, t("stat.color")],
      [`${infer} ms`, t("stat.infer")],
    ].map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");

    $("#flow").innerHTML = t("pipe.steps").map(([h, d]) => `<li><b>${h}</b><span>${d}</span></li>`).join("");

    const cols = t("data.cols");
    $("#data-table").innerHTML = `<thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
      t("data.rows").map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td class="up">${r[3]}</td></tr>`).join("") +
      "</tbody>";

    const k = t("res.k");
    $("#cases").innerHTML = t("res.cases").map(([h, a, b, c]) =>
      `<article class="card"><h3>${h}</h3><dl><dt>${k[0]}</dt><dd>${a}</dd><dt>${k[1]}</dt><dd>${b}</dd><dt>${k[2]}</dt><dd>${c}</dd></dl></article>`).join("");

    $("#limits-list").innerHTML = t("lim.items").map((s) => `<li>${s}</li>`).join("");
  }

  // ---------- 데이터 ----------
  let index = [], mapData = null, run = null, cond = store.get("dl-cond", "clear_noon");
  const cache = {};
  const front = $("#front"), chase = $("#chase");

  async function loadJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(url); return r.json(); }

  async function selectCond(id) {
    cond = id;
    store.set("dl-cond", id);
    renderTabs();
    renderCondStats();
    run = cache[id] || (cache[id] = await loadJSON(`assets/drive/${id}/frames.json`));
    if (!mapData || mapData.map !== run.meta.map) mapData = await loadJSON(`assets/maps/${run.meta.map}.json`);
    const wasPlaying = !front.paused;
    front.poster = `assets/drive/${id}/poster.jpg`;
    front.src = `assets/drive/${id}/front.mp4`;
    chase.src = `assets/drive/${id}/chase.mp4`;
    front.currentTime = 0;
    buildTimeline();
    if (wasPlaying) play();
    drawAll();
  }

  function renderTabs() {
    $("#cond-tabs").innerHTML = CONDS.filter((c) => index.some((i) => i.id === c)).map((c) =>
      `<button class="tab" role="tab" aria-selected="${c === cond}" data-c="${c}">${t("cond." + c)}</button>`).join("");
    document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => selectCond(b.dataset.c)));
  }

  function renderCondStats() {
    const s = index.find((i) => i.id === cond);
    if (!s) return;
    const pct = (v) => `${(v * 100).toFixed(1)}%`;
    $("#cond-stats").innerHTML = [
      [pct(s.agree), t("cs.agree")], [pct(s.found), t("cs.found")], [pct(s.color), t("cs.color")],
      [s.red_run, t("cs.redrun")], [s.stops, t("cs.stops")], [s.infer_ms, t("cs.infer")],
    ].map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
  }

  // ---------- 재생 ----------
  const fps = () => (run ? run.meta.fps : 20);
  const frameAt = (sec) => (run ? run.frames[Math.max(0, Math.min(run.frames.length - 1, Math.round(sec * fps())))] : null);

  function play() { front.play(); chase.play(); }
  $("#play").addEventListener("click", () => (front.paused ? play() : (front.pause(), chase.pause())));
  front.addEventListener("play", () => { $("#play").textContent = t("ctl.pause"); });
  front.addEventListener("pause", () => { $("#play").textContent = t("ctl.play"); drawAll(); });
  front.addEventListener("seeked", () => { chase.currentTime = front.currentTime; drawAll(); });

  function tick() {
    if (!front.paused) {
      if (Math.abs(chase.currentTime - front.currentTime) > 0.15) chase.currentTime = front.currentTime;
      drawAll();
    }
    requestAnimationFrame(tick);
  }

  ["#t-boxes", "#t-expected", "#t-hud"].forEach((s) => $(s).addEventListener("change", drawAll));
  $("#t-bright").addEventListener("change", (e) => $("#video-wrap").classList.toggle("bright", e.target.checked));

  // ---------- 오버레이 ----------
  const ov = $("#overlay");
  function fitCanvas(c) {
    const r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(r.width * dpr) || c.height !== Math.round(r.height * dpr)) {
      c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
    }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [g, r.width, r.height];
  }

  function drawOverlay(f) {
    const [g, W, H] = fitCanvas(ov);
    g.clearRect(0, 0, W, H);
    $("#hud").classList.toggle("hidden", !$("#t-hud").checked || !f);
    if (!f || !run) return;
    // 영상은 contain으로 그려짐 → 실제 영상 영역에 맞춘다
    const vw = run.meta.width, vh = run.meta.height, s = Math.min(W / vw, H / vh);
    const ox = (W - vw * s) / 2, oy = (H - vh * s) / 2;
    const box = (b) => [ox + b[0] * s, oy + b[1] * s, (b[2] - b[0]) * s, (b[3] - b[1]) * s];
    if ($("#t-expected").checked && f.exp) {
      g.setLineDash([4, 3]); g.strokeStyle = "#ffffff"; g.lineWidth = 1.5;
      f.exp.forEach((b) => { const [x, y, w, h] = box(b); g.strokeRect(x - 3, y - 3, w + 6, h + 6); });
      g.setLineDash([]);
    }
    if ($("#t-boxes").checked && f.det) {
      f.det.forEach((d, i) => {
        const [x, y, w, h] = box(d.slice(1, 5));
        g.strokeStyle = css(CLS_VAR[d[0]] || "--tl-unknown");
        g.lineWidth = i === f.pick ? 3 : 1.5;
        g.strokeRect(x, y, w, h);
        if (i === f.pick) {
          const label = `MY LIGHT ${SHORT[d[0]]} ${d[5].toFixed(2)}`;
          g.font = "600 12px system-ui, sans-serif";
          const tw = g.measureText(label).width;
          g.fillStyle = "rgba(0,0,0,.65)"; g.fillRect(x, y - 18, tw + 8, 16);
          g.fillStyle = "#fff"; g.fillText(label, x + 4, y - 6);
        }
      });
    }
    const st = (v) => v || "-";
    $("#hud").innerHTML =
      `<span class="k">${t("hud.perc")} </span>${run.meta.perception === "model" ? "MY MODEL (YOLO11n)" : "ground truth"}\n` +
      `<span class="k">${t("hud.light")} </span><b style="color:${css(STATE_VAR[f.tl] || "--tl-unknown")}">${st(f.tl)}</b>` +
      `${f.d != null ? `  ${f.d.toFixed(1)} m` : ""}   <span class="k">gt</span> ${st(f.gt)}\n` +
      `<span class="k">${t("hud.state")} </span>${f.st}\n` +
      `<span class="k">${t("hud.speed")} </span>${f.v.toFixed(1)} / ${f.vt.toFixed(1)} km/h\n` +
      `<span class="k">${t("hud.infer")} </span>${f.ms != null ? f.ms.toFixed(1) + " ms" : "-"}  (${f.det ? f.det.length : 0} det)`;
  }

  // ---------- 2D 지도 ----------
  const mc = $("#map");
  let mapFollow = true;
  $("#map-mode").addEventListener("click", () => {
    mapFollow = !mapFollow;
    $("#map-mode").textContent = mapFollow ? t("map.full") : t("map.follow");
    drawAll();
  });

  function drawMap(f) {
    const [g, W, H] = fitCanvas(mc);
    g.clearRect(0, 0, W, H);
    if (!mapData || !run || !f) return;
    g.save();
    if (mapFollow) {
      // 내 차 중심, 진행 방향이 위 (CARLA 좌표는 화면 좌표와 같은 손잡이 → 회전만)
      const scale = Math.min(W, H) / 70;
      g.translate(W / 2, H * 0.62);
      g.rotate((-90 - f.yaw) * Math.PI / 180);
      g.scale(scale, scale);
      g.translate(-f.x, -f.y);
    } else {
      const [x0, y0, x1, y1] = mapData.bbox, s = Math.min(W / (x1 - x0), H / (y1 - y0));
      g.translate((W - (x1 - x0) * s) / 2, (H - (y1 - y0) * s) / 2);
      g.scale(s, s);
      g.translate(-x0, -y0);
    }
    const unit = (window.devicePixelRatio || 1) / Math.hypot(g.getTransform().a, g.getTransform().b);   // 화면 1px의 월드 길이
    g.lineCap = "round"; g.lineJoin = "round";
    // 차로: 차선 폭만큼 두껍게
    for (const lane of mapData.lanes) {
      g.strokeStyle = css(lane.j ? "--road-j" : "--road");
      g.lineWidth = lane.w * 0.92;
      g.beginPath();
      lane.p.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      g.stroke();
    }
    // 경로: 전체(옅게) + 지나온 부분(진하게)
    const fr = run.frames, cur = fr.indexOf(f);
    g.lineWidth = 2.2 * unit; g.strokeStyle = css("--trail"); g.globalAlpha = 0.35;
    g.beginPath(); fr.forEach((p, i) => (i ? g.lineTo(p.x, p.y) : g.moveTo(p.x, p.y))); g.stroke();
    g.globalAlpha = 1; g.strokeStyle = css("--accent"); g.lineWidth = 3 * unit;
    g.beginPath(); for (let i = 0; i <= cur; i += 2) { const p = fr[i]; i ? g.lineTo(p.x, p.y) : g.moveTo(p.x, p.y); } g.stroke();
    // 정지선: 접근 중이면 실제 신호 색, 아니면 회색
    for (const s of run.stops) {
      const active = f.t >= s.t0 && f.t <= s.t1;
      const a = (s.yaw + 90) * Math.PI / 180, hw = 3.2;
      g.strokeStyle = active ? css(STATE_VAR[f.gt] || "--tl-unknown") : css("--tl-unknown");
      g.globalAlpha = active ? 1 : 0.5;
      g.lineWidth = (active ? 1.2 : 0.7);
      g.beginPath(); g.moveTo(s.x - Math.cos(a) * hw, s.y - Math.sin(a) * hw); g.lineTo(s.x + Math.cos(a) * hw, s.y + Math.sin(a) * hw); g.stroke();
    }
    g.globalAlpha = 1;
    // 내 차
    g.save();
    g.translate(f.x, f.y); g.rotate(f.yaw * Math.PI / 180);
    const L = mapFollow ? 2.4 : 6 * unit, Wd = mapFollow ? 1.0 : 3 * unit;
    g.fillStyle = css("--ego"); g.strokeStyle = css("--bg"); g.lineWidth = 0.4 * (mapFollow ? 1 : unit * 2);
    g.beginPath(); g.moveTo(L, 0); g.lineTo(-L, Wd); g.lineTo(-L * 0.6, 0); g.lineTo(-L, -Wd); g.closePath(); g.fill(); g.stroke();
    g.restore();
    g.restore();
  }

  // ---------- 타임라인 ----------
  const tc = $("#timeline"), tip = $("#tip");
  let tlMax = 40;
  function buildTimeline() { tlMax = Math.max(35, ...run.frames.map((f) => f.v)) * 1.1; }

  function drawTimeline(f) {
    const [g, W, H] = fitCanvas(tc);
    g.clearRect(0, 0, W, H);
    if (!run) return;
    const fr = run.frames, T = fr[fr.length - 1].t, padL = 78, padR = 8, x = (sec) => padL + (sec / T) * (W - padL - padR);
    const bandH = 16, rows = [[t("tl.model"), "tl", 4], [t("tl.truth"), "gt", 4 + bandH + 6]];
    g.font = "12px system-ui, sans-serif"; g.textBaseline = "middle";
    for (const [label, key, y] of rows) {
      g.fillStyle = css("--ink-2"); g.fillText(label, 0, y + bandH / 2);
      let start = 0;
      for (let i = 1; i <= fr.length; i++) {
        if (i === fr.length || fr[i][key] !== fr[start][key]) {
          const v = fr[start][key];
          if (v) {
            g.fillStyle = css(STATE_VAR[v] || "--tl-unknown");
            g.fillRect(x(fr[start].t), y, Math.max(1, x(fr[i - 1].t) - x(fr[start].t) + 1), bandH);
          }
          start = i;
        }
      }
    }
    // 속도 (한 축: km/h)
    const top = 4 + 2 * bandH + 16, bot = H - 16, y = (v) => bot - (v / tlMax) * (bot - top);
    g.strokeStyle = css("--line"); g.lineWidth = 1;
    [0, 10, 20, 30].forEach((v) => { g.beginPath(); g.moveTo(padL, y(v)); g.lineTo(W - padR, y(v)); g.stroke(); g.fillStyle = css("--ink-3"); g.fillText(`${v}`, padL - 22, y(v)); });
    g.fillStyle = css("--ink-2"); g.fillText(t("tl.speed"), 0, (top + bot) / 2);
    const line = (key, color, w, dash) => {
      g.strokeStyle = color; g.lineWidth = w; g.setLineDash(dash);
      g.beginPath(); fr.forEach((p, i) => (i ? g.lineTo(x(p.t), y(p[key])) : g.moveTo(x(p.t), y(p[key])))); g.stroke(); g.setLineDash([]);
    };
    line("vt", css("--ink-3"), 1, [3, 3]);
    line("v", css("--accent"), 2, []);
    // 시간 눈금
    g.fillStyle = css("--ink-3");
    for (let s = 0; s <= T; s += 30) g.fillText(`${s}s`, x(s) - 8, H - 6);
    // 재생 위치
    if (f) {
      g.strokeStyle = css("--ink"); g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(x(f.t), 0); g.lineTo(x(f.t), H - 14); g.stroke();
    }
    tc._x = { padL, padR, T, W };
  }

  function timeFromEvent(e) {
    const r = tc.getBoundingClientRect(), m = tc._x;
    if (!m) return null;
    const sec = ((e.clientX - r.left - m.padL) / (r.width - m.padL - m.padR)) * m.T;
    return sec < 0 || sec > m.T ? null : sec;
  }
  tc.addEventListener("click", (e) => { const s = timeFromEvent(e); if (s != null) { front.currentTime = s; } });
  tc.addEventListener("mousemove", (e) => {
    const s = timeFromEvent(e), f = s != null && frameAt(s);
    if (!f) { tip.hidden = true; return; }
    const r = tc.getBoundingClientRect(), pr = tc.parentElement.getBoundingClientRect();
    tip.hidden = false;
    tip.style.left = `${e.clientX - pr.left}px`; tip.style.top = `${r.top - pr.top + 4}px`;
    tip.innerHTML = `${f.t.toFixed(1)} s · ${t("tl.model")} <b>${f.tl || t("tl.none")}</b> · ${t("tl.truth")} <b>${f.gt || t("tl.none")}</b> · ${f.v.toFixed(0)} km/h`;
  });
  tc.addEventListener("mouseleave", () => { tip.hidden = true; });

  function renderLegends() {
    const sw = (v, label) => `<li><i style="background:${css(v)}"></i>${label}</li>`;
    $("#tl-legend").innerHTML = sw("--tl-red", "RED") + sw("--tl-yellow", "YELLOW") + sw("--tl-green", "GREEN") +
      sw("--tl-unknown", t("tl.unknown")) + sw("--accent", t("tl.speed")) + `<li>┄ ${t("tl.target")}</li>`;
    $("#map-legend").innerHTML = sw("--road", t("lg.road")) + sw("--accent", t("lg.trail")) + sw("--ego", t("lg.ego")) + sw("--tl-red", t("lg.stop"));
  }

  // ---------- 그리기 ----------
  function drawAll() {
    const f = frameAt(front.currentTime || 0);
    $("#time").textContent = `${(front.currentTime || 0).toFixed(1)} s`;
    drawOverlay(f);
    drawMap(f);
    drawTimeline(f);
  }
  window.addEventListener("resize", drawAll);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { renderLegends(); drawAll(); });

  // ---------- 시작 ----------
  (async () => {
    try { index = await loadJSON("assets/drive/index.json"); } catch { index = []; }
    if (!index.some((i) => i.id === cond)) cond = index[0] ? index[0].id : cond;
    applyI18n();
    if (index.length) await selectCond(cond);
    requestAnimationFrame(tick);
  })();
})();
