// 직접 해 보기: 학습한 검출 모델(ONNX)을 브라우저에서 실행. 전처리·후처리는 scripts/42_build_try.py 의 letterbox/decode와 같은 식
(() => {
  const $ = (s) => document.querySelector(s);
  const ORT_VER = "1.20.1";
  const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
  const CLS_VAR = { vehicle: "--vehicle", tl_red: "--tl-red", tl_yellow: "--tl-yellow", tl_green: "--tl-green", pedestrian: "--ped" };
  const SHORT = { vehicle: "car", tl_red: "red", tl_yellow: "yellow", tl_green: "green", pedestrian: "person" };
  const PAD = 114 / 255, IOU = 0.7, MIN_CONF = 0.05;
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  const store = {
    get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* 저장 불가 환경 */ } },
  };
  let lang = store.get("dl-lang", (navigator.language || "ko").startsWith("ko") ? "ko" : "en");
  const t = (k) => (window.I18N[lang] || {})[k] ?? k;
  const fmt = (s, o) => s.replace(/\{(\w+)\}/g, (_, k) => o[k] ?? "");

  let meta = null, session = null, backend = null;
  let current = null;        // { bitmap, sample|null, cands, ms }
  const hidden = new Set();  // 숨긴 클래스
  let status = { k: "try.status.engine" }, busy = false, pending = null;

  // ---------- 모델 ----------
  function loadScript(src) {
    return new Promise((ok, fail) => {
      const s = document.createElement("script");
      s.src = src; s.onload = ok; s.onerror = () => fail(new Error(src));
      document.head.appendChild(s);
    });
  }

  async function fetchWithProgress(url, total) {
    const r = await fetch(url);
    if (!r.ok || !r.body) throw new Error(`${url} ${r.status}`);
    const size = +r.headers.get("content-length") || total;
    const reader = r.body.getReader(), parts = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); got += value.length;
      setStatus("try.status.loading", { p: size ? `${Math.min(100, Math.round((got / size) * 100))}%` : `${(got / 1e6).toFixed(1)} MB` });
    }
    const buf = new Uint8Array(got);
    let o = 0;
    for (const p of parts) { buf.set(p, o); o += p.length; }
    return buf;
  }

  async function initModel() {
    const gpu = !!navigator.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null));
    // WebGPU 빌드는 WASM도 포함 (GPU 세션 실패 시 같은 스크립트로 CPU 실행)
    await loadScript(ORT_CDN + (gpu ? "ort.webgpu.min.js" : "ort.wasm.min.js"));
    const ort = window.ort;
    ort.env.wasm.wasmPaths = ORT_CDN;
    if (!self.crossOriginIsolated) ort.env.wasm.numThreads = 1;   // GitHub Pages는 멀티스레드(SharedArrayBuffer) 불가
    const bytes = await fetchWithProgress(meta.model, meta.model_mb * 1e6);
    setStatus("try.status.engine");
    for (const ep of gpu ? ["webgpu", "wasm"] : ["wasm"]) {
      try {
        session = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: "all" });
        backend = ep;
        break;
      } catch (e) {
        console.warn(`[try] ${ep} 실패`, e);
      }
    }
    if (!session) throw new Error("no backend");
    setStatus("try.status.ready", { b: t(`try.backend.${backend}`) });
  }

  // ---------- 추론 ----------
  function preprocess(bitmap) {
    const [W, H] = meta.input;
    const r = Math.min(W / bitmap.width, H / bitmap.height);
    const nw = Math.round(bitmap.width * r), nh = Math.round(bitmap.height * r);
    const px = Math.floor((W - nw) / 2), py = Math.floor((H - nh) / 2);
    const c = new OffscreenCanvasOr(W, H), g = c.getContext("2d", { willReadFrequently: true });
    g.fillStyle = "rgb(114,114,114)"; g.fillRect(0, 0, W, H);
    g.imageSmoothingQuality = "medium";
    g.drawImage(bitmap, px, py, nw, nh);
    const px8 = g.getImageData(0, 0, W, H).data, n = W * H, x = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      x[i] = px8[i * 4] / 255; x[n + i] = px8[i * 4 + 1] / 255; x[2 * n + i] = px8[i * 4 + 2] / 255;
    }
    return { x, r, px, py };
  }
  function OffscreenCanvasOr(w, h) {
    if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
    const c = document.createElement("canvas"); c.width = w; c.height = h; return c;
  }

  // (1, 4+C, N) → 후보 [cls, score, x1, y1, x2, y2] (원본 이미지 좌표, score ≥ MIN_CONF)
  function candidates(out, { r, px, py }) {
    const [, rows, N] = out.dims, d = out.data, C = rows - 4, cands = [];
    for (let j = 0; j < N; j++) {
      let best = 0, bc = 0;
      for (let c = 0; c < C; c++) { const s = d[(4 + c) * N + j]; if (s > best) { best = s; bc = c; } }
      if (best < MIN_CONF) continue;
      const cx = d[j], cy = d[N + j], w = d[2 * N + j], h = d[3 * N + j];
      cands.push([bc, best, (cx - w / 2 - px) / r, (cy - h / 2 - py) / r, (cx + w / 2 - px) / r, (cy + h / 2 - py) / r]);
    }
    return cands;
  }

  const iou = (a, b) => {
    const iw = Math.max(0, Math.min(a[4], b[4]) - Math.max(a[2], b[2]));
    const ih = Math.max(0, Math.min(a[5], b[5]) - Math.max(a[3], b[3]));
    const inter = iw * ih;
    return inter / ((a[4] - a[2]) * (a[5] - a[3]) + (b[4] - b[2]) * (b[5] - b[3]) - inter);
  };
  // 신뢰도 기준 → 클래스별 NMS (Python decode와 같은 순서)
  function detections(cands, conf) {
    const keep = [], byCls = new Map();
    cands.forEach((c) => { if (c[1] >= conf) (byCls.get(c[0]) || byCls.set(c[0], []).get(c[0])).push(c); });
    for (const list of byCls.values()) {
      list.sort((a, b) => b[1] - a[1]);
      const alive = list.slice();
      while (alive.length) {
        const top = alive.shift();
        keep.push(top);
        for (let i = alive.length - 1; i >= 0; i--) if (iou(top, alive[i]) >= IOU) alive.splice(i, 1);
      }
    }
    return keep;
  }

  async function run(item) {
    if (busy) { pending = item; return; }
    busy = true;
    current = { ...item, cands: null, ms: null };
    draw();
    if (!session) { busy = false; return; }   // 모델 준비되면 initModel 뒤에 다시 부름
    setStatus("try.status.running", { b: t(`try.backend.${backend}`) });
    await new Promise((r) => requestAnimationFrame(() => setTimeout(r)));   // 상태 문구가 먼저 그려지게
    try {
      const pre = preprocess(item.bitmap);
      const [W, H] = meta.input;
      const input = new window.ort.Tensor("float32", pre.x, [1, 3, H, W]);
      const t0 = performance.now();
      const res = await session.run({ [session.inputNames[0]]: input });
      const ms = performance.now() - t0;
      if (current && current.bitmap === item.bitmap) {
        current.cands = candidates(res[session.outputNames[0]], pre);
        current.ms = ms;
      }
      setStatus("try.status.done", { b: t(`try.backend.${backend}`), ms: ms.toFixed(0) });
      if (current.cands) {
        const d = detections(current.cands, +$("#conf").value);
        console.info(`[try] ${item.sample ? item.sample.id : "upload"} ${backend} ${ms.toFixed(0)}ms`,
          JSON.stringify(d.map(([k, sc, ...b]) => [k, +sc.toFixed(3), ...b.map((v) => Math.round(v))])));
      }
    } catch (e) {
      console.error(e);
      setStatus("try.status.fail", { e: e.message || e });
    }
    busy = false;
    draw();
    if (pending) { const p = pending; pending = null; run(p); }
  }

  // ---------- 그리기 ----------
  const view = $("#view");
  function draw() {
    const box = view.parentElement.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const bm = current && current.bitmap;
    const aspect = bm ? bm.height / bm.width : 9 / 16;
    const W = box.width, H = Math.round(W * aspect);
    view.style.height = `${H}px`;
    view.width = Math.round(W * dpr); view.height = Math.round(H * dpr);
    const g = view.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#000"; g.fillRect(0, 0, W, H);
    if (!bm) return renderCounts([]);
    const s = W / bm.width;
    g.drawImage(bm, 0, 0, W, H);

    const names = meta.names;
    const sample = current.sample;
    if (sample && $("#t-gt").checked) {
      g.setLineDash([5, 4]); g.lineWidth = 1.5; g.strokeStyle = "#ffffff";
      sample.gt.forEach(([k, x1, y1, x2, y2]) => {
        if (!hidden.has(names[k])) g.strokeRect(x1 * s - 2, y1 * s - 2, (x2 - x1) * s + 4, (y2 - y1) * s + 4);
      });
      g.setLineDash([]);
    }
    const dets = current.cands ? detections(current.cands, +$("#conf").value) : [];
    const labels = $("#t-labels").checked;
    const shown = dets.filter((d) => !hidden.has(names[d[0]]));
    shown.forEach(([k, , x1, y1, x2, y2]) => {
      g.strokeStyle = css(CLS_VAR[names[k]] || "--tl-unknown"); g.lineWidth = 2;
      g.strokeRect(x1 * s, y1 * s, (x2 - x1) * s, (y2 - y1) * s);
    });
    if (labels) {
      // 신뢰도 높은 것부터 이름표, 이미 붙인 이름표와 겹치면 생략 (작은 신호등이 몰린 곳)
      g.font = "600 11px system-ui, sans-serif";
      const placed = [];
      [...shown].sort((a, b) => b[1] - a[1]).forEach(([k, sc, x1, y1, , y2]) => {
        const name = names[k], txt = `${SHORT[name] || name} ${sc.toFixed(2)}`;
        const w = g.measureText(txt).width + 6, x = Math.min(x1 * s, W - w), y = y1 * s > 15 ? y1 * s - 15 : y2 * s;
        if (placed.some((p) => x < p[0] + p[2] && p[0] < x + w && y < p[1] + 15 && p[1] < y + 15)) return;
        placed.push([x, y, w]);
        g.fillStyle = css(CLS_VAR[name] || "--tl-unknown"); g.fillRect(x, y, w, 15);
        g.fillStyle = "#fff"; g.fillText(txt, x + 3, y + 11);
      });
    }
    renderCounts(dets);
  }

  function renderCounts(dets) {
    if (!meta) return;
    const cls = t("try.cls"), sample = current && current.sample;
    const n = (k) => dets.filter((d) => meta.names[d[0]] === k).length;
    const g = (k) => (sample ? sample.gt.filter((b) => meta.names[b[0]] === k).length : null);
    $("#counts").innerHTML = meta.names.map((k) => {
      const gt = g(k);
      return `<button type="button" class="chip try-chip${hidden.has(k) ? " off" : ""}" data-k="${k}" aria-pressed="${!hidden.has(k)}">` +
        `<i style="background:${css(CLS_VAR[k])}"></i>${cls[k] || k} <b>${fmt(t("try.count"), { n: n(k) })}</b>` +
        `${gt != null ? ` <span class="note">${fmt(t("try.gtCount"), { n: gt })}</span>` : ""}</button>`;
    }).join("");
    $("#counts").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      hidden.has(b.dataset.k) ? hidden.delete(b.dataset.k) : hidden.add(b.dataset.k);
      draw();
    }));
    const kinds = t("try.kind");
    $("#caption").textContent = sample ? `${t(`cond.${sample.cond}`)} · ${kinds[sample.kind] || ""} · ${sample.frame}` : current ? t("try.ownPhoto") : "";
    $("#gt-wrap").classList.toggle("disabled", !sample);
    $("#t-gt").disabled = !sample;
  }

  function setStatus(k, o = {}) { status = { k, o }; $("#status").textContent = fmt(t(k), o); }

  // ---------- 입력 ----------
  const bitmaps = new Map();
  async function pickSample(s) {
    document.querySelectorAll(".thumb").forEach((b) => b.setAttribute("aria-pressed", b.dataset.id === s.id));
    if (s.kind === "miss") $("#t-gt").checked = true;
    if (!bitmaps.has(s.id)) {
      const blob = await (await fetch(s.src)).blob();
      bitmaps.set(s.id, await createImageBitmap(blob));
    }
    run({ bitmap: bitmaps.get(s.id), sample: s });
  }

  async function pickFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    document.querySelectorAll(".thumb").forEach((b) => b.setAttribute("aria-pressed", "false"));
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    run({ bitmap, sample: null });
  }

  $("#file").addEventListener("change", (e) => pickFile(e.target.files[0]));
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => { e.preventDefault(); pickFile(e.dataTransfer.files[0]); });

  $("#conf").addEventListener("input", (e) => { $("#conf-v").textContent = (+e.target.value).toFixed(2); draw(); });
  ["#t-gt", "#t-labels"].forEach((s) => $(s).addEventListener("change", draw));
  window.addEventListener("resize", draw);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);

  // ---------- 문구 ----------
  function applyI18n() {
    document.documentElement.lang = lang;
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    $("#lang").textContent = lang === "ko" ? "EN" : "한국어";
    if (meta) {
      $("#try-lead").textContent = fmt(t("try.lead"), { w: meta.weights_mb });
      $("#how").innerHTML = t("try.how").map(([h, p]) =>
        `<div class="card"><h3>${h}</h3><p class="note">${fmt(p, { m: meta.model_mb })}</p></div>`).join("");
      $("#thumbs").innerHTML = meta.samples.map((s) =>
        `<button type="button" class="thumb" data-id="${s.id}" aria-pressed="${current && current.sample === s}" title="${t("try.kind")[s.kind] || ""}">` +
        `<img src="${s.thumb}" alt="${t(`cond.${s.cond}`)}" loading="lazy" width="320" height="180"><span>${t(`cond.${s.cond}`)}</span></button>`).join("");
      $("#thumbs").querySelectorAll(".thumb").forEach((b) =>
        b.addEventListener("click", () => pickSample(meta.samples.find((s) => s.id === b.dataset.id))));
    }
    setStatus(status.k, status.o);
    draw();
  }
  $("#lang").addEventListener("click", () => { lang = lang === "ko" ? "en" : "ko"; store.set("dl-lang", lang); applyI18n(); });

  (async () => {
    applyI18n();
    try {
      meta = await (await fetch("assets/try/try.json")).json();
    } catch (e) {
      return setStatus("try.status.fail", { e: "try.json" });
    }
    applyI18n();
    pickSample(meta.samples[0]);
    try {
      await initModel();
      if (current) run({ bitmap: current.bitmap, sample: current.sample });
    } catch (e) {
      console.error(e);
      setStatus("try.status.fail", { e: e.message || e });
    }
  })();
})();
