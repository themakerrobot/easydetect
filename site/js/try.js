// Apache-2.0
/* The "try it" page: pick a hub model (or drop your own .onnx), then a picture,
   a video or the webcam. Everything runs in this page (js/engine.js). */
import { runtime, preferred, fetchModel, headKind, Model, render, colour, ENCODER, SAM_DECODER, POSE } from "./engine.js";

const REPO = "leeyunjai/easydetect-hub";
const RAW = `https://huggingface.co/${REPO}/resolve/main`;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const TASKS = ["detect", "segment", "pose"];
const HEAD = { segment: "mask_decoder.onnx", pose: "pose.onnx" };

const T = {
  en: {
    navModels: "Models", navTry: "Try in the browser", navTryS: "Try it", model: "Model", fromHub: "Hub models", fromMine: "My model",
    mineDrop: "Drop .onnx files here, or click to choose",
    mineHint: "A model exported from the lab or easydetect. For keypoints or masks, also choose the .pose.onnx / .mask_decoder.onnx beside it.",
    input: "Input", open: "Open a picture or video", camOn: "Turn on the webcam", camOff: "Stop the webcam", camSwap: "Switch camera",
    conf: "Score threshold", runOn: "Run on", epAuto: "Auto (WebGPU when there is a GPU)", epWasm: "CPU only (WASM)",
    found: "Found", foundHint: "Click a name to hide it or show it again.",
    privacy: "Pictures and videos are processed in this browser only: nothing is uploaded. Models download once and stay in the browser.",
    save: "Save the result", hello: "Drop a picture or video here", hello2: "or turn on the webcam",
    detect: "Detection", segment: "Segmentation", pose: "Keypoints", px: "px", int8: "8-bit",
    loading: "Downloading", loadingRt: "Starting the runtime", ready: "Ready", noModel: "Pick a model first.",
    err: "Could not load: ", errList: "Could not read the model list from Hugging Face.",
    mineNeed: "Choose the detector's .onnx (the one without .pose / .mask_decoder in its name).",
    mineHeadless: "No {0} beside it: easydetect's general one is used ({1}).",
    noteSeg: "Masks also need MobileSAM's encoder (28 MB, once).",
    notePose: "Keypoints go on the boxes named \"person\" (every box when there is none).",
    noteLic: "This model follows its dataset's terms. Check them before commercial use.",
    camErr: "Could not open the webcam: ", count: "{n} found", ms: "{0} ms", fps: "{0} fps",
    samDefault: "MobileSAM", poseDefault: "COCO people", epNow: "running on {0}",
  },
  ko: {
    navModels: "모델", navTry: "브라우저에서 해 보기", navTryS: "해 보기", model: "모델", fromHub: "허브 모델", fromMine: "내 모델",
    mineDrop: ".onnx 파일을 끌어다 놓거나 눌러서 고르세요",
    mineHint: "lab이나 easydetect에서 내보낸 모델이에요. 포즈·세그 모델이면 옆에 있는 .pose.onnx / .mask_decoder.onnx도 같이 고르세요.",
    input: "입력", open: "사진 · 영상 열기", camOn: "웹캠 켜기", camOff: "웹캠 끄기", camSwap: "카메라 바꾸기",
    conf: "점수 기준", runOn: "실행", epAuto: "자동 (GPU가 있으면 WebGPU)", epWasm: "CPU만 (WASM)",
    found: "찾은 것", foundHint: "이름을 누르면 숨기거나 다시 보여요.",
    privacy: "사진과 영상은 이 브라우저 안에서만 처리돼요. 어디에도 올라가지 않아요. 모델은 처음 한 번만 받고 브라우저에 저장해 둬요.",
    save: "결과 저장", hello: "사진이나 영상을 여기에 끌어다 놓으세요", hello2: "또는 웹캠을 켜 보세요",
    detect: "탐지", segment: "세그멘테이션", pose: "키포인트", px: "px", int8: "8비트",
    loading: "받는 중", loadingRt: "실행기 준비 중", ready: "준비됐어요", noModel: "먼저 모델을 고르세요.",
    err: "불러오지 못했어요: ", errList: "Hugging Face에서 모델 목록을 읽지 못했어요.",
    mineNeed: "탐지 모델 .onnx(이름에 .pose / .mask_decoder가 없는 것)를 고르세요.",
    mineHeadless: "옆에 {0} 파일이 없어서 easydetect 기본 모델을 써요({1}).",
    noteSeg: "마스크에는 MobileSAM 인코더(28 MB)도 한 번 받아요.",
    notePose: "키포인트는 \"person\" 상자에 붙어요(그런 이름이 없으면 모든 상자).",
    noteLic: "이 모델은 데이터셋 조건을 따라요. 상업용은 먼저 확인하세요.",
    camErr: "웹캠을 열지 못했어요: ", count: "{n}개", ms: "{0} ms", fps: "{0} fps",
    samDefault: "MobileSAM", poseDefault: "COCO 사람", epNow: "{0}에서 실행 중",
  },
};
let lang = (() => { try { return localStorage.getItem("hub.lang"); } catch { return null; } })()
  || ((navigator.language || "").toLowerCase().startsWith("ko") ? "ko" : "en");
const t = (k) => (T[lang] && T[lang][k]) ?? T.en[k] ?? k;


const S = {
  src: "hub", list: [], ds: "", task: "detect", size: 640, int8: false,
  mine: { det: null, head: null, task: "detect" },
  model: null, loadId: 0, conf: null, hidden: new Set(),
  frame: null, last: null, live: null, facing: "user", busy: false,
};
const ep = (() => { try { return localStorage.getItem("try.ep") || "auto"; } catch { return "auto"; } })();

/* ---------------------------------------------------------------- the model list */

async function loadList() {
  try {
    const r = await fetch(`${RAW}/index.json`, { cache: "no-cache" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const by = new Map();
    for (const [key, m] of Object.entries(await r.json())) {
      const [id, name] = key.split("/");
      if (!by.has(id)) by.set(id, { id, variants: [] });
      by.get(id).variants.push({ key, name, ...m, task: m.task || "detect" });
    }
    S.list = [...by.values()].map((d) => {
      const m = d.variants.find((v) => v.task === "detect") || d.variants[0];
      d.title = prettyName(d.id, m.title || d.id); d.category = m.category || "Other"; d.meta = m;
      d.tasks = TASKS.filter((k) => d.variants.some((v) => v.task === k));
      return d;
    }).sort((a, b) => a.category.localeCompare(b.category) || a.title.localeCompare(b.title));
  } catch (e) {
    $("modelNote").textContent = t("errList") + ` (${e.message})`;
    return;
  }
  const [want, variant] = decodeURIComponent(location.hash.slice(1)).split("/");
  const d = S.list.find((x) => x.id === want) || S.list.find((x) => x.id === "coco") || S.list[0];
  S.ds = d.id;
  const v = variant && d.variants.find((x) => x.name === variant.replace(/-int8$/, ""));
  if (v) { S.task = v.task; S.size = v.imgsz; S.int8 = /-int8$/.test(variant); }
  else { S.task = d.tasks[0]; S.size = 320; S.int8 = false; }
  drawHub();
  loadHub();
}
function prettyName(id, title) {
  if (id.startsWith("rf100-")) {
    let n = id.slice(6).replace(/-(?=[a-z0-9]{5}$)(?=.*\d)[a-z0-9]{5}$/, "").replace(/-/g, " ");
    return n.charAt(0).toUpperCase() + n.slice(1);
  }
  return title.split(" (")[0].replace(" Dataset", "");
}
const dataset = () => S.list.find((x) => x.id === S.ds);
const variant = () => { const d = dataset(); return d && (d.variants.find((v) => v.task === S.task && v.imgsz === S.size) || d.variants.find((v) => v.task === S.task)); };
function stemOf(d, v) {
  const named = `${d.id}-${v.name}`;
  return !(v.files || []).length || v.files.some((f) => f.startsWith(named)) ? named : v.name;
}
function headOf(d, v) {
  const head = HEAD[v.task]; if (!head) return null;
  const own = `${stemOf(d, v)}.${head}`;
  return !(v.files || []).length || v.files.includes(own) ? own : head;
}
const url = (v, f) => `${RAW}/${v.key.split("/").map(encodeURIComponent).join("/")}/${encodeURIComponent(f)}`;

function drawHub() {
  const groups = new Map();
  for (const d of S.list) { if (!groups.has(d.category)) groups.set(d.category, []); groups.get(d.category).push(d); }
  $("ds").innerHTML = [...groups].map(([c, ds]) => `<optgroup label="${esc(c)}">${ds.map((d) =>
    `<option value="${esc(d.id)}"${d.id === S.ds ? " selected" : ""}>${esc(d.title)}</option>`).join("")}</optgroup>`).join("");
  const d = dataset(); if (!d) return;
  $("tasks").innerHTML = d.tasks.length > 1 ? d.tasks.map((k) => `<button type="button" class="db${k === S.task ? " on" : ""}" data-task="${k}">${esc(t(k))}</button>`).join("") : "";
  const v = variant();
  const sizes = d.variants.filter((x) => x.task === S.task).sort((a, b) => a.imgsz - b.imgsz);
  $("sizes").innerHTML = sizes.map((x) => {
    const i8 = x.score_int8 != null || x.map50_95_int8 != null;
    return `<button type="button" class="db${x === v && !S.int8 ? " on" : ""}" data-size="${x.imgsz}">${x.imgsz} ${t("px")}</button>`
      + (i8 ? `<button type="button" class="db${x === v && S.int8 ? " on" : ""}" data-size="${x.imgsz}" data-int8="1">${x.imgsz} ${t("int8")}</button>` : "");
  }).join("");
}

/* ---------------------------------------------------------------- loading */

function progress(name, done, total) {
  $("prg").hidden = false; $("prgName").textContent = `${t("loading")} · ${name}`;
  const mb = (n) => (n / 1e6).toFixed(1);
  $("prgN").textContent = total ? `${mb(done)} / ${mb(total)} MB` : `${mb(done)} MB`;
  $("prgFill").style.width = total ? `${Math.round(done / total * 100)}%` : "100%";
}
async function get(u, name) { return fetchModel(u, (a, b) => progress(name, a, b)); }

async function build(id, make, note) {
  $("modelNote").textContent = note; $("save").disabled = true;
  try {
    $("prg").hidden = false; $("prgName").textContent = t("loadingRt"); $("prgN").textContent = ""; $("prgFill").style.width = "0";
    await runtime(ep);
    const model = await make();
    if (id !== S.loadId) { model.release(); return; }
    S.model?.release(); S.model = model;
    S.conf = model.conf; $("conf").value = model.conf; $("confV").textContent = model.conf.toFixed(2);
    S.hidden.clear();
    $("prg").hidden = true;
    $("title").textContent = `${t("ready")} · ${t(model.task)} · ${model.size} ${t("px")} · ${t("epNow").replace("{0}", model.ep === "webgpu" ? "WebGPU" : "WASM")}`;
    if (S.last) analyse(S.last.src, true);
  } catch (e) {
    if (id !== S.loadId) return;
    $("prg").hidden = true; $("modelNote").textContent = t("err") + (e.message || e);
    console.error(e);
  }
}

function loadHub() {
  const d = dataset(); const v = variant(); if (!d || !v) return;
  const id = ++S.loadId;
  history.replaceState(null, "", `#${d.id}/${v.name}${S.int8 ? "-int8" : ""}`);
  const stem = stemOf(d, v); const det = `${stem}${S.int8 ? "-int8" : ""}.onnx`; const head = headOf(d, v);
  const notes = [t("noteLic")];
  if (v.task === "segment") notes.unshift(t("noteSeg"));
  if (v.task === "pose") notes.unshift(t("notePose"));
  build(id, async () => {
    const bytes = await get(url(v, det), det);
    const h = head ? await get(url(v, head), head) : null;
    const enc = v.task === "segment" ? await get(ENCODER, "encoder.onnx") : null;
    return Model.create({ det: bytes, head: h, task: v.task, encoder: enc, names: Object.fromEntries(Object.entries(d.meta.classes || {}).map(([k, x]) => [Number(k), x])),
      prefer: preferred(), int8: S.int8 });
  }, notes.join(" "));
}


/* your own model: the detector, and a keypoint net or mask decoder if one came with it */
async function addMine(files) {
  for (const f of files) {
    if (!/\.onnx$/i.test(f.name)) continue;
    const bytes = new Uint8Array(await f.arrayBuffer());
    const kind = /\.pose\.onnx$/i.test(f.name) ? "pose" : /\.mask_decoder\.onnx$/i.test(f.name) ? "segment"
      : await headKind(await ensureRt(bytes));
    if (kind) { S.mine.head = { name: f.name, bytes, kind }; S.mine.task = kind; }
    else S.mine.det = { name: f.name, bytes };
  }
  drawMine();
  loadMine();
}
async function ensureRt(bytes) { await runtime(ep); return bytes; }
function drawMine() {
  const m = S.mine;
  $("mineList").innerHTML = [m.det, m.head].filter(Boolean).map((f) => `<div class="file"><span class="ic">ONNX</span><span><b>${esc(f.name)}</b><small>${(f.bytes.length / 1e6).toFixed(1)} MB${f.kind ? " · " + esc(t(f.kind)) : ""}</small></span></div>`).join("");
  $("mineTasks").innerHTML = m.det ? TASKS.map((k) => `<button type="button" class="db${k === m.task ? " on" : ""}" data-mtask="${k}">${esc(t(k))}</button>`).join("") : "";
}
function loadMine() {
  const m = S.mine;
  if (!m.det) {
    S.loadId++; S.model?.release(); S.model = null; $("prg").hidden = true;
    $("modelNote").textContent = m.head ? t("mineNeed") : ""; $("title").textContent = t("mineDrop");
    return;
  }
  const id = ++S.loadId;
  const own = m.head && m.head.kind === m.task ? m.head.bytes : null;
  const notes = [];
  if (m.task !== "detect" && !own) notes.push(t("mineHeadless").replace("{0}", m.task === "pose" ? ".pose.onnx" : ".mask_decoder.onnx").replace("{1}", t(m.task === "pose" ? "poseDefault" : "samDefault")));
  if (m.task === "segment") notes.push(t("noteSeg"));
  if (m.task === "pose") notes.push(t("notePose"));
  build(id, async () => {
    const head = m.task === "detect" ? null : own || await get(m.task === "pose" ? POSE : SAM_DECODER, m.task === "pose" ? "pose-s.onnx" : "decoder.onnx");
    const enc = m.task === "segment" ? await get(ENCODER, "encoder.onnx") : null;
    return Model.create({ det: m.det.bytes, head, task: m.task, encoder: enc, prefer: preferred() });
  }, notes.join(" "));
}

/* ---------------------------------------------------------------- running */

/* the picture the model reads: a canvas no bigger than 1600 px (1280 live) */
function grab(el, live) {
  const w0 = el.videoWidth || el.naturalWidth || el.width, h0 = el.videoHeight || el.naturalHeight || el.height;
  if (!w0 || !h0) return null;
  const k = Math.min(1, (live ? 1280 : 1600) / Math.max(w0, h0));
  const w = Math.round(w0 * k), h = Math.round(h0 * k);
  const c = S.frame && S.frame.width === w && S.frame.height === h && live ? S.frame : Object.assign(document.createElement("canvas"), { width: w, height: h });
  c.getContext("2d").drawImage(el, 0, 0, w, h);
  if (live) S.frame = c;
  return c;
}

async function analyse(src, still) {
  const model = S.model; if (!model) { toast(t("noModel")); return; }
  const w = src.width, h = src.height;
  const t0 = performance.now();
  const raw = await model.detect(src);
  const emb = model.task === "segment" ? await model.embed(src, w, h) : null;
  const job = { src, raw, emb, model };
  if (still) S.last = job;
  await finish(job);
  S.ms = performance.now() - t0;
  stat();
}

async function finish({ src, raw, emb, model }) {
  if (model !== S.model) return;
  const w = src.width, h = src.height;
  const all = model.boxes(raw, w, h, S.conf ?? model.conf);
  const dets = all.filter((d) => !S.hidden.has(d.cls));
  if (model.task === "pose" && model.pose) await model.keypoints(src, dets, { flip: !S.live });
  if (model.task === "segment" && emb) await model.masks(emb, dets);
  render($("out"), src, dets, model);
  $("out").hidden = false; $("hello").hidden = true; $("save").disabled = false;
  drawFound(all, model);
  S.count = dets.length; stat();
}
function stat() {
  const speed = S.live ? (S.fps ? t("fps").replace("{0}", S.fps.toFixed(1)) : "") : (S.ms ? t("ms").replace("{0}", Math.round(S.ms)) : "");
  $("stat").textContent = `· ${t("count").replace("{n}", S.count || 0)}${speed ? " · " + speed : ""}`;
}

function drawFound(all, model) {
  const n = new Map();
  for (const d of all) n.set(d.cls, (n.get(d.cls) || 0) + 1);
  for (const k of S.hidden) if (!n.has(k)) n.set(k, 0);
  $("found").innerHTML = [...n].sort((a, b) => b[1] - a[1]).map(([k, v]) => {
    const [r, g, b] = colour(k);
    return `<button type="button" class="db${S.hidden.has(k) ? " off" : ""}" data-cls="${k}"><i style="background:rgb(${r},${g},${b})"></i>${esc(model.names[k] ?? k)} <small>${v}</small></button>`;
  }).join("");
}

/* pictures, videos and the webcam */
async function openFile(f) {
  stopLive();
  if (f.type.startsWith("video/")) { const v = $("vid"); v.srcObject = null; v.src = URL.createObjectURL(f); await v.play(); startLive(v, false); return; }
  const img = new Image(); img.src = URL.createObjectURL(f);
  await img.decode();
  const c = grab(img, false); URL.revokeObjectURL(img.src);
  if (S.model) await analyse(c, true);
  else { S.last = { src: c }; drawPlain(c); }
}
function drawPlain(c) { const o = $("out"); o.width = c.width; o.height = c.height; o.getContext("2d").drawImage(c, 0, 0); o.hidden = false; $("hello").hidden = true; }

async function camera() {
  if (S.live && S.live.cam) { stopLive(); return; }
  stopLive();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: S.facing, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    const v = $("vid"); v.src = ""; v.srcObject = stream; await v.play();
    startLive(v, true);
    $("cam").textContent = t("camOff"); $("cam").classList.add("on");
    navigator.mediaDevices.enumerateDevices().then((ds) => { $("flipCam").hidden = ds.filter((d) => d.kind === "videoinput").length < 2; }).catch(() => {});
  } catch (e) { toast(t("camErr") + (e.message || e.name)); }
}
function startLive(video, cam) {
  S.last = null; S.fps = 0;
  const live = { video, cam, stop: false };
  S.live = live;
  let prev = performance.now();
  const tick = async () => {
    if (live.stop) return;
    if (S.model && video.readyState >= 2) {
      const c = grab(video, true);
      if (c) {
        try { await analyse(c, false); } catch (e) { console.error(e); }
        const now = performance.now(); S.fps = S.fps ? S.fps * 0.8 + 200 / (now - prev) : 1000 / (now - prev); prev = now;
      }
    } else if (video.readyState >= 2) { const c = grab(video, true); if (c) drawPlain(c); }
    if (!live.stop) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
function stopLive() {
  const live = S.live; if (!live) return;
  live.stop = true; S.live = null;
  const v = live.video;
  if (S.frame) { const c = document.createElement("canvas"); c.width = S.frame.width; c.height = S.frame.height; c.getContext("2d").drawImage(S.frame, 0, 0); S.last = { src: c }; if (S.model) analyse(c, true); }
  if (v.srcObject) { v.srcObject.getTracks().forEach((tr) => tr.stop()); v.srcObject = null; }
  v.pause(); if (v.src && v.src.startsWith("blob:")) URL.revokeObjectURL(v.src);
  $("cam").textContent = t("camOn"); $("cam").classList.remove("on"); $("flipCam").hidden = true;
}

let toastTimer;
function toast(s) { const e = $("toast"); e.textContent = s; e.classList.add("on"); clearTimeout(toastTimer); toastTimer = setTimeout(() => e.classList.remove("on"), 2400); }

/* ---------------------------------------------------------------- events */

function applyLang() {
  document.documentElement.lang = lang;
  $("lang").textContent = lang === "ko" ? "EN" : "한국어";
  document.querySelectorAll("[data-t]").forEach((e) => { e.textContent = t(e.dataset.t); });
  if (S.live && S.live.cam) $("cam").textContent = t("camOff");
  if (S.list.length) drawHub();
  drawMine();
}
$("lang").addEventListener("click", () => { lang = lang === "ko" ? "en" : "ko"; try { localStorage.setItem("hub.lang", lang); } catch {} applyLang(); });
$("src").addEventListener("click", (e) => {
  const b = e.target.closest("[data-src]"); if (!b || b.dataset.src === S.src) return;
  S.src = b.dataset.src;
  $("src").querySelectorAll(".db").forEach((x) => x.classList.toggle("on", x === b));
  $("hubBox").hidden = S.src !== "hub"; $("mineBox").hidden = S.src !== "mine";
  if (S.src === "hub") loadHub(); else loadMine();
});
$("ds").addEventListener("change", () => {
  S.ds = $("ds").value; const d = dataset();
  if (!d.tasks.includes(S.task)) S.task = d.tasks[0];
  if (!d.variants.some((v) => v.task === S.task && v.imgsz === S.size)) S.size = 320;
  S.int8 = false; drawHub(); loadHub();
});
$("tasks").addEventListener("click", (e) => { const b = e.target.closest("[data-task]"); if (!b) return; S.task = b.dataset.task; S.int8 = false; drawHub(); loadHub(); });
$("sizes").addEventListener("click", (e) => { const b = e.target.closest("[data-size]"); if (!b) return; S.size = Number(b.dataset.size); S.int8 = !!b.dataset.int8; drawHub(); loadHub(); });
$("mineFiles").addEventListener("change", (e) => { addMine([...e.target.files]); e.target.value = ""; });
$("mineTasks").addEventListener("click", (e) => { const b = e.target.closest("[data-mtask]"); if (!b) return; S.mine.task = b.dataset.mtask; drawMine(); loadMine(); });
for (const [zone, on] of [[$("mineDrop"), (fs) => addMine(fs)], [$("stage"), (fs) => fs[0] && openFile(fs[0])]]) {
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); zone.classList.remove("over"); on([...e.dataTransfer.files]); });
}
$("pick").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) openFile(f); e.target.value = ""; });
$("cam").addEventListener("click", camera);
$("flipCam").addEventListener("click", () => { S.facing = S.facing === "user" ? "environment" : "user"; stopLive(); camera(); });
let ct;
$("conf").addEventListener("input", () => {
  S.conf = Number($("conf").value); $("confV").textContent = S.conf.toFixed(2);
  clearTimeout(ct); ct = setTimeout(() => { if (!S.live && S.last && S.last.raw) finish(S.last); }, 60);
});
$("found").addEventListener("click", (e) => {
  const b = e.target.closest("[data-cls]"); if (!b) return; const k = Number(b.dataset.cls);
  if (S.hidden.has(k)) S.hidden.delete(k); else S.hidden.add(k);
  if (!S.live && S.last && S.last.raw) finish(S.last);
});
$("ep").value = ep;
$("ep").addEventListener("change", () => { try { localStorage.setItem("try.ep", $("ep").value); } catch {} location.reload(); });
$("save").addEventListener("click", () => {
  $("out").toBlob((b) => { const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "easydetect.png"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 30000); }, "image/png");
});
window.addEventListener("hashchange", () => {
  const [want, name] = decodeURIComponent(location.hash.slice(1)).split("/");
  const d = S.list.find((x) => x.id === want); if (!d) return;
  const v = name && d.variants.find((x) => x.name === name.replace(/-int8$/, ""));
  const int8 = /-int8$/.test(name || "");
  if (S.ds === d.id && v && v.task === S.task && v.imgsz === S.size && int8 === S.int8) return;
  S.ds = d.id; if (v) { S.task = v.task; S.size = v.imgsz; S.int8 = int8; } else { S.task = d.tasks[0]; S.size = 320; S.int8 = false; }
  if (S.src !== "hub") $("src").querySelector('[data-src="hub"]').click(); else { drawHub(); loadHub(); }
});

applyLang();
loadList();
