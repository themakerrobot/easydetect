// Apache-2.0
/* easydetect in the browser: the same pre- and postprocessing as the Python
   package (easydetect/predictor.py, pose.py, segment.py), on ONNX Runtime Web.
   Pictures never leave the page: models are fetched once and kept in the
   browser's cache, and everything runs here. */

const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
export const ASSETS = "https://huggingface.co/leeyunjai/easydetect/resolve/main";
export const ENCODER = `${ASSETS}/mobile_sam/encoder.onnx`;
export const SAM_DECODER = `${ASSETS}/mobile_sam/decoder.onnx`;
export const POSE = `${ASSETS}/pose/pose-s.onnx`;

export const CONF = 0.5;
const IOU = 0.7;
const MAX_DET = 300;

let ort = null;
let gpu = false;

/** Load ONNX Runtime Web once: its WebGPU build when the browser has a GPU
    adapter and "wasm" was not asked for, its WASM build otherwise. */
export const preferred = () => (gpu ? "webgpu" : "wasm");

export async function runtime(want = "auto") {
  if (ort) return { ort, gpu };
  if (want !== "wasm" && navigator.gpu) {
    try { gpu = !!(await navigator.gpu.requestAdapter()); } catch { gpu = false; }
  }
  ort = await import(ORT + (gpu ? "ort.webgpu.min.mjs" : "ort.wasm.min.mjs"));
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  return { ort, gpu };
}

/* ---------------------------------------------------------------- files */

const CACHE = "easydetect-models-v1";

/** A model's bytes: from the browser's cache, or fetched (with progress
    ``onProgress(done, total)``) and kept there for next time. */
export async function fetchModel(url, onProgress = () => {}) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch {}
  const hit = cache && await cache.match(url).catch(() => null);
  if (hit) { const b = new Uint8Array(await hit.arrayBuffer()); onProgress(b.length, b.length); return b; }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url.split("/").pop()}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const parts = []; let done = 0;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (end) break;
    parts.push(value); done += value.length; onProgress(done, total);
  }
  const bytes = new Uint8Array(done); let at = 0;
  for (const p of parts) { bytes.set(p, at); at += p.length; }
  if (cache) cache.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } })).catch(() => {});
  return bytes;
}

/** The ``metadata_props`` of an ONNX file (ModelProto field 14, each a key = 1,
    value = 2 pair): only the top level is walked, the graph is skipped. */
export function onnxMetadata(bytes) {
  const props = {}; const dec = new TextDecoder();
  let i = 0;
  const varint = (end) => {
    let v = 0, m = 1;
    while (i < end) { const b = bytes[i++]; v += (b & 0x7f) * m; if (b < 0x80) return v; m *= 128; }
    return -1;
  };
  const n = bytes.length;
  while (i < n) {
    const tag = varint(n); if (tag < 0) break;
    const field = Math.floor(tag / 8), wire = tag % 8;
    if (wire === 0) varint(n);
    else if (wire === 1) i += 8;
    else if (wire === 5) i += 4;
    else if (wire === 2) {
      const len = varint(n); const end = i + len;
      if (field === 14) {
        let k = "", v = "";
        while (i < end) {
          const t = varint(end); const l = varint(end);
          const s = dec.decode(bytes.subarray(i, i + l)); i += l;
          if (t >> 3 === 1) k = s; else if (t >> 3 === 2) v = s;
        }
        if (k) props[k] = v;
      }
      i = end;
    } else break;
  }
  return props;
}

async function session(bytes, prefer) {
  const opts = (ep) => ({ executionProviders: [ep], graphOptimizationLevel: "all" });
  if (prefer === "webgpu" && gpu) {
    try { return { s: await ort.InferenceSession.create(bytes, opts("webgpu")), ep: "webgpu" }; } catch (e) { console.warn("webgpu:", e); }
  }
  return { s: await ort.InferenceSession.create(bytes, opts("wasm")), ep: "wasm" };
}

/** What a head model is, from its inputs: a keypoint network ("pose") or a
    SAM mask decoder ("segment"). */
export async function headKind(bytes) {
  const { s } = await session(bytes, "wasm");
  const ins = s.inputNames, outs = s.outputNames; s.release?.();
  if (ins.includes("image_embeddings")) return "segment";
  if (outs.includes("boxes") || outs.includes("scores")) return null;
  return outs.length === 2 ? "pose" : null;
}

/* ---------------------------------------------------------------- picture helpers */

function canvas(w, h) {
  const c = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h) : Object.assign(document.createElement("canvas"), { width: w, height: h });
  return c;
}
/** RGBA pixels → planar RGB float32, times ``scale``. */
function planar(data, n, scale, out = new Float32Array(3 * n), at = 0) {
  for (let p = 0, q = 0; p < n; p++, q += 4) {
    out[at + p] = data[q] * scale;
    out[at + n + p] = data[q + 1] * scale;
    out[at + 2 * n + p] = data[q + 2] * scale;
  }
  return out;
}

/* ---------------------------------------------------------------- detector */

export class Model {
  /** ``det`` the detector's bytes; ``head`` a keypoint network or mask
      decoder (optional); ``task`` detect | segment | pose; ``prefer`` webgpu | wasm. */
  static async create({ det, head = null, task = "detect", encoder = null, names = null, prefer = "wasm", int8 = false }) {
    const m = new Model();
    const meta = onnxMetadata(det);
    m.task = task;
    m.names = names || {};
    try { if (meta["easydetect.names"]) m.names = Object.fromEntries(Object.entries(JSON.parse(meta["easydetect.names"])).map(([k, v]) => [Number(k), String(v)])); } catch {}
    m.conf = Number(meta["easydetect.conf"]) || CONF;
    // quantised graphs run on the CPU kernels: WebGPU would only shuttle them back and forth
    const ep = int8 ? "wasm" : prefer;
    const d = await session(det, ep);
    m.det = d.s; m.ep = d.ep;
    const shape = m.det.inputMetadata?.[0]?.shape || [];
    m.size = Number(shape[2]) > 0 ? Number(shape[2]) : 640;
    m.inName = m.det.inputNames[0];
    if (task === "pose" && head) {
      const p = await session(head, ep);
      m.pose = p.s; m.spec = specOf(onnxMetadata(head));
    }
    if (task === "segment" && head && encoder) {
      m.encoder = (await session(encoder, ep)).s;
      m.decoder = (await session(head, "wasm")).s;
    }
    return m;
  }

  release() { for (const s of [this.det, this.pose, this.encoder, this.decoder]) s?.release?.(); }

  /** Raw detector output for a picture (a canvas): boxes (Q x 4, cxcywh 0-1)
      and per-class scores (Q x C). */
  async detect(src) {
    const S = this.size; const c = canvas(S, S); const g = c.getContext("2d", { willReadFrequently: true });
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "medium";
    g.drawImage(src, 0, 0, S, S);
    const px = g.getImageData(0, 0, S, S).data;
    const x = new ort.Tensor("float32", planar(px, S * S, 1 / 255), [1, 3, S, S]);
    const out = await this.det.run({ [this.inName]: x });
    const names = this.det.outputNames;
    const boxes = out[names.includes("boxes") ? "boxes" : names[0]];
    const scores = out[names.includes("scores") ? "scores" : names[1]];
    return { boxes: boxes.data, scores: scores.data, q: boxes.dims[1], k: scores.dims[2] };
  }

  /** Raw output → ``[{box: [x1, y1, x2, y2], score, cls}]`` in picture pixels:
      each query's best class, at least ``conf``, best first, a box covering a
      better one by more than IoU 0.7 dropped (predictor.postprocess). */
  boxes(raw, w, h, conf = this.conf) {
    const { boxes, scores, q, k } = raw;
    let logits = false;
    for (let i = 0; i < scores.length; i++) if (scores[i] < 0 || scores[i] > 1) { logits = true; break; }
    let out = [];
    for (let i = 0; i < q; i++) {
      let best = -Infinity, cls = 0;
      for (let j = 0; j < k; j++) { const s = scores[i * k + j]; if (s > best) { best = s; cls = j; } }
      if (logits) best = 1 / (1 + Math.exp(-best));
      if (best < conf) continue;
      const cx = boxes[i * 4], cy = boxes[i * 4 + 1], bw = boxes[i * 4 + 2], bh = boxes[i * 4 + 3];
      const clip = (v, hi) => Math.min(Math.max(v, 0), hi);
      out.push({ box: [clip((cx - bw / 2) * w, w), clip((cy - bh / 2) * h, h), clip((cx + bw / 2) * w, w), clip((cy + bh / 2) * h, h)], score: best, cls });
    }
    out.sort((a, b) => b.score - a.score);
    out = out.slice(0, MAX_DET);
    const area = (b) => Math.max(b[2] - b[0], 0) * Math.max(b[3] - b[1], 0);
    const alive = out.map(() => true);
    for (let i = 0; i < out.length; i++) {
      if (!alive[i]) continue;
      const a = out[i].box;
      for (let j = i + 1; j < out.length; j++) {
        if (!alive[j]) continue;
        const b = out[j].box;
        const iw = Math.max(Math.min(a[2], b[2]) - Math.max(a[0], b[0]), 0);
        const ih = Math.max(Math.min(a[3], b[3]) - Math.max(a[1], b[1]), 0);
        const inter = iw * ih;
        if (inter / Math.max(area(a) + area(b) - inter, 1e-9) > IOU) alive[j] = false;
      }
    }
    return out.filter((_, i) => alive[i]);
  }

  /* ------------------------------------------------------------ keypoints */

  /** Which detections get keypoints: the classes the model was trained for,
      or for COCO's model the boxes named "person" (every box when there is
      no such class). */
  keypointRows(dets) {
    if (this.spec.classes) return dets.filter((d) => this.spec.classes.includes(d.cls));
    const person = Object.entries(this.names).filter(([, v]) => v.toLowerCase() === "person").map(([k]) => Number(k));
    return person.length ? dets.filter((d) => person.includes(d.cls)) : dets;
  }

  /** Keypoints for ``dets`` (each gets ``.kpts``: [x, y, conf] per keypoint). */
  async keypoints(src, dets, { flip = true, limit = 20 } = {}) {
    const people = this.keypointRows(dets).slice(0, limit);
    if (!people.length) return;
    const W = 192, H = 256, n = W * H;
    const maps = people.map((d) => cropMatrix(d.box));
    const count = people.length * (flip ? 2 : 1);
    const data = new Float32Array(count * 3 * n);
    const c = canvas(W, H); const g = c.getContext("2d", { willReadFrequently: true });
    maps.forEach((m, i) => {
      g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = "#000"; g.fillRect(0, 0, W, H);
      // cv2.warpAffine maps pixel centres; a canvas maps pixel corners
      g.setTransform(m[0], m[3], m[1], m[4], m[2] - 0.5 * (m[0] + m[1]) + 0.5, m[5] - 0.5 * (m[3] + m[4]) + 0.5);
      g.drawImage(src, 0, 0);
      const px = g.getImageData(0, 0, W, H).data;
      planar(px, n, 1, data, i * 3 * n);
      if (flip) {
        const at = (people.length + i) * 3 * n, from = i * 3 * n;
        for (let ch = 0; ch < 3; ch++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
          data[at + ch * n + y * W + x] = data[from + ch * n + y * W + (W - 1 - x)];
      }
    });
    const name = this.pose.inputNames[0];
    const out = await this.pose.run({ [name]: new ort.Tensor("float32", data, [count, 3, H, W]) });
    const [xo, yo] = this.pose.outputNames.map((k) => out[k]);
    const K = xo.dims[1], BX = xo.dims[2], BY = yo.dims[2];
    const xs = xo.data, ys = yo.data;
    const P = people.length;
    for (let p = 0; p < P; p++) {
      const kpts = [];
      for (let k = 0; k < K; k++) {
        const lx = new Float32Array(BX), ly = new Float32Array(BY);
        const ox = (p * K + k) * BX, oy = (p * K + k) * BY;
        for (let b = 0; b < BX; b++) lx[b] = xs[ox + b];
        for (let b = 0; b < BY; b++) ly[b] = ys[oy + b];
        if (flip) {  // the mirror's keypoint k is this crop's flip[k], its x bins reversed (one bin off)
          const f = this.spec.flip[k];
          const fx = ((P + p) * K + f) * BX, fy = ((P + p) * K + f) * BY;
          for (let b = 0; b < BX; b++) lx[b] = (lx[b] + xs[fx + (b < BX - 1 ? BX - 2 - b : 0)]) / 2;
          for (let b = 0; b < BY; b++) ly[b] = (ly[b] + ys[fy + b]) / 2;
        }
        const [x, cx] = decode1(lx, PEAK_X), [y, cy] = decode1(ly, PEAK_Y);
        const m = invert(maps[p]);
        kpts.push([m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5], Math.min(cx, cy)]);
      }
      people[p].kpts = kpts;
    }
  }

  /* ------------------------------------------------------------ masks */

  /** The encoder's picture embedding (the slow part, once a picture). */
  async embed(src, w, h) {
    const S = 1024; const s = S / Math.max(w, h);
    const c = canvas(S, S); const g = c.getContext("2d", { willReadFrequently: true });
    g.fillStyle = "#000"; g.fillRect(0, 0, S, S);
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    g.drawImage(src, 0, 0, Math.round(w * s), Math.round(h * s));
    const px = g.getImageData(0, 0, S, S).data;
    const out = await this.encoder.run({ [this.encoder.inputNames[0]]: new ort.Tensor("float32", planar(px, S * S, 1), [1, 3, S, S]) });
    return { emb: out[this.encoder.outputNames[0]], scale: s, w, h };
  }

  /** A mask for each of ``dets`` from an embedding (each gets ``.mask``: the
      256 x 256 logits and the box region it may cover). */
  async masks(embedding, dets, limit = 40) {
    const rows = dets.slice(0, limit);
    if (!rows.length) return;
    const { emb, scale, w, h } = embedding; const N = rows.length;
    const coords = new Float32Array(N * 4), labels = new Float32Array(N * 2);
    rows.forEach((d, i) => { for (let j = 0; j < 4; j++) coords[i * 4 + j] = d.box[j] * scale; labels[i * 2] = 2; labels[i * 2 + 1] = 3; });
    const r = 256 / Math.max(w, h);
    const out = await this.decoder.run({
      image_embeddings: emb,
      point_coords: new ort.Tensor("float32", coords, [N, 2, 2]),
      point_labels: new ort.Tensor("float32", labels, [N, 2]),
      mask_input: new ort.Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]),
      has_mask_input: new ort.Tensor("float32", new Float32Array(1), [1]),
      orig_im_size: new ort.Tensor("float32", new Float32Array([Math.max(Math.round(h * r), 1), Math.max(Math.round(w * r), 1)]), [2]),
    });
    const low = out.low_res_masks.data;
    rows.forEach((d, i) => { d.mask = { logits: low.subarray(i * 65536, (i + 1) * 65536), region: region(d.box, w, h) }; });
  }
}

/* ---------------------------------------------------------------- pose geometry */

const SPLIT = 2, PAD = 1.25;
const peak = (sigma, bins) => {
  let max = 0, sum = 0;
  for (let i = 0; i < bins; i++) { const d = i - bins / 2; const g = Math.exp(-(d * d) / (2 * sigma * sigma)); sum += g; if (g > max) max = g; }
  return max / sum;
};
const PEAK_X = peak(4.9, 192 * SPLIT), PEAK_Y = peak(5.66, 256 * SPLIT);

/** One keypoint's bins → (position in crop pixels, confidence 0-1): the peak
    moved by a parabola through its log-probabilities (pose.decode). */
function decode1(l, peakMax) {
  const n = l.length; let p = 0;
  for (let i = 1; i < n; i++) if (l[i] > l[p]) p = i;
  let sum = 0; for (let i = 0; i < n; i++) sum += Math.exp(l[i] - l[p]);
  const conf = Math.min(Math.max((1 / sum) / peakMax, 0), 1);
  const inner = Math.min(Math.max(p, 1), n - 2);
  const a = l[inner - 1], b = l[inner], c = l[inner + 1];
  const curve = a - 2 * b + c;
  let off = curve < 0 ? 0.5 * (a - c) / curve : 0;
  off = p === inner ? Math.min(Math.max(off, -0.5), 0.5) : 0;
  return [(p + off) / SPLIT, conf];
}

/** A box → the 2x3 map (row-major, 6 numbers) from picture pixels to the
    256 x 192 crop: widened to 3:4 and padded (pose.box_to_crop, crop_matrix). */
function cropMatrix([x1, y1, x2, y2]) {
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  let w = Math.max(x2 - x1, 1), h = Math.max(y2 - y1, 1);
  const aspect = 192 / 256;
  if (w > aspect * h) h = w / aspect; else w = h * aspect;
  w *= PAD; h *= PAD;
  const sx = 192 / w, sy = 256 / h;
  return [sx, 0, 96 - sx * cx, 0, sy, 128 - sy * cy];
}
function invert([a, b, c, d, e, f]) {
  const det = a * e - b * d;
  return [e / det, -b / det, (b * f - c * e) / det, -d / det, a / det, (c * d - a * f) / det];
}

export const COCO_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15];
export const COCO_SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
  [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]];
function specOf(meta) {
  try {
    const s = JSON.parse(meta["easydetect.keypoints"]);
    return { names: s.names, flip: s.flip || s.names.map((_, i) => i), skeleton: s.skeleton || [], classes: s.classes ?? null, coco: false };
  } catch {
    return { names: null, flip: COCO_FLIP, skeleton: COCO_SKELETON, classes: null, coco: true };
  }
}

/* ---------------------------------------------------------------- mask geometry */

function region([x1, y1, x2, y2], w, h) {
  const px = (x2 - x1) * 0.1, py = (y2 - y1) * 0.1;
  return [Math.max(Math.floor(x1 - px), 0), Math.max(Math.floor(y1 - py), 0), Math.min(Math.ceil(x2 + px), w), Math.min(Math.ceil(y2 + py), h)];
}

/** Paint each detection's mask onto ``img`` (an ImageData of the picture, w x h):
    the 256 x 256 logits read bilinearly at each picture pixel of the box's
    region, kept where above 0 (segment.upscale_logits), tinted ``alpha``. */
export function paintMasks(img, dets, colour, alpha = 0.45) {
  const { width: w, height: h, data } = img;
  const s = 1024 / Math.max(w, h);
  const pw = Math.floor(w * s + 0.5), ph = Math.floor(h * s + 0.5);
  const fx = pw / w, fy = ph / h, k = 0.25;
  for (const d of dets) {
    if (!d.mask) continue;
    const { logits, region: [x1, y1, x2, y2] } = d.mask;
    const [r, g, b] = colour(d.cls);
    const lxs = new Float32Array(x2 - x1);
    for (let x = x1; x < x2; x++) lxs[x - x1] = Math.min(Math.max((Math.min(Math.max((x + 0.5) * fx - 0.5, 0), pw - 1) + 0.5) * k - 0.5, 0), 255);
    for (let y = y1; y < y2; y++) {
      const ly = Math.min(Math.max((Math.min(Math.max((y + 0.5) * fy - 0.5, 0), ph - 1) + 0.5) * k - 0.5, 0), 255);
      const y0 = Math.floor(ly), y1b = Math.min(y0 + 1, 255), ty = ly - y0;
      for (let x = x1; x < x2; x++) {
        const lx = lxs[x - x1];
        const x0 = Math.floor(lx), x1b = Math.min(x0 + 1, 255), tx = lx - x0;
        const v = (logits[y0 * 256 + x0] * (1 - tx) + logits[y0 * 256 + x1b] * tx) * (1 - ty)
                + (logits[y1b * 256 + x0] * (1 - tx) + logits[y1b * 256 + x1b] * tx) * ty;
        if (v <= 0) continue;
        const q = (y * w + x) * 4;
        data[q] = data[q] * (1 - alpha) + r * alpha;
        data[q + 1] = data[q + 1] * (1 - alpha) + g * alpha;
        data[q + 2] = data[q + 2] * (1 - alpha) + b * alpha;
      }
    }
  }
}

/* ---------------------------------------------------------------- drawing (plotting.py) */

/* BGR in plotting.py, RGB here */
export const PALETTE = [[255, 56, 56], [255, 157, 151], [255, 112, 31], [255, 178, 29], [207, 210, 49], [72, 249, 10], [146, 204, 23],
  [61, 219, 134], [26, 147, 52], [0, 212, 187], [44, 153, 168], [0, 194, 255], [52, 69, 147], [100, 115, 255], [0, 24, 236],
  [132, 56, 255], [82, 0, 133], [203, 56, 255], [255, 149, 200], [255, 55, 199]];
export const colour = (i) => PALETTE[i % PALETTE.length];
const LIMB = { face: [80, 200, 255], arm: [255, 220, 80], body: [120, 255, 120], leg: [200, 120, 255] };
const EDGE = [[80, 200, 255], [255, 220, 80], [120, 255, 120], [200, 120, 255], [255, 160, 80], [255, 120, 200], [120, 255, 255], [200, 200, 120]];
function limb(a, b) {
  const s = [a, b];
  const is = (x, y) => (a === x && b === y) || (a === y && b === x);
  if (is(5, 6) || is(5, 11) || is(6, 12) || is(11, 12)) return "body";
  if (s.every((v) => v >= 5 && v <= 10)) return "arm";
  if (s.every((v) => v >= 11 && v <= 16)) return "leg";
  return "face";
}

/** Draw a picture and its detections (masks, boxes and labels, keypoints) onto canvas ``c``. */
export function render(c, src, dets, model) {
  const w = src.width, h = src.height;
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const g = c.getContext("2d", { willReadFrequently: model.task === "segment" });
  g.drawImage(src, 0, 0);
  if (model.task === "segment") {
    const img = g.getImageData(0, 0, w, h);
    paintMasks(img, dets, colour);
    g.putImageData(img, 0, 0);
  }
  const lw = Math.max(Math.round((w + h) / 2 * 0.003), 2);
  const fs = Math.max(Math.round((w + h) / 2 * 0.022), 12);
  g.font = `600 ${fs}px 'Pretendard Variable', Pretendard, sans-serif`; g.textBaseline = "top";
  const placed = [];
  for (const d of dets) {
    const [x1, y1, x2, y2] = d.box; const [r, gg, b] = colour(d.cls);
    g.strokeStyle = `rgb(${r},${gg},${b})`; g.lineWidth = lw; g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${model.names[d.cls] ?? d.cls} ${d.score.toFixed(2)}`;
    const tw = g.measureText(label).width + fs * 0.5, th = fs * 1.3;
    const left = Math.max(0, Math.min(x1, w - tw));
    let top = null;
    for (const y of [y1 - th, y2, y1, y1 + th]) {   // above the box, below, inside: the first free slot
      const yy = Math.max(0, Math.min(y, h - th));
      if (!placed.some((p) => left < p[2] && left + tw > p[0] && yy < p[3] && yy + th > p[1])) { top = yy; break; }
    }
    if (top === null) top = Math.max(0, Math.min(y1 - th, h - th));
    placed.push([left, top, left + tw, top + th]);
    g.fillStyle = `rgb(${r},${gg},${b})`; g.fillRect(left, top, tw, th);
    g.fillStyle = (r * 0.299 + gg * 0.587 + b * 0.114) > 150 ? "#111" : "#fff";
    g.fillText(label, left + fs * 0.25, top + fs * 0.15);
  }
  if (model.task === "pose") {
    const sk = model.spec?.skeleton || COCO_SKELETON; const coco = model.spec?.coco !== false;
    for (const d of dets) {
      if (!d.kpts) continue;
      const seen = d.kpts.map((p) => p[2] >= 0.3);
      sk.forEach(([a, b], n) => {
        if (!seen[a] || !seen[b]) return;
        const [r, gg, bb] = coco ? LIMB[limb(a, b)] : EDGE[n % EDGE.length];
        g.strokeStyle = `rgb(${r},${gg},${bb})`; g.lineWidth = lw; g.lineCap = "round";
        g.beginPath(); g.moveTo(d.kpts[a][0], d.kpts[a][1]); g.lineTo(d.kpts[b][0], d.kpts[b][1]); g.stroke();
      });
      for (const [x, y] of d.kpts.filter((_, i) => seen[i])) {
        g.beginPath(); g.arc(x, y, lw + 1, 0, Math.PI * 2); g.fillStyle = "#fff"; g.fill();
        g.lineWidth = 1; g.strokeStyle = "#282828"; g.stroke();
      }
    }
  }
}



/* ---------------------------------------------------------------- the whole pass */

/** A picture (or video frame) as a canvas no bigger than ``max`` px a side;
    ``reuse`` a canvas of the right size to draw into. */
export function frame(el, max = 1600, reuse = null) {
  const w0 = el.videoWidth || el.naturalWidth || el.width, h0 = el.videoHeight || el.naturalHeight || el.height;
  if (!w0 || !h0) return null;
  const k = Math.min(1, max / Math.max(w0, h0));
  const w = Math.round(w0 * k), h = Math.round(h0 * k);
  const c = reuse && reuse.width === w && reuse.height === h ? reuse : Object.assign(document.createElement("canvas"), { width: w, height: h });
  c.getContext("2d").drawImage(el, 0, 0, w, h);
  return c;
}

/** Detections for a canvas with their keypoints or masks: ``{all, dets}``,
    ``dets`` without the classes in ``hidden``. */
export async function analyse(model, src, { conf = model.conf, hidden = null, flip = true } = {}) {
  const raw = await model.detect(src);
  const emb = model.task === "segment" && model.encoder ? await model.embed(src, src.width, src.height) : null;
  return { src, raw, emb, model, ...(await heads({ src, raw, emb, model }, { conf, hidden, flip })) };
}
/** The postprocessing and heads of ``analyse`` again, for a new threshold. */
export async function heads({ src, raw, emb, model }, { conf = model.conf, hidden = null, flip = true } = {}) {
  const all = model.boxes(raw, src.width, src.height, conf);
  const dets = hidden ? all.filter((d) => !hidden.has(d.cls)) : all;
  if (model.task === "pose" && model.pose) await model.keypoints(src, dets, { flip });
  if (model.task === "segment" && emb) await model.masks(emb, dets);
  return { all, dets };
}
