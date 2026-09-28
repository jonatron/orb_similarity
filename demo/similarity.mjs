// Similarity demo: ImageDecoder → grayscale → OpenCV.js ORB per image, then
// pairwise Hamming-distance matching via similarity-core (wasm i32.popcnt
// when popcnt.wasm is available, JS fallback otherwise).
import { scanPair as scanPairJS, makeWasmScanPair, summarize, countGrid } from './similarity-core.mjs';
import { loadOrb } from './orb-opencv.mjs';

const $ = (id) => document.getElementById(id);
const stripSec = $('stripSec');
const matrixSec = $('matrixSec');
const rankSec = $('rankSec');
const pairSec = $('pairSec');
const stripEl = $('strip');
const statusEl = $('status');
const matrixEl = $('matrix');
const rankEl = $('ranking');
const pairViewEl = $('pairView');
const pairInfoEl = $('pairInfo');
const timingSec = $('timingSec');
const imgTimingEl = $('imgTiming');
const pairTimingEl = $('pairTiming');
const fileInput = $('file');
const loadTestBtn = $('loadTest');
const thrInput = $('thr');
const thrVal = $('thrVal');
const nfInput = $('nf');
const nfVal = $('nfVal');

let scanPair = scanPairJS; // upgraded to the wasm popcnt path once loaded
let images = []; // { name, bmp, w, h, n, kps, desc }
let pairs = [];  // { i, j, ab, ba }
let timings = []; // { name, decode, gray, orb } — ms per pipeline stage
let pairMs = [];  // ms for both scanPair() calls of each pair
let selected = -1;
let reprocessP = null; // serializes ORB re-runs (slider drags / file loads)

let popcntExports = null; // null until popcnt.wasm is loaded and the scan upgraded
async function loadPopcnt() {
  if (popcntExports) return popcntExports;
  try {
    const res = await fetch('../build/popcnt.wasm?v=2');
    const bytes = await res.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {
      env: { abort: () => { throw new Error('wasm abort'); }, trace() {}, seed: () => 123 },
    });
    popcntExports = instance.exports;
    scanPair = makeWasmScanPair(popcntExports);
    return popcntExports;
  } catch (e) {
    // keep the JS fallback; the demo still works without the wasm module
    return null;
  }
}

// OpenCV's BGR2GRAY rounding formula applied to sRGB (R, G, B):
//   gray = (114*B + 587*G + 299*R + 500) / 1000
function rgbaToGray(img) {
  const w = img.width;
  const h = img.height;
  const off = new OffscreenCanvas(w, h);
  const octx = off.getContext('2d', { willReadFrequently: true });
  octx.drawImage(img, 0, 0);
  const { data } = octx.getImageData(0, 0, w, h);
  const gray = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = (114 * data[p + 2] + 587 * data[p + 1] + 299 * data[p] + 500) / 1000 | 0;
  }
  return { gray, w, h };
}

async function runOrb(gray, w, h) {
  const orbw = await loadOrb();
  orbw.setup(w, h);
  const sf = Math.fround(1.2);
  for (let l = 0; l < 8; l++) orbw.setLayerScale(l, Math.fround(Math.pow(sf, l)));
  orbw.setNDPow(Math.fround(Math.pow(Math.fround(1 / sf), 8)));
  orbw.setNFeatures(+nfInput.value);
  const input = new Uint8Array(orbw.memory.buffer, orbw.inputPtr(), gray.length);
  input.set(gray);
  const n = orbw.run();
  const kps = [];
  for (let i = 0; i < n; i++) {
    kps.push({ x: orbw.kpX(i), y: orbw.kpY(i), size: orbw.kpSize(i), angle: orbw.kpAngle(i) });
  }
  const desc = new Uint8Array(orbw.memory.buffer, orbw.descPtr(), orbw.descLen());
  return { n, kps, desc: desc.slice() };
}

async function decodeSource(src) {
  const dec = new ImageDecoder({ data: src.data, type: src.type });
  await dec.tracks.ready;
  const { image } = await dec.decode();
  await image.prepareForRendering?.();
  const bmp = await createImageBitmap(image);
  return bmp;
}

// Loads a list of image sources ({ name, type, data: ArrayBuffer }) and runs
// the full pipeline: decode → grayscale → ORB → pairwise matching.
async function loadImages(sources) {
  statusEl.textContent = 'Decoding images…';
  images = [];
  pairs = [];
  timings = [];
  pairMs = [];
  selected = -1;
  let failed = 0;
  loadPopcnt(); // fetch while the images decode below
  for (const src of sources) {
    try {
      const tDec = performance.now();
      const bmp = await decodeSource(src);
      const decode = performance.now() - tDec;
      images.push({ name: src.name, bmp, decodeMs: decode });
    } catch (e) {
      failed++;
      console.error(src.name, e);
    }
  }
  if (!images.length) {
    statusEl.textContent = 'No images could be decoded.';
    return;
  }
  await (reprocessP = Promise.resolve(reprocessP).then(processImages));
  if (failed) statusEl.textContent += ` (${failed} failed to decode)`;
}

fileInput.addEventListener('change', async () => {
  const files = [...fileInput.files];
  fileInput.value = '';
  if (!files.length) return;
  const sources = await Promise.all(files.map(async (f) => ({
    name: f.name,
    type: f.type,
    data: await f.arrayBuffer(),
  })));
  await loadImages(sources);
});

const TEST_IMAGE_COUNT = 20;

loadTestBtn.addEventListener('click', async () => {
  statusEl.textContent = `Loading ${TEST_IMAGE_COUNT} test images…`;
  const sources = [];
  for (let k = 0; k < TEST_IMAGE_COUNT; k++) {
    const name = `ukbench${String(k).padStart(5, '0')}.jpg`;
    const res = await fetch(`../test_images/${name}`);
    if (!res.ok) throw new Error(`${name}: fetch failed (${res.status})`);
    const data = await res.arrayBuffer();
    sources.push({ name, type: 'image/jpeg', data });
  }
  await loadImages(sources);
});

// ORB extraction + pairwise matching for the loaded images, then re-render.
// Runs on load and whenever the descriptor-count slider changes.
async function processImages() {
  const t0 = performance.now();
  const popcntP = loadPopcnt(); // fetch while ORB runs below
  timings = [];
  for (const img of images) {
    const tGray = performance.now();
    const { gray, w, h } = rgbaToGray(img.bmp);
    const grayMs = performance.now() - tGray;

    const tOrb = performance.now();
    const { n, kps, desc } = await runOrb(gray, w, h);
    const orbMs = performance.now() - tOrb;

    img.w = w;
    img.h = h;
    img.n = n;
    img.kps = kps;
    img.desc = desc;
    timings.push({ name: img.name, decode: img.decodeMs, gray: grayMs, orb: orbMs });
  }
  pairs = [];
  pairMs = [];
  await popcntP;
  const usingWasm = scanPair !== scanPairJS;
  for (let i = 0; i < images.length; i++) {
    for (let j = i + 1; j < images.length; j++) {
      const tScan = performance.now();
      const ab = scanPair(images[i].desc, images[j].desc);
      const ba = scanPair(images[j].desc, images[i].desc);
      pairMs.push(performance.now() - tScan);
      pairs.push({ i, j, ab, ba });
    }
  }
  selected = bestPairIdx();
  renderStrip();
  renderTiming();
  const ms = (performance.now() - t0).toFixed(0);
  if (!pairs.length) {
    statusEl.textContent =
      `Processed 1 image in ${ms} ms. Select at least 2 images to compare pairs.`;
    return;
  }
  renderMatrix();
  renderRanking();
  renderPair();
  statusEl.textContent =
    `${images.length} image(s) processed in ${ms} ms, ${pairs.length} pair(s) compared` +
    ` (matching: ${usingWasm ? 'wasm popcnt' : 'JS fallback'}).`;
}

nfInput.addEventListener('input', () => {
  nfVal.textContent = nfInput.value;
  if (!images.length) return;
  reprocessP = Promise.resolve(reprocessP).then(processImages);
});

thrInput.addEventListener('input', () => {
  thrVal.textContent = thrInput.value;
  if (!pairs.length) return;
  selected = bestPairIdx();
  renderMatrix();
  renderRanking();
  renderPair();
});

function selectPair(idx) {
  if (idx === selected) return;
  selected = idx;
  renderMatrix();
  renderRanking();
  renderPair();
}

// Index of the pair with the most good matches (ties: lower mean distance).
function bestPairIdx() {
  let best = 0;
  let bestS = null;
  for (let k = 0; k < pairs.length; k++) {
    const s = pairStats(pairs[k]);
    if (!bestS || s.good > bestS.good || (s.good === bestS.good && s.mean < bestS.mean)) {
      best = k;
      bestS = s;
    }
  }
  return best;
}

function pairStats(p) {
  return summarize(p.ab, p.ba, +thrInput.value);
}

function renderStrip() {
  stripSec.hidden = false;
  stripEl.textContent = '';
  for (const img of images) {
    const fig = document.createElement('figure');
    const canvas = document.createElement('canvas');
    const scale = Math.min(1, 200 / img.w);
    canvas.width = Math.round(img.w * scale);
    canvas.height = Math.round(img.h * scale);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img.bmp, 0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#ff2d2d';
    ctx.lineWidth = 1;
    for (const kp of img.kps) {
      ctx.beginPath();
      ctx.arc(kp.x * scale, kp.y * scale, Math.max(1.5, kp.size * scale * 0.35), 0, 2 * Math.PI);
      ctx.stroke();
    }
    const cap = document.createElement('figcaption');
    cap.textContent = `${img.name} — ${img.w}×${img.h}, ${img.n} keypoints`;
    fig.appendChild(canvas);
    fig.appendChild(cap);
    stripEl.appendChild(fig);
  }
}

function renderTiming() {
  timingSec.hidden = false;
  imgTimingEl.textContent = '';
  pairTimingEl.textContent = '';

  const fmt = (v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1));

  let hdr = document.createElement('tr');
  for (const t of ['Image', 'Decode (ms)', 'Grayscale (ms)', 'ORB (ms)']) {
    const th = document.createElement('th');
    th.textContent = t;
    hdr.appendChild(th);
  }
  imgTimingEl.appendChild(hdr);

  let tDec = 0;
  let tGray = 0;
  let tOrb = 0;
  for (const tm of timings) {
    tDec += tm.decode;
    tGray += tm.gray;
    tOrb += tm.orb;
    const tr = document.createElement('tr');
    for (const cell of [tm.name, fmt(tm.decode), fmt(tm.gray), fmt(tm.orb)]) {
      const td = document.createElement('td');
      td.textContent = cell;
      tr.appendChild(td);
    }
    imgTimingEl.appendChild(tr);
  }
  const row = document.createElement('tr');
  for (const cell of ['Total', fmt(tDec), fmt(tGray), fmt(tOrb)]) {
    const td = document.createElement('td');
    td.style.fontWeight = '600';
    td.textContent = cell;
    row.appendChild(td);
  }
  imgTimingEl.appendChild(row);

  hdr = document.createElement('tr');
  for (const t of ['Pair', 'Matching scan (ms)', 'per scan (avg)']) {
    const th = document.createElement('th');
    th.textContent = t;
    hdr.appendChild(th);
  }
  pairTimingEl.appendChild(hdr);
  if (!pairs.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 3;
    td.textContent = 'Load at least 2 images to compare pairs.';
    tr.appendChild(td);
    pairTimingEl.appendChild(tr);
    return;
  }
  const MAX_ROWS = 10;
  let total = 0;
  for (let k = 0; k < pairs.length; k++) {
    total += pairMs[k];
    if (k >= MAX_ROWS) continue;
    const p = pairs[k];
    const tr = document.createElement('tr');
    for (const cell of [
      `${images[p.i].name} ↔ ${images[p.j].name}`,
      fmt(pairMs[k]),
      fmt(pairMs[k] / 2), // one scan per direction
    ]) {
      const td = document.createElement('td');
      td.textContent = cell;
      tr.appendChild(td);
    }
    pairTimingEl.appendChild(tr);
  }
  const sum = document.createElement('tr');
  for (const cell of [
    'Total / avg per pair',
    fmt(total),
    fmt(total / pairs.length),
  ]) {
    const td = document.createElement('td');
    td.style.fontWeight = '600';
    td.textContent = cell;
    sum.appendChild(td);
  }
  pairTimingEl.appendChild(sum);
}

function renderMatrix() {
  matrixSec.hidden = false;
  const n = images.length;
  const grid = countGrid(n, (i, j) => {
    const p = pairs.find((q) => q.i === i && q.j === j);
    return pairStats(p).good;
  });
  const max = Math.max(0, ...grid.flat());
  matrixEl.textContent = '';
  const head = document.createElement('tr');
  head.appendChild(document.createElement('th'));
  for (const img of images) {
    const th = document.createElement('th');
    th.textContent = img.name;
    head.appendChild(th);
  }
  matrixEl.appendChild(head);
  for (let i = 0; i < n; i++) {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.textContent = images[i].name;
    tr.appendChild(th);
    for (let j = 0; j < n; j++) {
      const td = document.createElement('td');
      if (i === j) {
        td.className = 'diag';
        td.textContent = '—';
      } else {
        const g = grid[i][j];
        td.style.background = max > 0 ? `rgba(34,139,34,${0.08 + 0.85 * (g / max)})` : '#fff';
        td.textContent = g;
        const pIdx = pairs.findIndex((q) => (q.i === i && q.j === j) || (q.i === j && q.j === i));
        if (pIdx === selected) td.classList.add('sel');
        td.title = `${images[i].name} vs ${images[j].name}: ${g} good matches`;
        td.addEventListener('click', () => selectPair(pIdx));
      }
      tr.appendChild(td);
    }
    matrixEl.appendChild(tr);
  }
}

function renderRanking() {
  rankSec.hidden = false;
  const T = +thrInput.value;
  const rows = pairs
    .map((p, idx) => ({ p, idx, s: pairStats(p) }))
    .filter(({ s }) => s.good >= 10);
  rows.sort((a, b) => b.s.good - a.s.good || a.s.mean - b.s.mean);
  rankEl.textContent = '';
  const head = document.createElement('tr');
  for (const t of ['#', 'Pair', 'Descriptors', 'Good matches', 'Mutual', 'Close cross-pairs', 'Mean best dist', 'Median best dist']) {
    const th = document.createElement('th');
    th.textContent = t;
    head.appendChild(th);
  }
  rankEl.appendChild(head);
  for (let r = 0; r < rows.length; r++) {
    const { p, idx, s } = rows[r];
    const tr = document.createElement('tr');
    if (idx === selected) tr.classList.add('sel');
    tr.title = `Click to compare ${images[p.i].name} vs ${images[p.j].name}`;
    for (const cell of [
      String(r + 1),
      `${images[p.i].name} ↔ ${images[p.j].name}`,
      `${images[p.i].n} / ${images[p.j].n}`,
      `${s.good}${r === 0 ? ' ★' : ''}`,
      String(s.mutual),
      String(s.close),
      s.mean.toFixed(1),
      s.median.toFixed(1),
    ]) {
      const td = document.createElement('td');
      td.textContent = cell;
      tr.appendChild(td);
    }
    tr.addEventListener('click', () => selectPair(idx));
    rankEl.appendChild(tr);
  }
}

function renderPair() {
  pairSec.hidden = false;
  const T = +thrInput.value;
  const p = pairs[selected];
  const a = images[p.i];
  const b = images[p.j];
  const s = pairStats(p);
  const maxH = 340;
  const maxW = 460;
  const sa = Math.min(1, maxW / a.w, maxH / a.h);
  const sb = Math.min(1, maxW / b.w, maxH / b.h);
  const gap = 24;
  const aw = Math.round(a.w * sa);
  const ah = Math.round(a.h * sa);
  const bw = Math.round(b.w * sb);
  const bh = Math.round(b.h * sb);
  const H = Math.max(ah, bh);
  const W = aw + gap + bw;
  pairViewEl.textContent = '';
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  pairViewEl.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(a.bmp, 0, 0, aw, ah);
  ctx.drawImage(b.bmp, aw + gap, Math.round((H - bh) / 2), bw, bh);
  // match lines, best matches <= T, capped for legibility
  const matches = [];
  for (let i = 0; i < a.n; i++) {
    const d = p.ab.bestD[i];
    if (d <= T) matches.push({ i, j: p.ab.bestIdx[i], d });
  }
  matches.sort((x, y) => x.d - y.d);
  for (const m of matches.slice(0, 150)) {
    const ax = a.kps[m.i].x * sa;
    const ay = a.kps[m.i].y * sa;
    const bx = aw + gap + b.kps[m.j].x * sb;
    const by = Math.round((H - bh) / 2) + b.kps[m.j].y * sb;
    ctx.strokeStyle = `hsla(${130 - 120 * Math.min(1, m.d / T)}, 90%, 40%, 0.55)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
  // keypoint circles on top
  for (const [img, scale, dx, dy] of [[a, sa, 0, 0], [b, sb, aw + gap, Math.round((H - bh) / 2)]]) {
    ctx.strokeStyle = '#ff2d2d';
    ctx.lineWidth = 1;
    for (const kp of img.kps) {
      ctx.beginPath();
      ctx.arc(kp.x * scale + dx, kp.y * scale + dy, Math.max(1.5, kp.size * scale * 0.3), 0, 2 * Math.PI);
      ctx.stroke();
    }
  }
  pairInfoEl.textContent =
    `${a.name} ↔ ${b.name}: ${s.good} good matches (dist ≤ ${T}), ${s.mutual} mutual, ` +
    `${s.close} close cross-pairs, mean best distance ${s.mean.toFixed(1)} (of 256). ` +
    `${matches.length > 150 ? 'Showing the 150 closest lines.' : ''}`;
}