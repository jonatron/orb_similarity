// Node verification harness, no DOM. Two parts:
//   1. popcnt.wasm (built from src/popcnt.ts) must produce results
//      bit-identical to the JS scanPair fallback, including edge cases.
//   2. The OpenCV.js ORB adapter (demo/orb-opencv.mjs, node branch of the
//      embedded opencv.node.cjs build) must return the descriptor layout the
//      browser demos consume.
// Run with: npm run verify  (or: node verify.mjs)

import { readFile } from 'node:fs/promises';
import { scanPair as scanPairJS, makeWasmScanPair } from './demo/similarity-core.mjs';
import { loadOrb } from './demo/orb-opencv.mjs';

let failures = 0;
const check = (cond, label, extra = '') => {
  if (cond) console.log(`  ok  ${label}${extra ? ` — ${extra}` : ''}`);
  else {
    failures++;
    console.error(`FAIL  ${label}${extra ? ` — ${extra}` : ''}`);
  }
};

// ---------- 1. popcnt.wasm vs JS scanPair ----------
console.log('popcnt.wasm vs JS scanPair (must be bit-identical):');
const wasmBytes = await readFile(new URL('./build/popcnt.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(wasmBytes, {
  env: { abort() { throw new Error('wasm abort'); }, trace() {}, seed: () => 123 },
});
const wasmScan = makeWasmScanPair(instance.exports);

let s = 0x2f6e2b1; // deterministic LCG so failures are reproducible
const rnd = (n) => ((s = (s * 1664525 + 1013904223) >>> 0), s % n);

const randDesc = (rows) => {
  const a = new Uint8Array(rows * 32);
  for (let i = 0; i < a.length; i++) a[i] = rnd(256);
  return a;
};

const cases = [
  ['small 5x3', randDesc(5), randDesc(3)],
  ['single 1x1', randDesc(1), randDesc(1)],
  ['zero targets', randDesc(4), new Uint8Array(0)],
  ['zero sources', new Uint8Array(0), randDesc(4)],
  ['large 200x150', randDesc(200), randDesc(150)],
  ['bit-complement rows', randDesc(3), (() => { const t = randDesc(3); const c = new Uint8Array(t.length); for (let i = 0; i < c.length; i++) c[i] = ~t[i]; return c; })()],
];
for (const [label, src, tgt] of cases) {
  let js, wm, threwJS = null, threwWasm = null;
  try { js = scanPairJS(src, tgt); } catch (e) { threwJS = e; }
  try { wm = wasmScan(src, tgt); } catch (e) { threwWasm = e; }
  if (threwJS || threwWasm) {
    check(false, label, `js threw: ${threwJS} | wasm threw: ${threwWasm}`);
    continue;
  }
  const same =
    js.bestD.length === wm.bestD.length &&
    js.bestD.every((v, i) => v === wm.bestD[i]) &&
    js.bestIdx.length === wm.bestIdx.length &&
    js.bestIdx.every((v, i) => v === wm.bestIdx[i]) &&
    js.hist.length === wm.hist.length &&
    js.hist.every((v, i) => v === wm.hist[i]);
  check(same, label, `nS=${src.length / 32} nT=${tgt.length / 32}`);
  if (!same) {
    for (let i = 0; i < js.hist.length; i++) {
      if (js.hist[i] !== wm.hist[i]) console.error(`     hist[${i}] js=${js.hist[i]} wasm=${wm.hist[i]}`);
    }
  }
}

{
  // a single exact bit-complement pair must land in hist[256] and keep bestD=255
  const tgt = randDesc(1);
  const src = new Uint8Array(32);
  for (let i = 0; i < 32; i++) src[i] = ~tgt[i];
  const wm = wasmScan(src, tgt);
  check(wm.hist[256] === 1, 'complement pair lands in hist[256]', `hist[256]=${wm.hist[256]}`);
  check(wm.bestD[0] === 255, 'complement pair keeps sentinel bestD=255', `bestD=${wm.bestD[0]}`);
}

{
  let threw = false;
  try { wasmScan(randDesc(2048), randDesc(1)); } catch { threw = true; }
  check(threw, 'row-cap overflow raises', 'nS+nT > 2048');
}

// ---------- 2. OpenCV.js ORB adapter (node branch) ----------
console.log('OpenCV.js ORB adapter:');
const orb = await loadOrb();
const w = 320, h = 240;
const gray = new Uint8Array(w * h);
for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) gray[y * w + x] = (x * 7 + y * 13) % 256;
orb.setup(w, h);
new Uint8Array(orb.memory.buffer, orb.inputPtr(), gray.length).set(gray);
const n = orb.run();
check(n > 0, 'synthetic gray image yields keypoints', `n=${n}`);
check(orb.descLen() === n * 32, 'descriptors are 32-byte rows', `descLen=${orb.descLen()}`);

if (orb.descLen()) {
  const d1 = new Uint8Array(orb.memory.buffer, orb.descPtr(), orb.descLen()).slice();
  const n2 = orb.run();
  const d2 = new Uint8Array(orb.memory.buffer, orb.descPtr(), orb.descLen()).slice();
  check(n2 === n, 'stable keypoint count across runs', `run1=${n} run2=${n2}`);
  check(d1.length === d2.length && d1.every((v, i) => v === d2[i]), 'deterministic descriptor bytes across runs');
}

orb.setNFeatures(50);
new Uint8Array(orb.memory.buffer, orb.inputPtr(), gray.length).set(gray);
const n50 = orb.run();
check(n50 > 0 && n50 <= 50, 'setNFeatures(50) caps keypoints', `n=${n50}`);
check(orb.descLen() === n50 * 32, 'descriptor layout intact after detector re-create');
check(
  [orb.kpX(0), orb.kpY(0), orb.kpSize(0), orb.kpAngle(0)].every((v) => typeof v === 'number'),
  'kp accessors return numbers',
);

console.log(failures === 0 ? '\nAll checks passed.' : `\n${failures} check(s) FAILED`);
process.exitCode = failures === 0 ? 0 : 1;