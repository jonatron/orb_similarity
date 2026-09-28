// Descriptor-matching logic shared by the similarity browser demo and the
// Node test harness. No DOM. The all-pairs scan runs either in JS (fallback)
// or, when the host supplies the popcnt.wasm exports via makeWasmScanPair(),
// inside the wasm module with the native i32.popcnt instruction. Input is raw
// 32-byte descriptor rows.

// Popcount for 0..255.
export const POP8 = (() => {
  const t = new Uint8Array(256);
  for (let i = 1; i < 256; i++) t[i] = (i & 1) + t[i >> 1];
  return t;
})();

// Hamming distance between two 32-byte descriptor rows.
export function hamming(a, aOff, b, bOff) {
  let d = 0;
  for (let k = 0; k < 32; k++) d += POP8[a[aOff + k] ^ b[bOff + k]];
  return d;
}

// For every descriptor row of src, find its best (min-hamming) match in tgt.
// Returns per-row best distance, the best match's row index, and a histogram
// of every cross-pair distance (buckets 0..256) so counts for any cutoff can
// be derived later without rescanning.
export function scanPair(src, tgt) {
  const nS = src.length / 32;
  const nT = tgt.length / 32;
  const bestD = new Uint8Array(nS);
  const bestIdx = new Uint32Array(nS);
  const hist = new Uint32Array(257);
  for (let i = 0; i < nS; i++) {
    const aOff = i * 32;
    // sentinel 255: bestD is u8, and a true 256 (impossible except when every
    // target row is the bit-complement of this row) is indistinguishable from
    // ">= 255" — match the wasm scan's clamping so both paths agree bit-for-bit
    let best = 255;
    let bi = 0;
    for (let j = 0; j < nT; j++) {
      const d = hamming(src, aOff, tgt, j * 32);
      hist[d]++;
      if (d < best) {
        best = d;
        bi = j;
      }
    }
    bestD[i] = best;
    bestIdx[i] = bi;
  }
  return { bestD, bestIdx, hist };
}

// Wrap popcnt.wasm's exports (cap/bufPtr/bestDPtr/bestIdxPtr/histPtr/scan)
// as a scanPair(src, tgt) with the same result shape as the JS fallback.
// The all-pairs loop runs entirely inside the wasm module — the host only
// stages the two buffers and reads the results back.
export function makeWasmScanPair(exports) {
  const { cap, bufPtr, bestDPtr, bestIdxPtr, histPtr, scan } = exports;
  return (src, tgt) => {
    const nS = src.length / 32;
    const nT = tgt.length / 32;
    const c = cap();
    if (nS + nT > c) {
      throw new RangeError(`scanPair: ${nS}+${nT} rows exceed wasm cap ${c}`);
    }
    const mem = exports.memory.buffer;
    const buf = new Uint8Array(mem, bufPtr(), (nS + nT) * 32);
    buf.set(src);
    buf.set(tgt, nS * 32);
    scan(nS, nT);
    return {
      bestD: new Uint8Array(mem, bestDPtr(), nS).slice(),
      bestIdx: new Uint32Array(mem, bestIdxPtr(), nS).slice(),
      hist: new Uint32Array(mem, histPtr(), 257).slice(),
    };
  };
}

// Derive per-pair statistics for cutoff T (distances <= T count as "good").
// `ab` is scanPair(A, B), `ba` scanPair(B, A).
export function summarize(ab, ba, T) {
  const nA = ab.bestD.length;
  let good = 0;
  let mutual = 0;
  for (let i = 0; i < nA; i++) {
    if (ab.bestD[i] <= T) {
      good++;
      const j = ab.bestIdx[i];
      if (ba.bestD[j] <= T && ba.bestIdx[j] === i) mutual++;
    }
  }
  let close = 0;
  for (let d = 0; d <= T; d++) close += ab.hist[d];
  const sorted = ab.bestD.slice().sort((x, y) => x - y);
  let mean = 0;
  for (const d of sorted) mean += d;
  mean /= nA;
  const mid = nA >> 1;
  const median = nA % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  return { good, mutual, close, mean, median };
}

// Build the symmetric N×N "good matches" grid for the demo matrix. The
// diagonal is -1 (an image vs itself); cell (i, j) mirrors cell (j, i).
export function countGrid(n, pairFn) {
  const grid = Array.from({ length: n }, () => new Array(n).fill(-1));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      grid[i][j] = grid[j][i] = pairFn(i, j);
    }
  }
  return grid;
}