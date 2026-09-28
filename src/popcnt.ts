// popcnt.ts — brute-force Hamming matcher for ORB descriptors (32-byte rows).
// The all-pairs scan runs inside the wasm so the inner loop uses the native
// i64.popcnt instruction over 8-byte words; the host only stages the two descriptor buffers and
// reads back per-row best distances, best-match indices, and the cross-pair
// distance histogram. Kept as a separate module so build/orb.wasm stays
// byte-identical to the verified port.

const ROW: i32 = 32; // bytes per BRIEF descriptor
const CAP: i32 = 2048; // max descriptor rows staged in buf (ORB default is 500)

const buf = new Uint8Array(CAP * ROW);
const bestD = new Uint8Array(CAP);
const bestIdx = new Uint32Array(CAP);
const hist = new Uint32Array(257);

export function cap(): i32 {
  return CAP;
}

export function bufPtr(): usize {
  return buf.dataStart;
}

export function bestDPtr(): usize {
  return bestD.dataStart;
}

export function bestIdxPtr(): usize {
  return bestIdx.dataStart;
}

export function histPtr(): usize {
  return hist.dataStart;
}

// All-pairs Hamming scan. The caller stages the source rows at byte offset 0
// and the target rows right after (buf[0 .. nS*32) and buf[nS*32 .. (nS+nT)*32)),
// then reads bestD/bestIdx/hist. bestD stores min(distance, 255) so a u8 cell
// can never wrap on the 256 sentinel (distances above 255 can only occur when
// every target row is the exact bit-complement of the source row).
export function scan(nS: i32, nT: i32): void {
  const base = buf.dataStart;
  for (let d = 0; d < 257; d++) hist[d] = 0;
  for (let i = 0; i < nS; i++) {
    const aOff: usize = <usize>(i * ROW);
    let best = 255;
    let bestJ = 0;
    for (let j = 0; j < nT; j++) {
      const bOff: usize = <usize>((nS + j) * ROW);
      let d = 0;
      for (let k = 0; k < ROW; k += 8) {
        const x = load<u64>(base + aOff + <usize>k);
        const y = load<u64>(base + bOff + <usize>k);
        d += <i32>popcnt<u64>(x ^ y);
      }
      hist[d]++;
      if (d < best) {
        best = d;
        bestJ = j;
      }
    }
    bestD[i] = best as u8;
    bestIdx[i] = bestJ as u32;
  }
}