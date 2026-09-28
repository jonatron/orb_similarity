// ORB engine adapter backed by OpenCV.js, used in place of the AssemblyScript
// orb.wasm build (which this project does not ship). Exposes the same surface
// the demos consume — setup/inputPtr/run/kpX/kpY/kpSize/kpAngle/descPtr/
// descLen/memory plus setNFeatures — so swapping engines is a one-line import.
//
// Browser: injects vendor/opencv.js — the repo's embedded single-file OpenCV
// build (wasm inlined, no opencv.wasm fetch) with the environment flag flipped
// to run outside Node; the npm package's opencv.js only wires its sync wasm
// loader for web workers. Node: loads the unpatched root opencv.node.cjs.
//
// Input is staged through a JS-owned staging ArrayBuffer, not the wasm heap:
// OpenCV.js does not expose heap views on its module object, and the wasm heap
// can grow mid-run, which would invalidate buffers the demos captured earlier.
// run() copies staged grayscale into cv.Mat, runs cv.ORB with the reference
// defaults (scaleFactor 1.2, 8 levels, Harris, patch 31), and copies the
// descriptors (32-byte rows) back into staging behind the input area.

const ROW = 32;
const MAX_DESC_ROWS = 2048; // capability ceiling: ORB output is capped by nfeatures
// const ORB_PARAMS = [1.2, 8, 31, 0, 2, 0 /* HARRIS_SCORE */, 31, 20]; // ORB_create defaults
const ORB_PARAMS = [2, 4, 31, 0, 2, 0 /* HARRIS_SCORE */, 31, 20]; // adjust 

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(s);
  });
}

let cvPromise = null;

async function ensureCv() {
  if (cvPromise) return cvPromise;
  cvPromise = (async () => {
    let cv;
    if (typeof window === 'undefined') {
      const { createRequire } = await import('node:module');
      const require = createRequire(import.meta.url);
      const loaded = require('../opencv.node.cjs');
      cv = typeof loaded?.then === 'function' ? await loaded : loaded;
    } else {
      if (!window.cv) {
        await loadScript(new URL('../vendor/opencv.js', import.meta.url).href);
      }
      const raw = window.cv;
      let mod = raw && typeof raw.then === 'function' ? await raw : raw;
      // Some builds swap window.cv to the initialized module only after
      // onRuntimeInitialized; poll briefly instead of racing the hook.
      const t0 = Date.now();
      while (!mod || !mod.Mat) {
        if (Date.now() - t0 > 30000) throw new Error('OpenCV.js initialization timed out');
        await new Promise((r) => setTimeout(r, 100));
        const fresh = window.cv;
        if (fresh && fresh !== raw) mod = typeof fresh.then === 'function' ? await fresh : fresh;
      }
      cv = mod;
    }
    if (!cv || !cv.Mat || (!cv.ORB && !cv.ORB_create)) {
      throw new Error('OpenCV.js loaded but the ORB binding is missing');
    }
    return cv;
  })();
  return cvPromise;
}

function createOrb(cv, nfeatures) {
  let orb;
  if (typeof cv.ORB_create === 'function') {
    orb = cv.ORB_create(nfeatures, ...ORB_PARAMS);
  } else {
    orb = new cv.ORB(nfeatures, ...ORB_PARAMS);
  }
  return orb;
}

export async function loadOrb() {
  const cv = await ensureCv();
  let orb = null;
  let w = 0;
  let h = 0;
  let staging = new Uint8Array(0); // input at [0, w*h), descriptors at [w*h, w*h + n*32)
  let kps = [];
  let nfeatures = 500;
  let descLen = 0;

  const api = {
    // WebAssembly.Memory-shaped shim: the demos read `orbw.memory.buffer` at
    // each use site; the getter resolves to the current staging buffer, which
    // is fixed per setup() and never replaced between staging and the
    // descriptor read, so captured views stay valid.
    get memory() {
      return { get buffer() { return staging.buffer; } };
    },
    setup(nw, nh) {
      w = nw;
      h = nh;
      staging = new Uint8Array(w * h + MAX_DESC_ROWS * ROW);
    },
    inputPtr: () => 0,
    setLayerScale() {}, // ORB_create params are fixed to the reference defaults
    setNDPow() {},
    setNFeatures(n) {
      if (n !== nfeatures) {
        nfeatures = n;
        orb?.delete();
        orb = null;
      }
    },
    run() {
      if (!orb) orb = createOrb(cv, nfeatures);
      const gray = new cv.Mat(h, w, cv.CV_8UC1);
      gray.data.set(staging.subarray(0, w * h));
      const kpVec = new cv.KeyPointVector();
      const desc = new cv.Mat();
      try {
        orb.detectAndCompute(gray, new cv.Mat(), kpVec, desc);
      } catch (err) {
        const msg =
          typeof err === 'number' && cv.exceptionFromPtr
            ? cv.exceptionFromPtr(err).msg
            : err instanceof Error
              ? err.message
              : String(err);
        throw new Error(`OpenCV ORB: ${msg}`);
      } finally {
        gray.delete();
      }
      const n = kpVec.size();
      kps = new Array(n);
      for (let i = 0; i < n; i++) {
        const kp = kpVec.get(i);
        kps[i] = { x: kp.pt.x, y: kp.pt.y, size: kp.size, angle: kp.angle };
      }
      descLen = desc.cols * desc.rows;
      if (descLen > 0) staging.set(desc.data.subarray(0, descLen), w * h);
      desc.delete();
      kpVec.delete();
      return n;
    },
    kpX: (i) => kps[i].x,
    kpY: (i) => kps[i].y,
    kpSize: (i) => kps[i].size,
    kpAngle: (i) => kps[i].angle,
    descPtr: () => w * h,
    descLen: () => descLen,
  };
  return api;
}