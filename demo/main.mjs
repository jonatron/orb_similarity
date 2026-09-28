// Browser demo: ImageDecoder → grayscale → OpenCV.js ORB → overlay keypoints.
import { loadOrb } from './orb-opencv.mjs';

const canvas = document.getElementById('out');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const descPre = document.getElementById('desc');
const fileInput = document.getElementById('file');

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

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;
  status.textContent = 'Decoding with ImageDecoder …';
  try {
    const buf = await file.arrayBuffer();
    const dec = new ImageDecoder({ data: buf, type: file.type });
    await dec.tracks.ready;
    const { image } = await dec.decode();
    await image.prepareForRendering?.();
    const bmp = await createImageBitmap(image);
    const { gray, w, h } = rgbaToGray(bmp);

    const orbw = await loadOrb();
    orbw.setup(w, h);
    // ORB_create defaults: scaleFactor 1.2, 8 levels, Harris score, patch 31.
    const sf = Math.fround(1.2);
    for (let l = 0; l < 8; l++) orbw.setLayerScale(l, Math.fround(Math.pow(sf, l)));
    orbw.setNDPow(Math.fround(Math.pow(Math.fround(1 / sf), 8)));

    const input = new Uint8Array(orbw.memory.buffer, orbw.inputPtr(), gray.length);
    input.set(gray);

    const t0 = performance.now();
    const n = orbw.run();
    const ms = (performance.now() - t0).toFixed(1);

    const scale = Math.min(canvas.width / w, canvas.height / h);
    const ow = Math.round(w * scale);
    const oh = Math.round(h * scale);
    canvas.width = ow;
    canvas.height = oh;
    ctx.drawImage(bmp, 0, 0, ow, oh);

    ctx.strokeStyle = '#ff2d2d';
    ctx.lineWidth = Math.max(1, scale * 1.2);
    for (let i = 0; i < n; i++) {
      const x = orbw.kpX(i) * scale;
      const y = orbw.kpY(i) * scale;
      const sz = orbw.kpSize(i) * scale * 0.5;
      const a = (orbw.kpAngle(i) * Math.PI) / 180;
      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, sz), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.cos(a) * Math.max(3, sz), y + Math.sin(a) * Math.max(3, sz));
      ctx.stroke();
    }

    const desc = new Uint8Array(orbw.memory.buffer, orbw.descPtr(), orbw.descLen());
    let rows = '';
    for (let r = 0; r < Math.min(4, n); r++) {
      rows += Array.from(desc.slice(r * 32, (r + 1) * 32))
        .map((b) => b.toString(16).padStart(2, '0')).join(' ') + '\n';
    }
    descPre.textContent = rows;
    status.textContent = `OK — ${n} keypoints in ${ms} ms (pipeline only; decode not counted).`;
  } catch (e) {
    console.error(e);
    status.textContent = 'Error: ' + e.message;
  }
});