// Browser verification driver: launches headless Chrome against
// verify-browser.html and reads the run outcome (written into the page title
// and #result) back over CDP. Prerequisite: `npm run serve` on port 8090.
// Run with: npm run verify:browser

import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const PORT = 9223;
const URL = 'http://localhost:8090/verify-browser.html';

const chrome = spawn('google-chrome', [
  '--headless=new',
  '--no-sandbox',
  '--disable-gpu',
  '--remote-debugging-port=' + PORT,
  `--user-data-dir=${mkdtempSync(join(tmpdir(), 'orb3verify-'))}`,
  URL,
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  let target;
  for (let i = 0; i < 100; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      const targets = await res.json();
      target = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
      if (target) break;
    } catch {}
    await sleep(200);
  }
  if (!target) throw new Error('chrome CDP endpoint not reachable');

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    ws.onopen = res;
    ws.onerror = () => rej(new Error('CDP websocket failed'));
  });

  let id = 0;
  const pending = new Map();
  const consoleErrors = [];
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) {
      const { res, rej } = pending.get(m.id);
      pending.delete(m.id);
      if (m.error) rej(new Error(m.error.message));
      else res(m.result);
    } else if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
      consoleErrors.push(m.params.args?.map((a) => a.value ?? a.description ?? '').join(' ') ?? '');
    }
  };
  const send = (method, params = {}) =>
    new Promise((res, rej) => {
      const i = ++id;
      pending.set(i, { res, rej });
      ws.send(JSON.stringify({ id: i, method, params }));
    });

  await send('Runtime.enable');

  let title = '';
  for (let i = 0; i < 240; i++) {
    const { result } = await send('Runtime.evaluate', { expression: 'document.title', returnByValue: true });
    title = result?.value ?? '';
    if (title.startsWith('ORBVERIFY')) break;
    await sleep(500);
  }

  const { result } = await send('Runtime.evaluate', {
    expression: 'document.getElementById("result")?.textContent ?? "(no result element)"',
    returnByValue: true,
  });
  console.log(result?.value ?? '(empty)');
  console.log(`\ntitle: ${title}`);

  if (consoleErrors.length) {
    console.log('\nconsole errors:');
    for (const e of consoleErrors.slice(0, 10)) console.log(`  ${e}`);
  }

  const ok = title.startsWith('ORBVERIFY') && !/ERROR/.test(title);
  process.exitCode = ok ? 0 : 1;
  ws.close();
}

try {
  await main();
} finally {
  chrome.kill('SIGKILL');
  try { rmSync(chrome.spawnargs.find((a) => a.startsWith('--user-data-dir=')).split('=')[1], { recursive: true, force: true }); } catch {}
}