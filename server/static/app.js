"use strict";

const $ = (id) => document.getElementById(id);
const log = (m) => { const el = $("log"); el.textContent += m + "\n"; el.scrollTop = el.scrollHeight; };

// ---- live-updating slider labels ----
const sliders = {
  num_step: "stepV", guidance_scale: "gsV", position_temperature: "ptV",
  class_temperature: "ctV", first_chunk_words: "fcV", max_words: "mwV",
};
for (const [id, out] of Object.entries(sliders)) {
  const s = $(id);
  const upd = () => { $(out).textContent = s.value; };
  s.addEventListener("input", upd); upd();
}

// ---- backend info ----
fetch("/api/info").then(r => r.json()).then(d => {
  $("beBadge").textContent = d.backend || d.configured_backend || "?";
  if (d.sampling_rate) $("srBadge").textContent = d.sampling_rate;
}).catch(() => {});

// ============================ AUDIO STREAMING ============================
let audioCtx = null;
let playCursor = 0;      // next scheduled start time (AudioContext clock)
let sampleRate = 24000;

function ensureAudio(sr) {
  sampleRate = sr || sampleRate;
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
  playCursor = audioCtx.currentTime + 0.05;
}

function schedulePCM(int16) {
  const n = int16.length;
  if (!n) return;
  const buf = audioCtx.createBuffer(1, n, sampleRate);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < n; i++) ch[i] = int16[i] / 32768;
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  const now = audioCtx.currentTime;
  if (playCursor < now) playCursor = now + 0.02;
  src.start(playCursor);
  playCursor += buf.duration;
}

// ============================ TTS WEBSOCKET ============================
let ws = null;
let pendingMeta = null;
let sumWall = 0, sumAudio = 0, peakVram = 0, chunkCount = 0;

function resetMetrics() {
  sumWall = 0; sumAudio = 0; peakVram = 0; chunkCount = 0;
  $("mTtfa").textContent = "…"; $("mRtf").textContent = "…";
  $("mAudio").textContent = "…"; $("mVram").textContent = "…";
  $("chunkTbl").querySelector("tbody").innerHTML = "";
  $("log").textContent = "";
}

function rtfClass(v) { return v < 0.5 ? "good" : (v < 1.0 ? "warnv" : "badv"); }

function startTTS() {
  const text = $("text").value.trim();
  if (!text) return;
  ensureAudio(sampleRate);
  resetMetrics();
  $("go").disabled = true; $("stop").disabled = false;
  $("status").textContent = "connecting…";

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/tts`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    $("status").textContent = "generating…";
    const seedRaw = $("seed").value;
    ws.send(JSON.stringify({
      text,
      language: $("language").value || null,
      instruct: $("instruct").value || null,
      num_step: parseInt($("num_step").value),
      guidance_scale: parseFloat($("guidance_scale").value),
      position_temperature: parseFloat($("position_temperature").value),
      class_temperature: parseFloat($("class_temperature").value),
      first_chunk_words: parseInt($("first_chunk_words").value),
      max_words: parseInt($("max_words").value),
      seed: seedRaw === "" ? null : parseInt(seedRaw),
    }));
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      const msg = JSON.parse(ev.data);
      if (msg.type === "start") {
        ensureAudio(msg.sample_rate);
        $("beBadge").textContent = msg.backend;
        $("srBadge").textContent = msg.sample_rate;
        log(`start · backend=${msg.backend} sr=${msg.sample_rate}`);
      } else if (msg.type === "chunk_meta") {
        pendingMeta = msg;
      } else if (msg.type === "done") {
        $("status").textContent = "done";
        finish();
      } else if (msg.type === "error") {
        $("status").textContent = "error";
        log("ERROR: " + msg.message);
        finish();
      }
    } else {
      // binary PCM16 for the last chunk_meta
      const int16 = new Int16Array(ev.data);
      schedulePCM(int16);
      if (pendingMeta) applyChunk(pendingMeta, int16.length);
      pendingMeta = null;
    }
  };

  ws.onclose = () => { $("go").disabled = false; $("stop").disabled = true; };
  ws.onerror = () => { $("status").textContent = "ws error"; };
}

function applyChunk(m, nSamples) {
  chunkCount++;
  sumWall += m.gen_wall_s;
  sumAudio += m.audio_dur_s;
  if (m.vram_alloc_mb) peakVram = Math.max(peakVram, m.vram_alloc_mb);

  if (chunkCount === 1 && m.ttfa_ms != null) $("mTtfa").textContent = m.ttfa_ms.toFixed(0);
  const rtf = sumAudio > 0 ? (sumWall / sumAudio) : 0;
  const rtfEl = $("mRtf");
  rtfEl.textContent = rtf.toFixed(3);
  rtfEl.className = "v " + rtfClass(rtf);
  $("mAudio").textContent = sumAudio.toFixed(2);
  $("mVram").textContent = peakVram ? peakVram.toFixed(0) : "n/a";

  const tb = $("chunkTbl").querySelector("tbody");
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${m.index}</td><td>${m.text}</td>
    <td>${m.gen_wall_s.toFixed(3)}</td><td>${m.audio_dur_s.toFixed(3)}</td>
    <td class="${rtfClass(m.rtf)}">${m.rtf.toFixed(3)}</td>`;
  tb.appendChild(tr);
}

function finish() {
  $("go").disabled = false; $("stop").disabled = true;
  if (ws) { try { ws.close(); } catch (e) {} }
}

$("go").onclick = startTTS;
$("stop").onclick = () => { $("status").textContent = "stopped"; finish(); };

// ============================ TELEMETRY ============================
function mkSpark(canvasId) {
  const c = $(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const resize = () => { c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr; };
  resize(); window.addEventListener("resize", resize);
  const data = [];
  const N = 120;
  return (val, color) => {
    if (val == null || isNaN(val)) return;
    data.push(val); if (data.length > N) data.shift();
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    const mn = Math.min(...data), mx = Math.max(...data);
    const rng = (mx - mn) || 1;
    ctx.beginPath();
    ctx.lineWidth = 2 * dpr;
    ctx.strokeStyle = color;
    data.forEach((v, i) => {
      const x = (i / (N - 1)) * c.width;
      const y = c.height - ((v - mn) / rng) * (c.height - 6 * dpr) - 3 * dpr;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  };
}
const spRam = mkSpark("cRam"), spUtil = mkSpark("cUtil"), spGpuP = mkSpark("cGpuP");
const spVin = mkSpark("cVin"), spGpuT = mkSpark("cGpuT"), spCpuT = mkSpark("cCpuT");

function connectTelemetry() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const t = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
  t.onopen = () => { $("teleDot").classList.add("on"); $("teleStat").textContent = "telemetry live"; };
  t.onclose = () => {
    $("teleDot").classList.remove("on"); $("teleStat").textContent = "telemetry off";
    setTimeout(connectTelemetry, 1500);
  };
  t.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    if (d.ram_used_mb != null) {
      const gb = d.ram_used_mb / 1024, tot = (d.ram_total_mb || 0) / 1024;
      $("tRam").textContent = `${gb.toFixed(1)} / ${tot.toFixed(0)} GB`;
      spRam(d.ram_used_mb, "#4da3ff");
    }
    if (d.gpu_util_pct != null) { $("tUtil").textContent = d.gpu_util_pct.toFixed(0) + " %"; spUtil(d.gpu_util_pct, "#37d39b"); }
    if (d.gpu_power_mw != null) { $("tGpuP").textContent = (d.gpu_power_mw/1000).toFixed(1) + " W"; spGpuP(d.gpu_power_mw, "#ffb454"); }
    if (d.board_power_mw != null) { $("tVin").textContent = (d.board_power_mw/1000).toFixed(1) + " W"; spVin(d.board_power_mw, "#ff6b6b"); }
    if (d.gpu_temp_c != null) { $("tGpuT").textContent = d.gpu_temp_c.toFixed(0) + " °C"; spGpuT(d.gpu_temp_c, "#c58bff"); }
    if (d.cpu_temp_c != null) { $("tCpuT").textContent = d.cpu_temp_c.toFixed(0) + " °C"; spCpuT(d.cpu_temp_c, "#8ca0bd"); }
  };
}
connectTelemetry();
