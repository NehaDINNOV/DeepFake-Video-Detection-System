/* ── TruthLens main.js ────────────────────────────────────────────────── */

// ── State ──────────────────────────────────────────────────────────────
let selectedFile = null;
let nFrames      = 8;
let useLlava     = false;
let frameResults = [];

// ── DOM refs ───────────────────────────────────────────────────────────
const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const btnAnalyse  = document.getElementById('btn-analyse');
const btnText     = document.getElementById('btn-text');
const progressWrap = document.getElementById('progress-wrap');
const progressFill = document.getElementById('progress-fill');
const progressMsg  = document.getElementById('progress-msg');
const results      = document.getElementById('results');
const uploadCard   = document.getElementById('upload-card');

// ── Health check ───────────────────────────────────────────────────────
(async () => {
  const dot   = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  const d     = dot.querySelector('.dot');
  try {
    const r   = await fetch('/health');
    const j   = await r.json();
    if (j.status === 'ok') {
      d.className = 'dot ok';
      label.textContent = j.device === 'cuda' ? 'GPU ready' : 'CPU ready';
    }
  } catch {
    d.className = 'dot warn';
    label.textContent = 'offline';
  }
})();

// ── Drag-and-drop ──────────────────────────────────────────────────────
['dragenter','dragover'].forEach(e => {
  dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('drag-over'); });
});
['dragleave','drop'].forEach(e => {
  dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('drag-over'); });
});
dropZone.addEventListener('drop', ev => {
  const f = ev.dataTransfer.files[0];
  if (f) setFile(f);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});
dropZone.addEventListener('click', e => {
  if (e.target.tagName !== 'LABEL') fileInput.click();
});

function setFile(f) {
  selectedFile = f;
  // Remove old chip
  dropZone.querySelector('.file-chip')?.remove();
  const chip = document.createElement('div');
  chip.className = 'file-chip';
  chip.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <rect x="1" y="1" width="12" height="12" rx="2" stroke="currentColor" stroke-width="1.2"/>
      <path d="M4 7h6M4 9.5h4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
    </svg>
    <span>${f.name}</span>
    <span style="opacity:0.5">${(f.size / 1e6).toFixed(1)} MB</span>`;
  dropZone.appendChild(chip);
  btnAnalyse.disabled = false;
  btnText.textContent = 'Analyse';
}

// ── Frame stepper ──────────────────────────────────────────────────────
document.getElementById('dec-frames').addEventListener('click', () => {
  if (nFrames > 1) { nFrames--; document.getElementById('n-frames-val').textContent = nFrames; }
});
document.getElementById('inc-frames').addEventListener('click', () => {
  if (nFrames < 32) { nFrames++; document.getElementById('n-frames-val').textContent = nFrames; }
});

// ── LLaVA toggle ──────────────────────────────────────────────────────
document.getElementById('use-llava').addEventListener('change', function() {
  useLlava = this.checked;
  document.getElementById('llava-txt').textContent = useLlava ? 'On' : 'Off';
});

// ── Analyse ───────────────────────────────────────────────────────────
btnAnalyse.addEventListener('click', () => {
  if (!selectedFile) return;
  startAnalysis();
});

function startAnalysis() {
  // Reset UI
  frameResults = [];
  results.hidden = true;
  progressWrap.hidden = false;
  setProgress(0, 'Uploading...');
  btnAnalyse.disabled = true;
  btnText.textContent  = 'Analysing...';

  // Hide sub-panels
  document.getElementById('model-row').hidden       = true;
  document.getElementById('timeline-wrap').hidden   = true;
  document.getElementById('heatmap-panel').hidden   = true;
  document.getElementById('explanation-card').hidden= true;
  document.getElementById('timeline').innerHTML     = '';

  // Build FormData
  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('n_frames', nFrames);
  fd.append('use_llava', useLlava ? 'true' : 'false');

  // Stream SSE
  fetch('/analyse', { method: 'POST', body: fd })
    .then(resp => {
      const reader  = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer    = '';

      function read() {
        reader.read().then(({ done, value }) => {
          if (done) return;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop();
          lines.forEach(line => {
            if (line.startsWith('data: ')) {
              try { handleEvent(JSON.parse(line.slice(6))); }
              catch (e) { console.error('Parse error', e); }
            }
          });
          read();
        });
      }
      read();
    })
    .catch(err => {
      setProgress(0, `Error: ${err.message}`);
      btnAnalyse.disabled = false;
      btnText.textContent  = 'Retry';
    });
}

function handleEvent(ev) {
  switch (ev.type) {
    case 'progress':
      setProgress(ev.pct, ev.msg);
      break;

    case 'frame':
      setProgress(ev.pct, `Analysing frame ${ev.frame_idx + 1} / ${ev.total}...`);
      frameResults.push({ idx: ev.frame_idx, ...ev.result });
      addFrameChip(ev.frame_idx, ev.result);
      break;

    case 'done':
      setProgress(100, 'Done.');
      renderResults(ev);
      progressWrap.hidden = true;
      btnAnalyse.disabled = false;
      btnText.textContent  = 'Analyse again';
      break;

    case 'error':
      setProgress(0, `Error: ${ev.msg}`);
      btnAnalyse.disabled = false;
      btnText.textContent  = 'Retry';
      break;
  }
}

// ── Helpers ────────────────────────────────────────────────────────────
function setProgress(pct, msg) {
  progressFill.style.width = pct + '%';
  progressMsg.textContent  = msg;
}

function addFrameChip(idx, res) {
  const tl   = document.getElementById('timeline');
  const chip = document.createElement('div');
  chip.className = `frame-chip ${res.label === 'FAKE' ? 'chip-fake' : 'chip-real'}`;
  chip.dataset.idx = idx;
  chip.innerHTML = `
    <img class="frame-thumb" src="data:image/png;base64,${res.original_b64}" alt="Frame ${idx+1}">
    <span class="frame-label">Frame ${idx+1}</span>
    <span class="frame-prob ${res.label === 'FAKE' ? 'fake' : 'real'}">${res.fake_prob}%</span>`;
  chip.addEventListener('click', () => showFrameDetail(res));
  tl.appendChild(chip);
}

function renderResults(ev) {
  // Show results section
  results.hidden = false;
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const isFake = ev.verdict === 'FAKE';
  const prob   = ev.fake_prob;

  // Verdict card
  const vc = document.getElementById('verdict-card');
  vc.className = `verdict-card ${isFake ? 'is-fake' : 'is-real'}`;
  document.getElementById('verdict-label').textContent =
    isFake ? '⚠ DEEPFAKE DETECTED' : '✓ AUTHENTIC';
  const vp = document.getElementById('verdict-prob');
  vp.textContent = prob + '%';
  vp.className   = `verdict-prob ${isFake ? 'is-fake' : 'is-real'}`;
  document.getElementById('verdict-meta').textContent =
    `${ev.fake_votes} / ${ev.total_frames} frames flagged as fake`;

  // Gauge animation
  const arc    = document.getElementById('gauge-arc');
  const full   = 220;
  const offset = full - (prob / 100) * full;
  arc.style.stroke       = isFake ? 'var(--red)' : 'var(--green)';
  arc.style.strokeDashoffset = offset;

  // Model row — use last frame's per-model probs
  if (frameResults.length > 0) {
    const last = frameResults[frameResults.length - 1];
    const avgXcp = avg(frameResults.map(r => r.prob_xcp));
    const avgEff = avg(frameResults.map(r => r.prob_eff));
    document.getElementById('prob-xcp').textContent  = avgXcp.toFixed(1) + '%';
    document.getElementById('prob-eff').textContent  = avgEff.toFixed(1) + '%';
    document.getElementById('bar-xcp').style.width   = avgXcp + '%';
    document.getElementById('bar-eff').style.width   = avgEff + '%';
    document.getElementById('model-row').hidden = false;
  }

  // Timeline (only for multi-frame)
  if (ev.total_frames > 1) {
    document.getElementById('timeline-wrap').hidden = false;
  }

  // Heatmap — populate from best frame
  const best = ev.best_frame;
  document.getElementById('img-overlay').src  = 'data:image/png;base64,' + best.overlay_b64;
  document.getElementById('img-xcp').src      = 'data:image/png;base64,' + best.xcp_cam_b64;
  document.getElementById('img-eff').src      = 'data:image/png;base64,' + best.eff_cam_b64;
  document.getElementById('img-original').src = 'data:image/png;base64,' + best.original_b64;
  document.getElementById('heatmap-panel').hidden = false;

  // Explanation
  if (ev.explanation) {
    document.getElementById('expl-body').textContent = ev.explanation;
    document.getElementById('explanation-card').hidden = false;
  }
}

function showFrameDetail(res) {
  // Switch to the chosen frame in heatmap panel
  document.getElementById('img-overlay').src  = 'data:image/png;base64,' + res.overlay_b64;
  document.getElementById('img-xcp').src      = 'data:image/png;base64,' + res.xcp_cam_b64;
  document.getElementById('img-eff').src      = 'data:image/png;base64,' + res.eff_cam_b64;
  document.getElementById('img-original').src = 'data:image/png;base64,' + res.original_b64;
  document.getElementById('heatmap-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function avg(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// ── Heatmap tabs ───────────────────────────────────────────────────────
document.querySelectorAll('.htab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.htab').forEach(b => b.classList.remove('htab-active'));
    document.querySelectorAll('.htab-img').forEach(i => i.classList.remove('active'));
    btn.classList.add('htab-active');
    const tab = btn.dataset.tab;
    document.querySelectorAll(`.htab-img[data-tab="${tab}"]`).forEach(i => i.classList.add('active'));
  });
});
