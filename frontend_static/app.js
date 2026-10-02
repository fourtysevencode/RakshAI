// RakshAI frontend (vanilla JS). The selected video never leaves this tab except for
// the one-time upload; the review player, overlay and worker thumbnails all use the
// local file, so the server never has to keep footage.
'use strict';

// Backend origin: empty when served by the backend itself; config.js sets it for a
// separately hosted frontend (e.g. Vercel -> Hugging Face Space).
const API = (window.RAKSHAI_API_ORIGIN || '').replace(/\/$/, '') + '/api/v1';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

const TYPES = {
  NO_HARDHAT: { label: 'No hardhat', color: '#E4572E' },
  NO_VEST: { label: 'No safety vest', color: '#E0A100' },
  NO_MASK: { label: 'No mask', color: '#9B7FE0' },
};
const ITEM_SHORT = { hardhat: 'Hardhat', vest: 'Vest', mask: 'Mask' };
const ITEM_STATUS = {
  violation: { label: 'Missing', cls: 'text-bad border-bad/40 bg-bad/10' },
  brief: { label: 'Brief lapse', cls: 'text-warn border-warn/40 bg-warn/10' },
  ok: { label: 'OK', cls: 'text-ok border-ok/40 bg-ok/10' },
  unseen: { label: 'Not seen', cls: 'text-fog-400 border-ink-600 bg-ink-800' },
};
const SEVERITY = {
  high: { label: 'High', cls: 'text-bad bg-bad/10' },
  medium: { label: 'Medium', cls: 'text-warn bg-warn/10' },
  low: { label: 'Low', cls: 'text-yellow-200 bg-yellow-200/10' },
  none: { label: 'None', cls: 'text-ok bg-ok/10' },
};
const WORKER_STATUS = {
  non_compliant: 'Violations',
  compliant: 'Compliant',
  unconfirmed: 'Unconfirmed',
};
const SECONDS_PER_FRAME = 0.2; // rough CPU inference cost, for the time estimate
const LAST_KEY = 'rakshai.lastAnalysis';

const state = {
  file: null,
  fileUrl: null,
  videoMeta: null,
  analysisId: null,
  xhr: null,
  pollTimer: null,
  results: null,
  persons: [],
  events: [],
  tracks: {},
  workerFilter: 'all',
  evSort: { key: 't_start', dir: 1 },
  thumbs: {},
  lastFocus: null,
};

// ---------------------------------------------------------------- utilities
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* storage unavailable */ } },
};

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function fmtTs(s) {
  if (s == null || isNaN(s)) return '–';
  s = Math.max(0, s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}:${String(m).padStart(2, '0')}:${String(Math.floor(sec)).padStart(2, '0')}`;
  return `${m}:${sec.toFixed(1).padStart(4, '0')}`;
}

function fmtDur(s) {
  if (s == null || isNaN(s)) return '–';
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  if (m < 60) return `${m} min ${String(sec).padStart(2, '0')} s`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, '0')} min`;
}

const fmtPct = (v) => (v == null ? '–' : `${Math.round(v * 100)}%`);
const fmtBytes = (b) => (b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`);

async function api(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    let detail = `Request failed (${r.status})`;
    try { const j = await r.json(); if (j.detail) detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch { /* not json */ }
    const err = new Error(detail); err.status = r.status; throw err;
  }
  return r.json();
}

function show(view) {
  for (const id of ['setupView', 'progressView', 'resultsView']) $('#' + id).classList.toggle('hidden', id !== view);
  window.scrollTo({ top: 0 });
}

// ---------------------------------------------------------------- health
async function loadHealth() {
  const el = $('#modelStatus');
  try {
    const h = await api('/health');
    const ok = h.model.loaded;
    el.innerHTML = `<span class="w-2 h-2 rounded-full ${ok ? 'bg-ok' : 'bg-bad'}"></span>` +
      `<span>${ok ? `Model ready · ${h.model.classes.length} classes` : 'Model unavailable'}</span>`;
    el.title = ok ? h.model.classes.join(', ') : (h.model.error || '');
  } catch {
    el.innerHTML = '<span class="w-2 h-2 rounded-full bg-bad"></span><span>Server unreachable</span>';
  }
}

// ---------------------------------------------------------------- setup form
function setFile(file) {
  if (!file) return;
  if (file.type && !file.type.startsWith('video/')) {
    showFormError('That file is not a video.');
    return;
  }
  showFormError('');
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.file = file;
  state.fileUrl = URL.createObjectURL(file);
  state.videoMeta = null;
  const v = $('#setupPreview');
  v.src = state.fileUrl;
  $('#dropzone').classList.add('hidden');
  $('#previewWrap').classList.remove('hidden');
  $('#fileMeta').textContent = `${file.name} · ${fmtBytes(file.size)}`;
  v.onloadedmetadata = () => {
    state.videoMeta = { duration: v.duration, width: v.videoWidth, height: v.videoHeight };
    $('#fileMeta').textContent = `${file.name} · ${fmtBytes(file.size)} · ${fmtDur(v.duration)} · ${v.videoWidth}×${v.videoHeight}`;
    updateFpsHint();
  };
  v.onerror = () => {
    $('#fileMeta').textContent = `${file.name} · ${fmtBytes(file.size)} · preview not supported in this browser (analysis still works)`;
  };
}

function selectedFps() { return Number($('input[name=fps]:checked').value); }
function requiredPpe() { return $$('#ppeChips .chip').filter((c) => c.getAttribute('aria-pressed') === 'true').map((c) => c.dataset.item); }

function updateFpsHint() {
  const fps = selectedFps();
  const d = state.videoMeta?.duration;
  let text = fps ? `${fps} frames per second of video.` : 'Every frame (slowest, most detail).';
  if (d && isFinite(d)) {
    const frames = Math.round(d * (fps || 30));
    const secs = frames * SECONDS_PER_FRAME;
    text += ` About ${frames.toLocaleString()} frames, roughly ${fmtDur(secs)} to analyse.`;
  }
  $('#fpsHint').textContent = text;
}

function showFormError(msg) {
  const el = $('#formError');
  el.textContent = msg;
  el.classList.toggle('hidden', !msg);
}

function initSetup() {
  const dz = $('#dropzone'), input = $('#videoInput');
  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag-over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', (e) => { e.preventDefault(); dz.classList.remove('drag-over'); setFile(e.dataTransfer.files?.[0]); });
  input.addEventListener('change', () => setFile(input.files?.[0]));
  $('#changeVideo').addEventListener('click', () => input.click());

  $$('#ppeChips .chip').forEach((chip) => chip.addEventListener('click', () => {
    chip.setAttribute('aria-pressed', chip.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
  }));
  $$('input[name=fps]').forEach((r) => r.addEventListener('change', updateFpsHint));
  $('#conf').addEventListener('input', (e) => { $('#confVal').textContent = Number(e.target.value).toFixed(2); });
  $('#minViol').addEventListener('input', (e) => { $('#minViolVal').textContent = `${Number(e.target.value).toFixed(1)} s`; });
  $('#analysisForm').addEventListener('submit', (e) => { e.preventDefault(); startAnalysis(); });
}

// ---------------------------------------------------------------- submit + progress
const STEPS = [['upload', 'Upload'], ['queued', 'Queue'], ['detecting', 'Detect & track'], ['report', 'Report']];

function renderStepper(current) {
  const idx = STEPS.findIndex(([k]) => k === current);
  $('#stepper').innerHTML = STEPS.map(([, label], i) => {
    const done = i < idx, active = i === idx;
    const bar = done ? 'bg-brand' : active ? 'bg-brand/60' : 'bg-ink-700';
    const txt = done ? 'text-fog-300' : active ? 'text-brand font-medium' : 'text-fog-500';
    return `<li><div class="h-1 rounded-full ${bar}"></div><p class="mt-2 ${txt}">${label}</p></li>`;
  }).join('');
}

function setProgress(pct, text, eta) {
  $('#progressFill').style.width = `${Math.max(0, Math.min(100, pct))}%`;
  $('#progressText').textContent = text;
  $('#progressEta').textContent = eta || '';
}

function uploadVideo(fd) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    state.xhr = xhr;
    xhr.open('POST', API + '/analysis');
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setProgress(100 * e.loaded / e.total, `Uploading ${Math.round(100 * e.loaded / e.total)}%`);
    };
    xhr.onload = () => {
      state.xhr = null;
      if (xhr.status >= 200 && xhr.status < 300) return resolve(JSON.parse(xhr.responseText));
      let msg = `Upload failed (${xhr.status})`;
      try { const d = JSON.parse(xhr.responseText).detail; if (d) msg = typeof d === 'string' ? d : JSON.stringify(d); } catch { /* not json */ }
      reject(new Error(msg));
    };
    xhr.onerror = () => { state.xhr = null; reject(new Error('Network error during upload.')); };
    xhr.onabort = () => { state.xhr = null; reject(Object.assign(new Error('Upload cancelled.'), { aborted: true })); };
    xhr.send(fd);
  });
}

async function startAnalysis() {
  const site = $('#site').value.trim(), cam = $('#camera').value.trim();
  const ppe = requiredPpe();
  if (!state.file) return showFormError('Choose a video first.');
  if (!ppe.length) return showFormError('Select at least one required PPE item.');
  showFormError('');

  const fd = new FormData();
  fd.append('video', state.file, state.file.name);
  fd.append('site_name', site);
  fd.append('camera_id', cam);
  fd.append('analysis_fps', String(selectedFps()));
  fd.append('required_ppe', ppe.join(','));
  fd.append('conf_threshold', $('#conf').value);
  fd.append('min_violation_s', $('#minViol').value);
  fd.append('debug', $('#debug').checked ? 'true' : 'false');

  show('progressView');
  // Blank site/camera are fine: the server fills in defaults
  $('#progressSubject').textContent = `${site || 'Untitled site'} · ${cam || 'CAM-01'} · ${state.file.name}`;
  $('#progressError').classList.add('hidden');
  $('#progressActions').classList.add('hidden');
  $('#cancelBtn').classList.remove('hidden');
  renderStepper('upload');
  setProgress(0, 'Uploading 0%');
  try {
    const job = await uploadVideo(fd);
    state.analysisId = job.analysis_id;
    store.set(LAST_KEY, job.analysis_id);
    pollStatus();
  } catch (err) {
    if (err.aborted) { show('setupView'); return; }
    progressFailed(err.message);
  }
}

function progressFailed(message) {
  $('#progressError').textContent = message;
  $('#progressError').classList.remove('hidden');
  $('#progressActions').classList.remove('hidden');
  $('#retryBtn').classList.toggle('hidden', !state.file);
  $('#cancelBtn').classList.add('hidden');
}

async function pollStatus() {
  clearTimeout(state.pollTimer);
  let j;
  try {
    j = await api(`/analysis/${state.analysisId}`);
  } catch (err) {
    if (err.status === 404) { store.del(LAST_KEY); progressFailed('This analysis no longer exists (the server was restarted).'); return; }
    state.pollTimer = setTimeout(pollStatus, 2000);
    return;
  }
  const stage = j.status === 'queued' ? 'queued' : j.stage === 'report' ? 'report' : 'detecting';
  renderStepper(j.status === 'completed' ? 'done' : stage);
  if (j.status === 'queued') {
    setProgress(0, j.queue_position > 1 ? `Waiting in queue (position ${j.queue_position})` : 'Starting…');
  } else if (j.status === 'processing') {
    const eta = j.eta_seconds != null ? `about ${fmtDur(j.eta_seconds)} left` : '';
    const text = j.stage === 'report' ? 'Building report…'
      : !j.frames_processed ? 'Starting analysis…'
        : `Analysed ${fmtTs(j.video_position_s)} of ${fmtTs(j.duration_s)} · ${j.frames_processed} frames`;
    setProgress(j.progress, text, eta);
  }
  if (j.status === 'completed') { setProgress(100, 'Done'); return loadResults(state.analysisId); }
  if (j.status === 'failed') return progressFailed(`Analysis failed: ${j.error || 'unknown error'}`);
  if (j.status === 'cancelled') { store.del(LAST_KEY); show('setupView'); return; }
  state.pollTimer = setTimeout(pollStatus, 1000);
}

async function cancelAnalysis() {
  if (state.xhr) { state.xhr.abort(); return; }
  if (!state.analysisId) return;
  $('#progressText').textContent = 'Cancelling…';
  try { await api(`/analysis/${state.analysisId}`, { method: 'DELETE' }); } catch { /* status poll will tell */ }
}

// ---------------------------------------------------------------- results
async function loadResults(id) {
  const [results, p, ev, tr] = await Promise.all([
    api(`/analysis/${id}/results`), api(`/analysis/${id}/persons`),
    api(`/analysis/${id}/events`), api(`/analysis/${id}/tracks`),
  ]);
  const status = await api(`/analysis/${id}`);
  state.analysisId = id;
  state.results = results;
  state.persons = p.persons;
  state.events = ev.events;
  state.tracks = tr.tracks;
  state.status = status;
  state.thumbs = {};
  renderResults();
  show('resultsView');
  renderTimeline(); // needs the view visible to measure its width
  attachPlayer();
}

function renderResults() {
  const r = state.results, s = r.summary, st = state.status;
  $('#resTitle').textContent = `${st.site_name} · ${st.camera_id}`;
  $('#resMeta').textContent = `${st.filename} · ${fmtDur(s.duration_s)} · analysed ${new Date(st.created_at).toLocaleString()}`;
  $('#pdfBtn').href = `${API}/analysis/${state.analysisId}/report`;
  $('#framesLink').href = `${API}/analysis/${state.analysisId}/frames?limit=200`;
  renderKpis(s);
  $('#findings').innerHTML = r.findings.map((f) =>
    `<li class="flex gap-3"><span class="mt-2 w-1.5 h-1.5 rounded-full bg-brand shrink-0"></span><span>${esc(f)}</span></li>`).join('');
  renderPpeBars(r.ppe);
  renderWorkerFilters();
  renderWorkers();
  renderEventFilters();
  renderEvents();
  renderScene(r);
  $('#rawJson').textContent = JSON.stringify({ results: r, persons: state.persons, events: state.events }, null, 2);
}

function renderKpis(s) {
  const tiles = [
    { v: s.workers_observed, l: 'Workers observed' },
    { v: s.workers_non_compliant, l: 'With violations', tone: s.workers_non_compliant ? 'text-bad' : 'text-ok' },
    { v: fmtPct(s.overall_compliance), l: 'PPE compliance' },
    { v: s.total_events, l: 'Violation events', tone: s.total_events ? 'text-bad' : 'text-ok' },
    { v: s.longest_event ? fmtDur(s.longest_event.duration) : '–', l: 'Longest violation',
      sub: s.longest_event ? `${s.longest_event.worker} · ${s.longest_event.label}` : '' },
  ];
  $('#kpis').innerHTML = tiles.map((t, i) => `
    <div class="card p-5 ${i === tiles.length - 1 ? 'col-span-2 md:col-span-1' : ''}">
      <p class="text-sm text-fog-400">${esc(t.l)}</p>
      <p class="mt-2 text-2xl sm:text-3xl font-semibold leading-none tracking-tight tabular ${t.tone || ''}">${esc(t.v)}</p>
      ${t.sub ? `<p class="text-xs text-fog-500 mt-2 truncate">${esc(t.sub)}</p>` : ''}
    </div>`).join('');
}

function renderPpeBars(ppe) {
  $('#ppeBars').innerHTML = ppe.map((b) => {
    const pct = b.compliance == null ? null : Math.round(b.compliance * 100);
    const color = b.required ? TYPES[b.type].color : '#5B636E';
    const detail = b.workers_observed
      ? `${b.workers_observed} worker${b.workers_observed === 1 ? '' : 's'} assessed` +
        (b.required ? ` · ${b.workers_violating} in violation · ${b.events} event${b.events === 1 ? '' : 's'}` : '')
      : 'Not visible on any worker';
    return `<div>
      <div class="flex items-baseline justify-between text-sm">
        <span>${esc(b.label)}${b.required ? '' : ' <span class="text-xs text-fog-500">not required</span>'}</span>
        <span class="tabular font-medium">${pct == null ? '–' : pct + '%'}</span>
      </div>
      <div class="mt-2 h-2 rounded-full bg-ink-800 overflow-hidden" role="img" aria-label="${esc(b.label)} compliance ${pct ?? 'unknown'}%">
        <div class="h-full rounded-full" style="width:${pct ?? 0}%;background:${color}"></div>
      </div>
      <p class="text-xs text-fog-500 mt-2">${detail}</p>
    </div>`;
  }).join('') + '<p class="text-xs text-fog-500 pt-1">Boots are not assessed (no footwear class in the model).</p>';
}

// ---------------------------------------------------------------- workers
function ppeChip([item, it]) {
  const s = ITEM_STATUS[it.status] || ITEM_STATUS.unseen;
  const dim = it.required ? '' : ' opacity-50';
  const pct = it.compliance == null ? '' : ` <span class="opacity-70 tabular">${Math.round(it.compliance * 100)}%</span>`;
  return `<span class="inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-xs ${s.cls}${dim}" title="${esc(it.label)}: ${s.label}${it.required ? '' : ' (not required)'}">` +
    `${ITEM_SHORT[item]}: ${s.label}${pct}</span>`;
}

function sortedWorkers() {
  const sort = $('#workerSort').value;
  let list = state.persons.slice();
  if (state.workerFilter !== 'all') list = list.filter((p) => p.status === state.workerFilter);
  const by = {
    severity: (a, b) => b.severity.score - a.severity.score || a.worker_id - b.worker_id,
    violation: (a, b) => b.violation_s - a.violation_s || a.worker_id - b.worker_id,
    first: (a, b) => a.first_seen - b.first_seen,
  }[sort];
  return list.sort(by);
}

function renderWorkerFilters() {
  const counts = { all: state.persons.length };
  for (const p of state.persons) counts[p.status] = (counts[p.status] || 0) + 1;
  const opts = [['all', 'All'], ['non_compliant', 'Violations'], ['compliant', 'Compliant'], ['unconfirmed', 'Unconfirmed']]
    .filter(([k]) => k === 'all' || counts[k]);
  $('#workerFilters').innerHTML = opts.map(([k, l]) =>
    `<button type="button" data-filter="${k}" aria-pressed="${state.workerFilter === k}"
      class="inline-flex h-7 items-center gap-1.5 rounded-md px-3 ${state.workerFilter === k ? 'bg-ink-700 text-fog-100 shadow-sm' : 'text-fog-400 hover:text-fog-100'}">${l} <span class="text-fog-500 tabular">${counts[k] || 0}</span></button>`).join('');
}

function thumbHtml(p, cls) {
  const src = state.thumbs[p.worker_id];
  return src
    ? `<img src="${src}" alt="${esc(p.label)}" class="${cls} object-cover">`
    : `<div class="${cls} grid place-items-center text-fog-500 text-lg font-semibold" data-thumb="${p.worker_id}" data-cls="${cls}">W${p.worker_id}</div>`;
}

function renderWorkers() {
  const list = sortedWorkers();
  $('#workerCount').textContent = `· ${state.persons.length}`;
  const brief = state.results.summary.brief_tracks;
  $('#briefNote').textContent = brief ? `${brief} brief detection${brief === 1 ? '' : 's'} (under 1 s on camera, no violations) not counted as workers.` : '';
  if (!list.length) {
    $('#workers').innerHTML = `<p class="text-sm text-fog-400 col-span-full py-6">${state.persons.length ? 'No workers match this filter.' : 'No workers were tracked in this footage.'}</p>`;
    return;
  }
  $('#workers').innerHTML = list.map((p) => {
    const sev = SEVERITY[p.severity.level];
    const nEv = p.event_ids.length;
    const exp = [];
    if (p.exposure.machinery) exp.push(`near machinery ${fmtDur(p.exposure.machinery)}`);
    if (p.exposure.vehicle) exp.push(`near vehicles ${fmtDur(p.exposure.vehicle)}`);
    return `<button type="button" data-worker="${p.worker_id}"
      class="card text-left overflow-hidden transition hover:border-white/[0.12] hover:bg-ink-850 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand/50">
      <div class="flex gap-4 p-5">
        <div class="w-20 h-28 rounded-lg bg-ink-800 overflow-hidden shrink-0 ring-1 ring-white/[0.06]">${thumbHtml(p, 'w-full h-full')}</div>
        <div class="min-w-0 flex-1">
          <div class="flex items-center justify-between gap-2">
            <h3 class="font-semibold">${esc(p.label)}</h3>
            ${p.status === 'non_compliant'
              ? `<span class="text-xs rounded-full px-2 py-0.5 ${sev.cls}">${sev.label}</span>`
              : `<span class="text-xs rounded-full px-2 py-0.5 ${p.status === 'compliant' ? 'text-ok bg-ok/10' : 'text-fog-400 bg-ink-800'}">${WORKER_STATUS[p.status]}</span>`}
          </div>
          <p class="text-xs text-fog-500 tabular mt-1">${fmtTs(p.first_seen)} – ${fmtTs(p.last_seen)} · ${fmtDur(p.visible_s)} visible</p>
          <div class="mt-3 flex flex-wrap gap-1.5">${Object.entries(p.ppe).map(ppeChip).join('')}</div>
          <p class="mt-3 text-xs text-fog-400">${nEv ? `${nEv} event${nEv === 1 ? '' : 's'} · ${fmtDur(p.violation_s)} in violation (${fmtPct(p.violation_share)})` : 'No violation events'}${exp.length ? ' · ' + exp.join(', ') : ''}</p>
        </div>
      </div>
      <div class="h-1 bg-ink-800"><div class="h-full bg-bad" style="width:${Math.round(p.violation_share * 100)}%"></div></div>
    </button>`;
  }).join('');
}

// ---------------------------------------------------------------- drawer
function openWorker(wid) {
  const p = state.persons.find((x) => x.worker_id === Number(wid));
  if (!p) return;
  state.lastFocus = document.activeElement;
  const evs = state.events.filter((e) => e.worker_id === p.worker_id);
  const sev = SEVERITY[p.severity.level];
  $('#drawerTitle').innerHTML = `${esc(p.label)} ${p.status === 'non_compliant'
    ? `<span class="ml-2 align-middle text-xs rounded-full px-2 py-0.5 ${sev.cls}">${sev.label} severity</span>` : ''}`;
  const facts = [
    ['On camera', `${fmtTs(p.first_seen)} – ${fmtTs(p.last_seen)}`],
    ['Visible for', fmtDur(p.visible_s)],
    ['In violation', `${fmtDur(p.violation_s)} (${fmtPct(p.violation_share)})`],
    ['Near machinery', fmtDur(p.exposure.machinery || 0)],
    ['Near vehicles', fmtDur(p.exposure.vehicle || 0)],
  ];
  const ppeRows = Object.values(p.ppe).map((it) => {
    const s = ITEM_STATUS[it.status] || ITEM_STATUS.unseen;
    const total = it.present_s + it.missing_s + it.unseen_s || 1;
    return `<div class="py-2.5 border-b border-ink-700 last:border-0 ${it.required ? '' : 'opacity-60'}">
      <div class="flex items-center justify-between text-sm">
        <span>${esc(it.label)}${it.required ? '' : ' <span class="text-xs text-fog-500">not required</span>'}</span>
        <span class="inline-flex items-center rounded-md border px-1.5 py-0.5 text-xs ${s.cls}">${s.label}${it.compliance == null ? '' : ' · ' + Math.round(it.compliance * 100) + '%'}</span>
      </div>
      <div class="mt-2 h-1.5 rounded-full overflow-hidden flex bg-ink-800" title="Wearing ${fmtDur(it.present_s)} · missing ${fmtDur(it.missing_s)} · not seen ${fmtDur(it.unseen_s)}">
        <div class="bg-ok" style="width:${100 * it.present_s / total}%"></div>
        <div class="bg-bad" style="width:${100 * it.missing_s / total}%"></div>
      </div>
      <p class="text-xs text-fog-500 mt-1 tabular">Wearing ${fmtDur(it.present_s)} · missing ${fmtDur(it.missing_s)} · not seen ${fmtDur(it.unseen_s)}</p>
    </div>`;
  }).join('');
  const evRows = evs.length ? evs.map((e) => `
    <button type="button" data-seek="${e.t_start}" class="w-full text-left flex items-center gap-3 rounded-lg px-3 py-2 hover:bg-ink-800 focus:outline-none focus-visible:bg-ink-800">
      <span class="w-2.5 h-2.5 rounded-sm shrink-0" style="background:${TYPES[e.type].color}"></span>
      <span class="flex-1 min-w-0">
        <span class="block text-sm">${esc(e.label)}${e.near_hazard ? ' <span class="text-xs text-warn">· near machinery/vehicle</span>' : ''}</span>
        <span class="block text-xs text-fog-500 tabular">${fmtTs(e.t_start)} – ${fmtTs(e.t_end)} · ${fmtDur(e.duration)} · conf ${e.peak_conf.toFixed(2)}</span>
      </span>
      <span class="text-xs text-brand shrink-0">Play ▸</span>
    </button>`).join('') : '<p class="text-sm text-fog-400 px-3">No violation events.</p>';

  $('#drawerBody').innerHTML = `
    <div class="flex gap-4">
      <div class="w-24 h-32 rounded-lg bg-ink-800 overflow-hidden shrink-0 ring-1 ring-white/[0.06]">${thumbHtml(p, 'w-full h-full')}</div>
      <dl class="text-sm grid grid-cols-[auto,1fr] gap-x-4 gap-y-1.5 content-start">
        ${facts.map(([k, v]) => `<dt class="text-fog-500">${k}</dt><dd class="tabular">${v}</dd>`).join('')}
      </dl>
    </div>
    <button type="button" data-seek="${p.best_frame.t}" class="mt-4 text-sm text-brand hover:underline">Jump to clearest view (${fmtTs(p.best_frame.t)}) ▸</button>
    <h3 class="mt-6 text-sm font-semibold">PPE</h3>
    <div class="mt-1">${ppeRows}</div>
    <h3 class="mt-6 mb-2 text-sm font-semibold">Violation events</h3>
    <div class="-mx-3">${evRows}</div>`;
  $('#drawer').classList.add('open');
  $('#drawerBackdrop').classList.add('open');
  $('#drawerClose').focus();
}

function closeDrawer() {
  if (!$('#drawer').classList.contains('open')) return;
  $('#drawer').classList.remove('open');
  $('#drawerBackdrop').classList.remove('open');
  state.lastFocus?.focus?.();
}

// ---------------------------------------------------------------- events table
function renderEventFilters() {
  const types = [...new Set(state.events.map((e) => e.type))];
  $('#evType').innerHTML = '<option value="">All types</option>' + types.map((t) => `<option value="${t}">${TYPES[t].label}</option>`).join('');
  const workers = [...new Set(state.events.map((e) => e.worker_id))].sort((a, b) => a - b);
  $('#evWorker').innerHTML = '<option value="">All workers</option>' + workers.map((w) => `<option value="${w}">Worker ${w}</option>`).join('');
}

function filteredEvents() {
  const t = $('#evType').value, w = $('#evWorker').value;
  const { key, dir } = state.evSort;
  return state.events
    .filter((e) => (!t || e.type === t) && (!w || e.worker_id === Number(w)))
    .sort((a, b) => (a[key] > b[key] ? 1 : a[key] < b[key] ? -1 : a.t_start - b.t_start) * dir);
}

function renderEvents() {
  const list = filteredEvents();
  $('#eventCount').textContent = `· ${state.events.length}`;
  $('#noEvents').classList.toggle('hidden', state.events.length > 0);
  $$('thead [data-sort]').forEach((b) => {
    const active = b.dataset.sort === state.evSort.key;
    b.innerHTML = b.textContent.replace(/[ ▲▼]+$/, '') + (active ? (state.evSort.dir > 0 ? ' ▲' : ' ▼') : '');
    b.className = active ? 'text-fog-100' : 'hover:text-fog-100';
  });
  $('#eventRows').innerHTML = list.map((e) => {
    const sevLevel = e.severity >= 15 ? 'high' : e.severity >= 5 ? 'medium' : 'low';
    return `<tr class="border-b border-white/[0.05] last:border-0 hover:bg-ink-850 cursor-pointer" data-seek="${e.t_start}">
      <td class="py-3 px-3 first:pl-0"><button type="button" data-worker="${e.worker_id}" class="hover:text-brand underline-offset-2 hover:underline">${esc(e.worker)}</button></td>
      <td class="py-3 px-3"><span class="inline-flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-sm" style="background:${TYPES[e.type].color}"></span>${esc(e.label)}</span></td>
      <td class="py-3 px-3">${fmtTs(e.t_start)}</td>
      <td class="py-3 px-3">${fmtDur(e.duration)}</td>
      <td class="py-3 px-3">${e.peak_conf.toFixed(2)}</td>
      <td class="py-3 px-3 ${e.near_hazard ? 'text-warn' : 'text-fog-500'}">${e.near_hazard ? 'Near machinery/vehicle' : '–'}</td>
      <td class="py-3 px-3 last:pr-0"><span class="rounded-md px-1.5 py-0.5 text-xs ${SEVERITY[sevLevel].cls}">${e.severity.toFixed(1)}</span></td>
    </tr>`;
  }).join('');
}

function exportCsv() {
  const cols = ['event_id', 'worker', 'type', 'label', 't_start', 't_end', 'duration', 'peak_conf', 'mean_conf', 'near_hazard', 'severity'];
  const rows = [cols.join(',')].concat(state.events.map((e) => cols.map((c) => {
    const v = String(e[c] ?? '');
    return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
  }).join(',')));
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `rakshai_events_${state.analysisId.slice(0, 8)}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function renderScene(r) {
  const sc = r.scene, st = r.settings;
  const un = Object.entries(sc.unattributed_observations || {}).map(([k, v]) => `${k} ×${v}`).join(', ');
  const rows = [
    ['Peak workers in frame', sc.peak_workers],
    ['Machinery', sc.peak_machinery ? `up to ${sc.peak_machinery} · ${fmtPct(sc.machinery_time_share)} of footage` : 'not seen'],
    ['Vehicles', sc.peak_vehicles ? `up to ${sc.peak_vehicles} · ${fmtPct(sc.vehicle_time_share)} of footage` : 'not seen'],
    ['Safety cones', sc.peak_cones ? `up to ${sc.peak_cones} · ${fmtPct(sc.cones_time_share)} of footage` : 'not seen'],
    ['Unlinked detections', un || 'none'],
    ['Sampling', st.analysis_fps ? `${st.sample_fps} fps · ${r.summary.frames_analyzed} frames` : `every frame · ${r.summary.frames_analyzed} frames`],
    ['Processing time', fmtDur(r.processing_s)],
  ];
  $('#scene').innerHTML = rows.map(([k, v]) => `<dt class="text-fog-500">${k}</dt><dd class="tabular">${esc(v)}</dd>`).join('');
}

// ---------------------------------------------------------------- review player + overlay
const player = () => $('#player');

function attachPlayer() {
  const v = player();
  if (state.fileUrl) {
    if (v.src !== state.fileUrl) v.src = state.fileUrl;
    $('#playerWrap').classList.remove('hidden');
    $('#noVideo').classList.add('hidden');
    captureThumbnails();
  } else {
    v.removeAttribute('src');
    $('#playerWrap').classList.add('hidden');
    $('#noVideo').classList.remove('hidden');
  }
  renderNow(0);
}

function maxGap() {
  const fps = state.results?.settings?.sample_fps || 5;
  return Math.max(0.4, 2.5 / fps);
}

// Workers visible at time t: [{wid, box:[x1,y1,x2,y2] (0-1), types:[...]}]
function workersAt(t) {
  const out = [];
  const gap = maxGap();
  for (const [wid, rows] of Object.entries(state.tracks)) {
    if (!rows.length || t < rows[0][0] - 0.05 || t > rows[rows.length - 1][0] + gap) continue;
    let lo = 0, hi = rows.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (rows[mid][0] <= t + 1e-3) lo = mid; else hi = mid - 1; }
    const row = rows[lo];
    if (row[0] > t + 0.05 || t - row[0] > gap) continue;
    out.push({ wid: Number(wid), box: row.slice(1, 5), types: row[5] ? row[5].split(',') : [] });
  }
  return out.sort((a, b) => a.wid - b.wid);
}

function drawOverlay() {
  const v = player(), c = $('#overlay');
  const rect = v.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(rect.width * dpr) || c.height !== Math.round(rect.height * dpr)) {
    c.width = Math.round(rect.width * dpr); c.height = Math.round(rect.height * dpr);
  }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const t = v.currentTime || 0;
  const active = workersAt(t);
  renderNow(t, active);
  updatePlayhead(t);
  if (!$('#showBoxes').checked || !v.videoWidth) return;
  const s = Math.min(rect.width / v.videoWidth, rect.height / v.videoHeight);
  const dw = v.videoWidth * s, dh = v.videoHeight * s, ox = (rect.width - dw) / 2, oy = (rect.height - dh) / 2;
  ctx.font = '600 12px Inter, system-ui, sans-serif';
  ctx.textBaseline = 'top';
  for (const w of active) {
    const [x1, y1, x2, y2] = w.box;
    const x = ox + x1 * dw, y = oy + y1 * dh, bw = (x2 - x1) * dw, bh = (y2 - y1) * dh;
    const bad = w.types.length > 0;
    const color = bad ? '#EF5350' : '#3FB67A';
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, bw, bh);
    const label = `Worker ${w.wid}${bad ? ' · ' + w.types.map((k) => TYPES[k].label).join(', ') : ''}`;
    const tw = ctx.measureText(label).width + 10;
    const ly = y > 20 ? y - 20 : y;
    ctx.fillStyle = color;
    ctx.fillRect(x - 1, ly, tw, 19);
    ctx.fillStyle = bad ? '#fff' : '#06140C';
    ctx.fillText(label, x + 4, ly + 3);
  }
}

let nowSig = '';
function renderNow(t, active = workersAt(t)) {
  const sig = active.map((w) => w.wid + ':' + w.types.join('+')).join('|');
  if (sig === nowSig && $('#nowList').childElementCount) return;
  nowSig = sig;
  if (!active.length) {
    $('#nowList').innerHTML = '<li class="text-fog-500">No tracked workers at this moment.</li>';
    return;
  }
  $('#nowList').innerHTML = active.map((w) => {
    const bad = w.types.length > 0;
    return `<li><button type="button" data-worker="${w.wid}" class="w-full text-left flex items-center gap-3 rounded-lg border border-white/[0.06] bg-ink-850 px-3 py-2.5 transition hover:border-white/[0.12]">
      <span class="w-2 h-2 rounded-full ${bad ? 'bg-bad' : 'bg-ok'}"></span>
      <span class="flex-1">Worker ${w.wid}</span>
      <span class="text-xs ${bad ? 'text-bad' : 'text-ok'}">${bad ? w.types.map((k) => TYPES[k].label).join(', ') : 'PPE OK'}</span>
    </button></li>`;
  }).join('');
}

function seek(t) {
  closeDrawer();
  const v = player();
  // Bring the player (or the "no video" panel) fully into view below the sticky header
  const target = v.getAttribute('src') ? $('#playerWrap') : $('#reviewCard');
  const r = target.getBoundingClientRect();
  if (r.top < 64 || r.bottom > window.innerHeight) target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  if (!v.getAttribute('src')) { updatePlayhead(t); renderNow(t); return; }
  v.currentTime = Math.max(0, t);
  v.pause();
}

function initPlayer() {
  const v = player();
  const loop = () => {
    drawOverlay();
    if (!v.paused && !v.ended) {
      if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(loop); else requestAnimationFrame(loop);
    }
  };
  v.addEventListener('play', loop);
  for (const ev of ['seeked', 'loadedmetadata', 'pause', 'timeupdate']) v.addEventListener(ev, drawOverlay);
  v.addEventListener('error', () => {
    if (!state.fileUrl) return;
    $('#playerWrap').classList.add('hidden');
    $('#noVideo').classList.remove('hidden');
    $('#noVideo p').textContent = "This browser can't play this video format, so the review player is unavailable.";
  });
  $('#showBoxes').addEventListener('change', drawOverlay);
  if (window.ResizeObserver) new ResizeObserver(() => drawOverlay()).observe($('#playerWrap'));
  $('#attachVideoBtn').addEventListener('click', () => $('#attachInput').click());
  $('#attachInput').addEventListener('change', (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
    state.file = f;
    state.fileUrl = URL.createObjectURL(f);
    attachPlayer();
  });
}

// Worker thumbnails are cropped in the browser from the local file at each worker's
// clearest frame, so nothing image-related is stored or served by the backend.
async function captureThumbnails() {
  if (!state.fileUrl || !state.persons.length) return;
  const url = state.fileUrl;
  const v = document.createElement('video');
  v.muted = true; v.preload = 'auto'; v.playsInline = true; v.src = url;
  const ready = await new Promise((res) => {
    const to = setTimeout(() => res(false), 8000);
    v.addEventListener('loadeddata', () => { clearTimeout(to); res(true); }, { once: true });
    v.addEventListener('error', () => { clearTimeout(to); res(false); }, { once: true });
  });
  if (!ready) return;
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  for (const p of state.persons) {
    if (state.fileUrl !== url) return; // video replaced meanwhile
    const ok = await new Promise((res) => {
      const to = setTimeout(() => res(false), 4000);
      v.addEventListener('seeked', () => { clearTimeout(to); res(true); }, { once: true });
      v.currentTime = Math.min(p.best_frame.t + 0.001, Math.max(0, v.duration - 0.05));
    });
    if (!ok) continue;
    const [x1, y1, x2, y2] = p.best_frame.bbox;
    const W = v.videoWidth, H = v.videoHeight;
    const mx = 0.1 * (x2 - x1), my = 0.05 * (y2 - y1);
    const sx = Math.max(0, (x1 - mx) * W), sy = Math.max(0, (y1 - my) * H);
    const sw = Math.min(W - sx, (x2 - x1 + 2 * mx) * W), sh = Math.min(H - sy, (y2 - y1 + 2 * my) * H);
    if (sw < 4 || sh < 4) continue;
    const scale = 240 / sh;
    canvas.width = Math.max(1, Math.round(sw * scale)); canvas.height = 240;
    ctx.drawImage(v, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    try {
      state.thumbs[p.worker_id] = canvas.toDataURL('image/jpeg', 0.82);
    } catch { return; }
    $$(`[data-thumb="${p.worker_id}"]`).forEach((el) => {
      const img = document.createElement('img');
      img.src = state.thumbs[p.worker_id];
      img.alt = p.label;
      img.className = `${el.dataset.cls} object-cover`;
      el.replaceWith(img);
    });
  }
  v.removeAttribute('src'); v.load();
}

// ---------------------------------------------------------------- timeline (SVG)
const TL = { left: 84, right: 12, top: 22, row: 26, bottom: 8 };

function renderTimeline() {
  const el = $('#timeline');
  const persons = state.persons.slice().sort((a, b) => a.worker_id - b.worker_id);
  const duration = state.results.summary.duration_s || 1;
  const types = Object.keys(TYPES).filter((t) => state.events.some((e) => e.type === t));
  $('#timelineLegend').innerHTML = types.map((t) =>
    `<span class="inline-flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-sm" style="background:${TYPES[t].color}"></span>${TYPES[t].label}</span>`).join('') +
    '<span class="inline-flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-sm bg-ink-700"></span>On camera</span>';
  if (!persons.length) { el.innerHTML = '<p class="text-sm text-fog-500 py-4">No workers were tracked.</p>'; return; }

  const W = Math.max(el.clientWidth, 320);
  const H = TL.top + persons.length * TL.row + TL.bottom;
  const x = (t) => TL.left + (t / duration) * (W - TL.left - TL.right);
  const gap = maxGap();
  const ticks = niceTicks(duration, Math.max(3, Math.floor((W - TL.left) / 90)));
  let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" class="block select-none cursor-pointer" role="img" aria-label="Violation timeline per worker">`;
  for (const t of ticks) {
    svg += `<line x1="${x(t)}" x2="${x(t)}" y1="${TL.top - 4}" y2="${H - TL.bottom}" stroke="#2A3038" stroke-width="1"/>` +
      `<text x="${x(t)}" y="12" fill="#7B838F" font-size="11" text-anchor="middle">${fmtTick(t, duration)}</text>`;
  }
  persons.forEach((p, i) => {
    const y = TL.top + i * TL.row;
    svg += `<rect class="lane-hit" x="0" y="${y}" width="${W}" height="${TL.row}" fill="transparent"/>`;
    svg += `<text x="8" y="${y + TL.row / 2 + 4}" fill="#B8BEC7" font-size="12">${esc(p.label)}</text>`;
    // on-camera segments
    const rows = state.tracks[String(p.worker_id)] || [];
    let start = null, prev = null;
    const seg = (a, b) => `<rect x="${x(a)}" y="${y + 5}" width="${Math.max(2, x(b) - x(a))}" height="${TL.row - 10}" rx="3" fill="#2A3038"/>`;
    for (const r of rows) {
      if (start === null) start = r[0];
      else if (r[0] - prev > gap) { svg += seg(start, prev + 0.2); start = r[0]; }
      prev = r[0];
    }
    if (start !== null) svg += seg(start, prev + 0.2);
    // events, stacked per type within the lane
    const evs = state.events.filter((e) => e.worker_id === p.worker_id);
    const subH = (TL.row - 10) / Math.max(1, types.length);
    for (const e of evs) {
      const k = types.indexOf(e.type);
      svg += `<rect x="${x(e.t_start)}" y="${y + 5 + k * subH}" width="${Math.max(3, x(e.t_end) - x(e.t_start))}" height="${subH}" rx="2" fill="${TYPES[e.type].color}">` +
        `<title>${esc(p.label)} · ${esc(e.label)} · ${fmtTs(e.t_start)}–${fmtTs(e.t_end)} (${fmtDur(e.duration)})${e.near_hazard ? ' · near machinery/vehicle' : ''}</title></rect>`;
    }
  });
  svg += `<line id="playhead" x1="${x(0)}" x2="${x(0)}" y1="${TL.top - 6}" y2="${H - TL.bottom}" stroke="#F5B400" stroke-width="2"/>`;
  svg += '</svg>';
  el.innerHTML = svg;
  const node = el.querySelector('svg');
  node.addEventListener('click', (ev) => {
    const r = node.getBoundingClientRect();
    const px = ev.clientX - r.left;
    if (px < TL.left) return;
    const t = ((px - TL.left) / (W - TL.left - TL.right)) * duration;
    seek(Math.max(0, Math.min(duration, t)));
  });
  state.tlScale = { x, duration };
  updatePlayhead(player().currentTime || 0);
}

function updatePlayhead(t) {
  const ph = $('#playhead');
  if (!ph || !state.tlScale) return;
  const X = state.tlScale.x(Math.min(t, state.tlScale.duration));
  ph.setAttribute('x1', X); ph.setAttribute('x2', X);
}

function niceTicks(duration, maxTicks) {
  const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  const step = steps.find((s) => duration / s <= maxTicks) || 3600;
  const out = [];
  for (let t = 0; t <= duration + 1e-6; t += step) out.push(t);
  return out;
}

function fmtTick(t, duration) {
  if (duration < 60) return `${Number.isInteger(t) ? t : t.toFixed(1)}s`;
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.round(t % 60);
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`;
}

// ---------------------------------------------------------------- wiring
function resetToSetup() {
  clearTimeout(state.pollTimer);
  closeDrawer();
  state.analysisId = null;
  state.results = null;
  store.del(LAST_KEY);
  player().pause();
  show('setupView');
}

function initGlobalHandlers() {
  document.addEventListener('click', (e) => {
    const worker = e.target.closest('[data-worker]');
    if (worker) { e.stopPropagation(); openWorker(worker.dataset.worker); return; }
    const seekEl = e.target.closest('[data-seek]');
    if (seekEl) { seek(Number(seekEl.dataset.seek)); return; }
    const filter = e.target.closest('[data-filter]');
    if (filter) { state.workerFilter = filter.dataset.filter; renderWorkerFilters(); renderWorkers(); return; }
    const sort = e.target.closest('thead [data-sort]');
    if (sort) {
      const k = sort.dataset.sort;
      state.evSort = { key: k, dir: state.evSort.key === k ? -state.evSort.dir : (k === 't_start' || k === 'worker_id' ? 1 : -1) };
      renderEvents();
      return;
    }
    if (e.target.closest('[data-action="new"]')) resetToSetup();
  });
  $('#workerSort').addEventListener('change', renderWorkers);
  $('#evType').addEventListener('change', renderEvents);
  $('#evWorker').addEventListener('change', renderEvents);
  $('#csvBtn').addEventListener('click', exportCsv);
  $('#cancelBtn').addEventListener('click', cancelAnalysis);
  $('#retryBtn').addEventListener('click', startAnalysis);
  $('#drawerClose').addEventListener('click', closeDrawer);
  $('#drawerBackdrop').addEventListener('click', closeDrawer);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.results) renderTimeline(); }, 150);
  });
}

// Resume the last analysis after a reload (results live in server memory).
async function resumeLast() {
  const id = store.get(LAST_KEY);
  if (!id) return;
  try {
    const j = await api(`/analysis/${id}`);
    state.analysisId = id;
    if (j.status === 'completed') await loadResults(id);
    else if (j.status === 'queued' || j.status === 'processing') {
      show('progressView');
      $('#progressSubject').textContent = `${j.site_name} · ${j.camera_id} · ${j.filename}`;
      pollStatus();
    } else store.del(LAST_KEY);
  } catch {
    store.del(LAST_KEY);
  }
}

initSetup();
initPlayer();
initGlobalHandlers();
loadHealth();
resumeLast();
