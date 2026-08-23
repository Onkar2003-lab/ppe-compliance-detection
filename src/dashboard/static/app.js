/* Dashboard client: pick a source, draw the zone, watch the run.
   The zone is kept in NORMALISED coordinates the whole way through, so what the browser draws
   and what the monitor scores are the same polygon whatever the display size. */

const $ = (id) => document.getElementById(id);

const state = {
  points: [],          // normalised [x, y], whichever way they were placed
  mode: 'rect',        // 'rect' = drag a box (default) · 'polygon' = click each corner
  sourceReady: false,
  running: false,
  seen: new Set(),     // violation rows already in the table
};

/* --------------------------------------------------------------- source */

/* Segmented controls are scoped to their own group. They were not: one handler matched every
   button on the page, so choosing a zone mode also cleared the source tabs and hid all three
   source panels, because a zone button has no data-tab to match a panel against. */
document.querySelectorAll('[data-tab]').forEach((tab) => {
  tab.addEventListener('click', () => {
    tab.parentElement.querySelectorAll('[data-tab]').forEach((t) => t.classList.toggle('active', t === tab));
    document.querySelectorAll('[data-panel]').forEach((p) =>
      p.classList.toggle('active', p.dataset.panel === tab.dataset.tab));
  });
});

/* ----------------------------------------------------------------- steps */

/* The rail is a sequence, so a step that has been answered folds to one line and the next one
   opens. What is left to do stays on screen; what is settled becomes a summary you can reopen. */
function openStep(id, open = true) {
  const step = $(id);
  step.classList.toggle('open', open);
  step.querySelector('.step-head').setAttribute('aria-expanded', String(open));
}

function markStep(id, done, summary) {
  const step = $(id);
  step.classList.toggle('done', done);
  if (summary !== undefined) step.querySelector('.step-sum').textContent = summary;
}

document.querySelectorAll('.step-head').forEach((head) => {
  head.addEventListener('click', () => {
    const step = head.closest('.step');
    openStep(step.id, !step.classList.contains('open'));
  });
});

/** Area of the normalised polygon, as a percentage of the frame, by the shoelace formula. */
function zoneArea(points) {
  let sum = 0;
  for (let i = 0; i < points.length; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
}

function updateRulesSummary() {
  const required = [];
  if ($('req-helmet').checked) required.push('Helmet');
  if ($('req-vest').checked) required.push('Vest');
  markStep('step-rules', required.length > 0, required.join(' + ') || 'Nothing required');
}
['req-helmet', 'req-vest'].forEach((id) => $(id).addEventListener('change', updateRulesSummary));

const drop = $('drop');
$('browse').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => e.target.files[0] && uploadFile(e.target.files[0]));

['dragenter', 'dragover'].forEach((type) =>
  drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((type) =>
  drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

document.querySelectorAll('[data-open]').forEach((button) => {
  button.addEventListener('click', () => {
    const spec = button.dataset.open === 'webcam' ? $('camera-index').value : $('stream-url').value;
    if (!spec) return say('Enter a camera index or stream URL first.', true);
    openSource({ source: spec }, button);
  });
});

async function uploadFile(file) {
  const body = new FormData();
  body.append('file', file);
  say(`Reading ${file.name}…`);
  await openSource(body);
}

async function openSource(payload, button) {
  if (button) { button.disabled = true; button.textContent = 'Connecting…'; }
  try {
    const isForm = payload instanceof FormData;
    const response = await fetch('/api/source', {
      method: 'POST',
      body: isForm ? payload : JSON.stringify(payload),
      headers: isForm ? {} : { 'Content-Type': 'application/json' },
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'could not open that source');
    showSource(data);
  } catch (error) {
    say(error.message, true);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = button.dataset.open === 'webcam' ? 'Connect camera' : 'Connect stream';
    }
  }
}

function showSource(info, keepZone = false) {
  state.sourceReady = true;
  // A new source invalidates the polygon drawn over the old one; returning to the editor
  // after a run must not.
  if (!keepZone) state.points = [];
  // A clip is named by its file, not by where this machine happened to put it: the full
  // upload path is this laptop's business, and it crowds out the name in a screenshot.
  $('source-label').textContent = info.kind === 'file'
    ? info.label.replace(/^file:/, '').split(/[\\/]/).pop()
    : info.label;
  $('fact-res').textContent = `${info.width} × ${info.height}`;
  $('fact-fps').textContent = `${info.fps} fps`;
  $('fact-len').textContent = info.duration ? `${info.duration} s (${info.frames} frames)` : 'live';
  $('source-facts').classList.remove('hidden');
  $('empty').classList.add('hidden');
  $('live').classList.add('hidden');
  $('canvas-wrap').classList.remove('hidden');
  const image = $('frame');
  image.onload = resizeCanvas;
  image.src = `/api/first-frame?t=${Date.now()}`;
  if (image.complete) resizeCanvas();  // a cached frame fires `load` before we listen
  $('start').disabled = false;
  setPill('idle', 'Ready');

  // Step one is answered: fold it to its summary and put the operator in front of the zone,
  // which is the only thing now standing between them and a run.
  markStep('step-source', true, `${$('source-label').textContent} · ${info.width}×${info.height}`);
  openStep('step-source', false);
  openStep('step-zone', true);
  updateRulesSummary();

  drawZone();
}

/** Return to the zone editor after a run, so the region can be adjusted and re-run. */
function showEditor() {
  $('live').classList.add('hidden');
  $('canvas-wrap').classList.remove('hidden');
  resizeCanvas();
}

/* ----------------------------------------------------------------- zone */

const canvas = $('zone-canvas');
const ctx = canvas.getContext('2d');
const MIN_DRAG = 0.02;  // ignore an accidental click-drag of a couple of percent

function resizeCanvas() {
  const image = $('frame');
  canvas.width = image.clientWidth;
  canvas.height = image.clientHeight;
  drawZone();
}
window.addEventListener('resize', resizeCanvas);

/* Two ways to mark the zone. Dragging a box is the default because most zones are a
   rectangle and one drag is the whole interaction; clicking a shape is there for the
   L-shaped bay around an obstruction. Both end up as a normalised polygon, so the server,
   the saved zone file and the membership test never learn which was used. */
document.querySelectorAll('[data-zone-mode]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-zone-mode]').forEach((b) =>
      b.classList.toggle('active', b === button));
    state.mode = button.dataset.zoneMode;
    state.points = [];
    $('zone-undo').disabled = state.mode === 'rect';
    $('zone-help').innerHTML = state.mode === 'rect'
      ? 'Drag across the frame to box off the area where PPE is mandatory. A worker counts as inside when their <strong>feet</strong> are within it.'
      : 'Click the frame to place each corner, right-click to undo. Three points or more makes a zone.';
    $('zone-hint').textContent = state.mode === 'rect'
      ? 'Drag across the frame to mark the safety zone'
      : 'Click to place each corner · right-click to undo';
    drawZone();
  });
});

const clamp = (value) => Math.min(1, Math.max(0, value));

function pointAt(event) {
  const box = canvas.getBoundingClientRect();
  return [clamp((event.clientX - box.left) / box.width), clamp((event.clientY - box.top) / box.height)];
}

/** The four corners of the box spanned by two dragged points. */
function corners([x1, y1], [x2, y2]) {
  const [left, right] = [Math.min(x1, x2), Math.max(x1, x2)];
  const [top, bottom] = [Math.min(y1, y2), Math.max(y1, y2)];
  return [[left, top], [right, top], [right, bottom], [left, bottom]];
}

let dragFrom = null;

canvas.addEventListener('mousedown', (event) => {
  if (state.running || state.mode !== 'rect' || event.button !== 0) return;
  dragFrom = pointAt(event);
  state.points = [];
});

canvas.addEventListener('mousemove', (event) => {
  if (!dragFrom) return;
  drawZone(corners(dragFrom, pointAt(event)));
});

['mouseup', 'mouseleave'].forEach((type) =>
  canvas.addEventListener(type, (event) => {
    if (!dragFrom) return;
    const to = pointAt(event);
    const box = corners(dragFrom, to);
    dragFrom = null;
    // A click with no real drag is not a zone; it would be an invisible sliver nobody meant.
    const wide = Math.abs(to[0] - box[0][0]) > MIN_DRAG;
    const tall = Math.abs(to[1] - box[0][1]) > MIN_DRAG;
    state.points = wide && tall ? box : [];
    drawZone();
    // Releasing the drag is the moment the zone is decided, so the step folds to its summary
    // here and nowhere else: collapsing mid-drag would snatch the controls away mid-thought.
    if (state.points.length) openStep('step-zone', false);
  }));

canvas.addEventListener('click', (event) => {
  if (state.running || state.mode !== 'polygon') return;
  state.points.push(pointAt(event));
  drawZone();
});
canvas.addEventListener('contextmenu', (event) => {
  event.preventDefault();
  if (state.mode === 'polygon') state.points.pop();
  else state.points = [];
  drawZone();
});
$('zone-undo').addEventListener('click', () => { state.points.pop(); drawZone(); });
$('zone-clear').addEventListener('click', () => { state.points = []; drawZone(); });

function drawZone(preview) {
  const points = preview || state.points;
  $('zone-status').textContent = points.length
    ? (state.mode === 'rect' ? 'Zone set. Drag again to replace it.'
                             : `${points.length} corners placed. Three or more makes a zone.`)
    : 'No zone set. The whole frame is watched.';

  // Only a finished polygon counts, and the summary is its share of the frame: "38% of frame"
  // says what was marked, where "4 points" only says something was.
  if (!preview) {
    const settled = points.length >= 3;
    markStep('step-zone', settled,
      settled ? `${Math.round(zoneArea(points) * 100)}% of frame` : 'Whole frame');
  }
  $('zone-hint').classList.toggle('hidden', points.length > 0 || state.running);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!points.length) return;

  const pixels = points.map(([x, y]) => [x * canvas.width, y * canvas.height]);
  ctx.beginPath();
  pixels.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  if (pixels.length > 2) {
    ctx.closePath();
    ctx.fillStyle = 'rgba(217, 159, 36, 0.18)';
    ctx.fill();
  }
  ctx.strokeStyle = '#d99f24';
  ctx.lineWidth = 2;
  ctx.stroke();

  pixels.forEach(([x, y]) => {
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = '#d99f24';
    ctx.fill();
    ctx.strokeStyle = '#0d1117';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

/* ------------------------------------------------------------- settings */

const dwell = $('dwell');
const memory = $('memory');
const conf = $('conf');
dwell.addEventListener('input', () => ($('dwell-out').textContent = `${(+dwell.value).toFixed(1)} s`));
memory.addEventListener('input', () => ($('memory-out').textContent = `${(+memory.value).toFixed(1)} s`));
conf.addEventListener('input', () => ($('conf-out').textContent = (+conf.value).toFixed(2)));

/* ------------------------------------------------------------------ run */

$('start').addEventListener('click', async () => {
  const required = [];
  if ($('req-helmet').checked) required.push('helmet');
  if ($('req-vest').checked) required.push('vest');
  if (!required.length) return say('Require at least one item of PPE.', true);
  if (state.points.length && state.points.length < 3) {
    return say('A zone needs three points or more, or clear it to watch the whole frame.', true);
  }

  const response = await fetch('/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      zone: state.points,
      zone_name: 'zone',
      required,
      conf: +conf.value,
      dwell: +dwell.value,
      memory: +memory.value,
      weights: $('model').value,
      device: $('device').value,
    }),
  });
  const data = await response.json();
  if (!response.ok) return say(data.error || 'could not start', true);

  state.running = true;
  state.seen.clear();
  $('rows').innerHTML = '<tr class="placeholder"><td colspan="7">Watching…</td></tr>';
  $('start').classList.add('hidden');
  $('stop').classList.remove('hidden');
  $('canvas-wrap').classList.add('hidden');
  $('live').classList.remove('hidden');
  $('live').src = `/api/stream?t=${Date.now()}`;
  setPill('running', 'Monitoring');
});

$('stop').addEventListener('click', async () => {
  $('stop').disabled = true;
  await fetch('/api/stop', { method: 'POST' });
});

/* --------------------------------------------------------------- status */

setInterval(async () => {
  let status;
  try {
    status = await (await fetch('/api/status')).json();
  } catch { return; }

  // A reload should not lose the session: if the server still holds a source, put it back on
  // screen rather than showing an empty console over a run that is happily going.
  if (!state.sourceReady && status.source && status.source.label) {
    showSource(status.source, true);
    if (status.running) {
      state.running = true;
      $('start').classList.add('hidden');
      $('stop').classList.remove('hidden');
      $('canvas-wrap').classList.add('hidden');
      $('live').classList.remove('hidden');
      $('live').src = `/api/stream?t=${Date.now()}`;
      setPill('running', 'Monitoring');
    }
  }

  $('kpi-frames').textContent = status.frames.toLocaleString();
  $('kpi-people').textContent = status.people.toLocaleString();
  $('kpi-alerts').textContent = status.alerts.toLocaleString();
  $('kpi-breach').textContent = status.in_breach;
  $('kpi-breach-tile').classList.toggle('live', status.in_breach > 0 && status.running);
  $('kpi-fps').textContent = status.latency?.median_fps ? status.latency.median_fps.toFixed(1) : '-';

  // No breach banner here. The frame draws its own, naming who is in breach and what each of
  // them is missing, and it travels with the evidence frames and the recording. A second one
  // laid over the top said less and hid the first.

  addRows(status.violations);

  if (state.running && !status.running) {
    state.running = false;
    $('stop').classList.add('hidden');
    $('stop').disabled = false;
    $('start').classList.remove('hidden');
    showEditor();  // hand the zone back, with the polygon still on it
    if (status.error) {
      setPill('error', 'Failed');
      say(status.error, true);
    } else {
      setPill('done', `Finished · ${status.alerts} logged`);
    }
  }
}, 500);

function addRows(violations) {
  if (!violations.length) return;
  const body = $('rows');
  const placeholder = body.querySelector('.placeholder');
  violations.forEach((violation) => {
    const key = `${violation.frame}-${violation.track_id}`;
    if (state.seen.has(key)) return;
    state.seen.add(key);
    if (placeholder) placeholder.remove();

    const snapshot = violation.snapshot ? violation.snapshot.split(/[\\/]/).pop() : '';
    const row = document.createElement('tr');
    row.innerHTML = `
      <td class="mono">${violation.seconds.toFixed(2)} s</td>
      <td class="mono">${violation.frame}</td>
      <td class="mono">#${violation.track_id}</td>
      <td>${escapeHtml(violation.zone)}</td>
      <td class="missing">${escapeHtml(violation.missing_ppe)}</td>
      <td class="mono">${violation.dwell_seconds.toFixed(1)} s</td>
      <td>${snapshot ? `<a href="/api/snapshot/${encodeURIComponent(snapshot)}" target="_blank">
            <img src="/api/snapshot/${encodeURIComponent(snapshot)}" alt="Evidence frame"></a>` : '-'}</td>`;
    body.prepend(row);
  });
}

/* ---------------------------------------------------------------- utils */

function setPill(kind, text) {
  const pill = $('status-pill');
  pill.className = `pill ${kind}`;
  pill.textContent = text;
}

function say(message, isError) {
  if (isError) setPill('error', 'Problem');
  $('source-label').textContent = message;
}

// Undo belongs to the click-a-corner mode; a dragged box is replaced by dragging again.
$('zone-undo').disabled = true;
updateRulesSummary();

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
