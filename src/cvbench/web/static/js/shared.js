/* ── Router ────────────────────────────────────────────────────────────────── */

window.addEventListener('hashchange', route);
window.addEventListener('DOMContentLoaded', route);

function route() {
  const hash = location.hash;
  if (!hash || hash === '#' || hash === '#/') {
    showRunsList();
  } else if (hash.startsWith('#/runs/')) {
    showRunDetail(decodeURIComponent(hash.slice(7)));
  } else if (hash === '#/inference') {
    showInference();
  } else if (hash === '#/datasets') {
    showDatasetsList();
  } else if (hash.startsWith('#/datasets/')) {
    showDatasetGallery(decodeURIComponent(hash.slice(11)));
  }
}

function navigate(hash) {
  location.hash = hash;
}

/* ── API helper ────────────────────────────────────────────────────────────── */

async function api(path, { signal } = {}) {
  const res = await fetch('/api' + path, signal ? { signal } : {});
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

async function apiFetch(method, path, body) {
  const opts = { method };
  if (body !== undefined) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch('/api' + path, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

/* ── Theme ─────────────────────────────────────────────────────────────────── */

function toggleTheme(checkbox) {
  document.documentElement.setAttribute('data-theme', checkbox.checked ? 'dark' : 'light');
  localStorage.setItem('theme', checkbox.checked ? 'dark' : 'light');
}

(function initTheme() {
  const saved = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  const cb = document.getElementById('theme-toggle');
  if (cb) cb.checked = saved === 'dark';
})();

/* ── Nav active state ──────────────────────────────────────────────────────── */

function setActive(id) {
  document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));
  if (id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('active');
  }
}

/* ── Shared formatters ─────────────────────────────────────────────────────── */

function fmtAcc(v) {
  return v != null ? (v * 100).toFixed(1) + '%' : '—';
}

/* ── Utilities ─────────────────────────────────────────────────────────────── */

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function copyCliCommand(btn) {
  const cmd = btn.dataset.cmd;
  const flash = () => {
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = orig; }, 1500);
  };

  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(cmd).then(flash).catch(() => fallbackCopy(cmd, flash));
  } else {
    fallbackCopy(cmd, flash);
  }
}

function fallbackCopy(text, onSuccess) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.cssText = 'position:fixed;top:-9999px;left:-9999px;opacity:0';
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {
    if (document.execCommand('copy')) onSuccess();
  } finally {
    document.body.removeChild(ta);
  }
}

/* ── Bounding-box overlay ──────────────────────────────────────────────────────
   Shared by the dataset gallery (datasets.js) and the detection Evaluation
   tab's sample gallery (experiments.js). */

// Per-class bounding-box colours, indexed by YOLO class id.
const _BOX_COLORS = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
  '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
];

function boxColor(classId) {
  return _BOX_COLORS[((classId % _BOX_COLORS.length) + _BOX_COLORS.length) % _BOX_COLORS.length];
}

// Renders BOXES as absolutely-positioned divs over a fitBoxOverlay()'d image.
// Each box is {class_id, class, x, y, w, h}, normalized 0..1, top-left origin.
// A box with variant: 'pred' renders dashed (vs. the default solid ground-truth
// style) and, if it carries a `confidence`, appends it to the label — this is
// how the detection Evaluation tab overlays predictions on top of ground truth.
// A box may also carry `match` ('matched' | 'background') — the detection
// Evaluation gallery uses it to add an outcome border style — and `dim: true`
// to fade it when a confusion-matrix cell filter is active.
function buildBoxLayer(boxes, fill) {
  const rects = boxes.map(b => {
    const isPred = b.variant === 'pred';
    const confSuffix = isPred && b.confidence != null ? ` ${(b.confidence * 100).toFixed(0)}%` : '';
    const matchCls = b.match ? ` ds-box--m-${b.match}` : '';
    const dimCls = b.dim ? ' ds-box--dim' : '';
    return `
    <div class="ds-box${isPred ? ' ds-box--pred' : ''}${matchCls}${dimCls}" style="left:${b.x * 100}%;top:${b.y * 100}%;
         width:${b.w * 100}%;height:${b.h * 100}%;--ds-box-color:${boxColor(b.class_id)}">
      <span class="ds-box-label">${escHtml(b.class)}${confSuffix}</span>
    </div>
  `;
  }).join('');
  return `<div class="ds-boxes${fill ? ' ds-boxes--fill' : ''}">${rects}</div>`;
}

// Legend for the detection Evaluation gallery's box encoding. The match outcome
// is conveyed by the confusion-matrix cell you clicked (and by dimming), so the
// legend only needs to explain the visual encoding of the boxes themselves.
// `vertical` stacks the items into a column — used in the zoom modal's side
// panel, where a horizontal legend would eat the width the image wants.
function buildBoxLegend(vertical) {
  return `<div class="ds-box-legend${vertical ? ' ds-box-legend--vertical' : ''}">
    <span class="ds-box-legend-item"><span class="ds-legend-swatch ds-legend-swatch--solid"></span> ground truth</span>
    <span class="ds-box-legend-item"><span class="ds-legend-swatch ds-legend-swatch--dashed"></span> prediction</span>
    <span class="ds-box-legend-item"><span class="ds-legend-swatch ds-legend-swatch--dotted"></span> unmatched (background)</span>
    <span class="ds-box-legend-item">box colour = class</span>
  </div>`;
}

// Tiles/frames are fixed but images are not, so the overlay has to be sized to
// the letterboxed ("object-fit: contain") image rather than to the wrapper.
function fitBoxOverlay(img) {
  const layer = img.parentElement && img.parentElement.querySelector('.ds-boxes');
  if (!layer || !img.naturalWidth || !img.naturalHeight) return;
  const cw = img.clientWidth;
  const ch = img.clientHeight;
  const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
  const w = img.naturalWidth * scale;
  const h = img.naturalHeight * scale;
  layer.style.left   = `${(cw - w) / 2}px`;
  layer.style.top    = `${(ch - h) / 2}px`;
  layer.style.width  = `${w}px`;
  layer.style.height = `${h}px`;
}

function refitBoxOverlays() {
  document.querySelectorAll('.ds-tile-img--contain').forEach(fitBoxOverlay);
}

window.addEventListener('resize', refitBoxOverlays);

/* ── Image modal ───────────────────────────────────────────────────────────── */

// State for the zoom modal's per-layer box toggles. `_modalBoxes` holds every
// box passed to openModal(); the GT/prediction checkboxes filter it in place
// without reopening the modal.
let _modalBoxes = [];
let _modalSrc = '';
const _modalLayers = { gt: true, pred: true };

function _modalVisibleBoxes() {
  return _modalBoxes.filter(b => (b.variant === 'pred' ? _modalLayers.pred : _modalLayers.gt));
}

function _renderModalBoxLayer() {
  const wrap = document.querySelector('#modal-content .ds-modal-img-wrap');
  if (!wrap) return;
  const old = wrap.querySelector('.ds-boxes');
  if (old) old.remove();
  wrap.insertAdjacentHTML('beforeend', buildBoxLayer(_modalVisibleBoxes(), false));
  const img = wrap.querySelector('img');
  if (img) fitBoxOverlay(img);
}

function toggleModalLayer(which, on) {
  _modalLayers[which] = on;
  _renderModalBoxLayer();
}

// Per-image detection summary for the zoom modal's side panel: box-match
// counts plus mean IoU / confidence, the image-level view of what the gallery
// caption shows per cell.
function _modalMetricsPanel() {
  const preds = _modalBoxes.filter(b => b.variant === 'pred');
  const gts   = _modalBoxes.filter(b => b.variant !== 'pred');
  if (preds.length === 0 && gts.length === 0) return '';

  const matched  = preds.filter(b => b.match === 'matched');
  const spurious = preds.filter(b => b.match === 'background');
  const missed   = gts.filter(b => b.match === 'background');
  const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

  const rows = [
    ['Matched', matched.length],
    ['Spurious', spurious.length],
    ['Missed', missed.length],
  ];
  if (matched.length) rows.push(['Mean IoU', mean(matched.map(b => b.iou || 0)).toFixed(2)]);
  if (preds.length)   rows.push(['Mean conf', (mean(preds.map(b => b.confidence || 0)) * 100).toFixed(0) + '%']);

  return `<div class="ds-modal-metrics">
    <span class="ds-modal-side-title">This image</span>
    ${rows.map(([k, v]) => `<div class="ds-modal-metric"><span>${k}</span><strong>${v}</strong></div>`).join('')}
  </div>`;
}

function openModal(src, filename, boxes) {
  _modalBoxes = Array.isArray(boxes) ? boxes : [];
  _modalSrc = src;
  const hasBoxes = _modalBoxes.length > 0;
  const hasGt = _modalBoxes.some(b => b.variant !== 'pred');
  const hasPred = _modalBoxes.some(b => b.variant === 'pred');

  const caption = filename
    ? `<div class="modal-filename">
        <span>${escHtml(filename)}</span>
        <button class="outline" style="padding:0.1rem 0.5rem;font-size:0.75rem"
                onclick="navigator.clipboard.writeText('${escHtml(filename)}');this.textContent='Copied!';setTimeout(()=>this.textContent='Copy',1500)">Copy</button>
       </div>`
    : '';

  if (!hasBoxes) {
    document.getElementById('modal-content').innerHTML = `<img src="${src}" />${caption}`;
    document.getElementById('modal-overlay').style.display = 'flex';
    return;
  }

  const toggle = (which, label, present) => present
    ? `<label class="ds-modal-toggle"><input type="checkbox" ${_modalLayers[which] ? 'checked' : ''}
         onchange="toggleModalLayer('${which}', this.checked)"> ${label}</label>`
    : '';

  document.getElementById('modal-content').innerHTML = `
    <div class="ds-modal">
      <div class="ds-modal-img-wrap ds-modal-img-wrap--zoom">
        <img class="ds-tile-img--contain" src="${src}" onload="fitBoxOverlay(this)" />
        ${buildBoxLayer(_modalVisibleBoxes(), false)}
      </div>
      <aside class="ds-modal-side">
        <div class="ds-modal-layers">
          <span class="ds-modal-side-title">Show</span>
          ${toggle('gt', 'Ground truth', hasGt)}
          ${toggle('pred', 'Predictions', hasPred)}
        </div>
        ${_modalMetricsPanel()}
        ${buildBoxLegend(true)}
        ${caption}
      </aside>
    </div>`;
  document.getElementById('modal-overlay').style.display = 'flex';
  requestAnimationFrame(refitBoxOverlays);
}

function closeModal() {
  document.getElementById('modal-overlay').style.display = 'none';
  document.getElementById('modal-content').innerHTML = '';
  _modalBoxes = [];
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
