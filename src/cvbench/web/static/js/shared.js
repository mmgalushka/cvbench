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
function buildBoxLayer(boxes, fill) {
  const rects = boxes.map(b => {
    const isPred = b.variant === 'pred';
    const confSuffix = isPred && b.confidence != null ? ` ${(b.confidence * 100).toFixed(0)}%` : '';
    return `
    <div class="ds-box${isPred ? ' ds-box--pred' : ''}" style="left:${b.x * 100}%;top:${b.y * 100}%;
         width:${b.w * 100}%;height:${b.h * 100}%;--ds-box-color:${boxColor(b.class_id)}">
      <span class="ds-box-label">${escHtml(b.class)}${confSuffix}</span>
    </div>
  `;
  }).join('');
  return `<div class="ds-boxes${fill ? ' ds-boxes--fill' : ''}">${rects}</div>`;
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

function openModal(src, filename) {
  const caption = filename
    ? `<div class="modal-filename">
        <span>${escHtml(filename)}</span>
        <button class="outline" style="padding:0.1rem 0.5rem;font-size:0.75rem"
                onclick="navigator.clipboard.writeText('${escHtml(filename)}');this.textContent='Copied!';setTimeout(()=>this.textContent='Copy',1500)">Copy</button>
       </div>`
    : '';
  document.getElementById('modal-content').innerHTML = `<img src="${src}" />${caption}`;
  document.getElementById('modal-overlay').style.display = 'flex';
}

function closeModal() {
  document.getElementById('modal-overlay').style.display = 'none';
  document.getElementById('modal-content').innerHTML = '';
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
