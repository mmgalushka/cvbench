/* ── Datasets state ─────────────────────────────────────────────────────────── */

const _ds = {
  dirId:     null,
  dataset:   null,   // full dataset object from /api/datasets
  page:      1,
  pageSize:  60,
  cls:       null,
  total:     0,
  pages:     1,
  classes:   [],
  format:    'classification',
  showBoxes: true,
};

// Per-class bounding-box colours, indexed by YOLO class id.
const _BOX_COLORS = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
  '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
];

function boxColor(classId) {
  return _BOX_COLORS[((classId % _BOX_COLORS.length) + _BOX_COLORS.length) % _BOX_COLORS.length];
}

let _loadGen   = 0;    // incremented on every loadGalleryPage call; stale responses check this
let _loadAbort = null; // AbortController for the in-flight loadGalleryPage request

/* ── Datasets list ──────────────────────────────────────────────────────────── */

async function showDatasetsList() {
  setActive('nav-datasets');
  const page = document.getElementById('page');
  page.innerHTML = '<p aria-busy="true">Loading datasets…</p>';
  try {
    const datasets = await api('/datasets');
    page.innerHTML = buildDatasetsList(datasets);
  } catch (e) {
    page.innerHTML = `<p class="error-msg">Failed to load datasets: ${escHtml(e.message)}</p>`;
  }
}

function buildDatasetsList(datasets) {
  if (datasets.length === 0) {
    return `
      <div class="page-header"><h2>Datasets</h2></div>
      <article><p>No datasets found. Train a model first to register a dataset.</p></article>
    `;
  }

  const rows = datasets.map(ds => {
    const splitBadges = Object.entries(ds.splits).map(([name, info]) => `
      <a class="badge badge-split"
         href="#/datasets/${encodeURIComponent(info.id)}"
         data-ds-id="${escHtml(JSON.stringify(ds))}"
         onclick="event.stopPropagation()"
         title="Browse ${name} split">${name}</a>
    `).join('');

    const defaultSplit = ds.splits.train || ds.splits.val || ds.splits.test;
    const rowHref = defaultSplit
      ? `#/datasets/${encodeURIComponent(defaultSplit.id)}`
      : '#/datasets';

    const fmt = `<span class="badge badge-format">${ds.format === 'yolo' ? 'YOLO' : 'classification'}</span>`;

    return `
      <tr class="ds-row" onclick="navigate('${rowHref}')">
        <td><strong>${escHtml(ds.name)}</strong></td>
        <td>${fmt}</td>
        <td>${ds.num_classes}</td>
        <td class="ds-splits-cell">${splitBadges || '—'}</td>
        <td class="ds-path-cell" title="${escHtml(ds.path)}">${escHtml(ds.path)}</td>
      </tr>
    `;
  }).join('');

  return `
    <div class="page-header">
      <h2>Datasets <small style="font-size:0.85rem;font-weight:400;color:var(--pico-muted-color)">${datasets.length} dataset${datasets.length !== 1 ? 's' : ''}</small></h2>
    </div>
    <div class="overflow-x">
      <table class="runs-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Format</th>
            <th>Classes</th>
            <th>Splits</th>
            <th>Path</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

/* ── Dataset gallery ────────────────────────────────────────────────────────── */

async function showDatasetGallery(dirId) {
  setActive('nav-datasets');
  const page = document.getElementById('page');
  page.innerHTML = '<p aria-busy="true">Loading gallery…</p>';

  // Load dataset context for split tabs and name display
  let dataset = null;
  try {
    const datasets = await api('/datasets');
    outer: for (const ds of datasets) {
      for (const info of Object.values(ds.splits)) {
        if (info.id === dirId) { dataset = ds; break outer; }
      }
    }
  } catch (_) { /* optional context */ }

  Object.assign(_ds, {
    dirId, dataset, page: 1, cls: null, total: 0, pages: 1, classes: [],
    format: (dataset && dataset.format) || 'classification',
  });

  page.innerHTML = buildGalleryShell(dataset, dirId);
  await loadGalleryPage();
}

function buildGalleryShell(dataset, dirId) {
  const name    = dataset ? escHtml(dataset.name) : 'Gallery';
  const dirName = dirId;  // shown as fallback; will be replaced once images load

  // Split tabs
  let splitTabs = '';
  if (dataset && Object.keys(dataset.splits).length > 1) {
    const tabs = Object.entries(dataset.splits).map(([sname, info]) => {
      const active = info.id === dirId ? 'active' : '';
      return `<button class="tab-btn ${active}"
                onclick="navigate('#/datasets/${encodeURIComponent(info.id)}')">${sname}</button>`;
    }).join('');
    splitTabs = `<div class="tabs" style="margin-bottom:0.75rem">${tabs}</div>`;
  }

  return `
    <div class="page-header">
      <a href="#/datasets" class="back-link" onclick="navigate('#/datasets');return false;">← Datasets</a>
      <h2>${name}</h2>
    </div>
    ${splitTabs}
    <div class="ds-filter-bar" id="ds-filter-bar">
      <select id="ds-class-select" onchange="dsFilterClass(this.value)">
        <option value="">All classes</option>
      </select>
      <button class="ds-upload-btn" onclick="dsOpenUpload()">+ Add Images</button>
      <input type="file" id="ds-file-input" multiple accept="image/*" style="display:none" onchange="dsHandleUpload(this.files)">
      <label class="ds-boxes-toggle" id="ds-boxes-toggle" style="display:none">
        <input type="checkbox" id="ds-boxes-check" checked onchange="dsToggleBoxes(this.checked)">
        Show boxes
      </label>
      <span class="ds-upload-status" id="ds-upload-status"></span>
      <span class="ds-count" id="ds-count"></span>
    </div>
    <div id="ds-pagination-top" class="ds-pagination"></div>
    <div id="ds-grid" class="ds-grid"></div>
    <div id="ds-pagination-bottom" class="ds-pagination"></div>
  `;
}

async function loadGalleryPage() {
  const gen = ++_loadGen;

  // Cancel any in-flight request so we don't queue up stale fetches
  if (_loadAbort) _loadAbort.abort();
  const controller = new AbortController();
  _loadAbort = controller;

  const grid    = document.getElementById('ds-grid');
  const countEl = document.getElementById('ds-count');
  if (!grid) return;

  grid.innerHTML = buildSkeletons(_ds.pageSize);

  const params = new URLSearchParams({ page: _ds.page, page_size: _ds.pageSize });
  if (_ds.cls) params.set('class', _ds.cls);

  let data;
  try {
    data = await api(`/datasets/${encodeURIComponent(_ds.dirId)}/images?${params}`, { signal: controller.signal });
  } catch (e) {
    if (controller.signal.aborted || gen !== _loadGen) return;
    grid.innerHTML = `<p class="error-msg">Failed to load images: ${escHtml(e.message)}</p>`;
    return;
  }

  if (gen !== _loadGen) return;
  _loadAbort = null;

  _ds.total   = data.total;
  _ds.pages   = data.pages;
  _ds.classes = data.classes;
  _ds.format  = data.format || _ds.format;

  const boxToggle = document.getElementById('ds-boxes-toggle');
  if (boxToggle) boxToggle.style.display = _ds.format === 'yolo' ? '' : 'none';

  // Guard: if current page is beyond the available pages (e.g. items were deleted),
  // correct it and reload — prevents the UI from getting stuck on a non-existent page.
  if (_ds.page > _ds.pages) {
    _ds.page = _ds.pages;
    loadGalleryPage();
    return;
  }

  // Populate class filter
  const sel = document.getElementById('ds-class-select');
  if (sel && sel.options.length <= 1) {
    data.classes.forEach(cls => {
      const opt = document.createElement('option');
      opt.value = cls;
      opt.textContent = cls;
      if (cls === _ds.cls) opt.selected = true;
      sel.appendChild(opt);
    });
  }

  // Count line
  if (countEl) {
    const start = (_ds.page - 1) * _ds.pageSize + 1;
    const end   = Math.min(_ds.page * _ds.pageSize, _ds.total);
    countEl.textContent = _ds.total > 0
      ? `${start}–${end} of ${_ds.total.toLocaleString()} images`
      : '0 images';
  }

  // Render tiles
  grid.innerHTML = data.items.length === 0
    ? '<p style="grid-column:1/-1;color:var(--pico-muted-color)">No images found.</p>'
    : data.items.map(item => buildTile(item)).join('');

  // Images restored from cache may not fire `load`, so fit their overlays too.
  requestAnimationFrame(refitBoxOverlays);

  renderPagination();
}

function buildSkeletons(n) {
  return Array.from({ length: n }, () =>
    `<div class="ds-tile ds-tile--skeleton"><div class="ds-tile-img-wrap"></div><div class="ds-tile-label"></div></div>`
  ).join('');
}

function boxSummary(boxes) {
  if (boxes.length === 0) return 'no objects';
  const counts = new Map();
  boxes.forEach(b => counts.set(b.class, (counts.get(b.class) || 0) + 1));
  return [...counts].map(([cls, n]) => (n > 1 ? `${cls} ×${n}` : cls)).join(', ');
}

function buildBoxLayer(boxes, fill) {
  const rects = boxes.map(b => `
    <div class="ds-box" style="left:${b.x * 100}%;top:${b.y * 100}%;
         width:${b.w * 100}%;height:${b.h * 100}%;--ds-box-color:${boxColor(b.class_id)}">
      <span class="ds-box-label">${escHtml(b.class)}</span>
    </div>
  `).join('');
  return `<div class="ds-boxes${fill ? ' ds-boxes--fill' : ''}">${rects}</div>`;
}

function buildTile(item) {
  const imgUrl  = `/api/datasets/${encodeURIComponent(_ds.dirId)}/file/${encodeURIComponent(item.path).replace(/%2F/g, '/')}`;
  const boxes   = item.boxes || [];
  const hasBox  = boxes.length > 0 || _ds.format === 'yolo';
  const label   = escHtml(item.boxes ? boxSummary(boxes) : (item.class || item.filename));
  const safeP   = escHtml(item.path);
  const hidden  = _ds.showBoxes ? '' : ' ds-tile--boxes-off';
  const onClick = hasBox ? 'dsOpenTileModal(this)' : `openModal('${imgUrl}', '${safeP}')`;
  return `
    <div class="ds-tile${hidden}" data-path="${safeP}"
         data-boxes="${escHtml(JSON.stringify(boxes))}">
      <div class="ds-tile-img-wrap">
        <img src="${imgUrl}" loading="lazy" alt="${escHtml(item.filename)}"
             class="${hasBox ? 'ds-tile-img--contain' : ''}"
             onload="${hasBox ? 'fitBoxOverlay(this)' : ''}"
             onclick="${onClick}">
        ${hasBox ? buildBoxLayer(boxes, false) : ''}
        <button class="ds-delete-btn" title="Delete image" onclick="dsStartDelete(this)" data-path="${safeP}">×</button>
        <a class="ds-download-btn" href="${imgUrl}" download="${escHtml(item.filename)}" title="Download image">↓</a>
      </div>
      <div class="ds-tile-confirm-bar">
        <span>Delete?</span>
        <div style="display:flex;gap:0.25rem">
          <button class="ds-confirm-yes" onclick="dsConfirmDelete(this)" data-path="${safeP}">Yes</button>
          <button class="ds-confirm-no"  onclick="dsCancelDelete(this)">No</button>
        </div>
      </div>
      <div class="ds-tile-label" title="${escHtml(item.filename)}">${label}</div>
    </div>
  `;
}

/* ── Bounding-box overlay ────────────────────────────────────────────────────── */

// Tiles are fixed squares while images are not, so the overlay has to be sized
// to the letterboxed ("object-fit: contain") image rather than to the wrapper.
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

function dsToggleBoxes(on) {
  _ds.showBoxes = on;
  document.querySelectorAll('.ds-tile').forEach(t => t.classList.toggle('ds-tile--boxes-off', !on));
  const modal = document.getElementById('modal-content');
  if (modal) {
    const layer = modal.querySelector('.ds-boxes');
    if (layer) layer.style.display = on ? '' : 'none';
  }
}

function dsOpenTileModal(img) {
  const tile = img.closest('.ds-tile');
  let boxes = [];
  try { boxes = JSON.parse(tile.dataset.boxes || '[]'); } catch (_) { /* no annotations */ }
  const path = tile.dataset.path;

  const layer = _ds.showBoxes ? buildBoxLayer(boxes, true) : '';
  document.getElementById('modal-content').innerHTML = `
    <div class="ds-modal-img-wrap">
      <img src="${img.src}" alt="${escHtml(path)}">
      ${layer}
    </div>
    <div class="modal-filename">
      <span>${escHtml(path)}</span>
      <span class="ds-modal-objects">${escHtml(boxSummary(boxes))}</span>
    </div>
  `;
  document.getElementById('modal-overlay').style.display = 'flex';
}

/* ── Gallery pagination ──────────────────────────────────────────────────────── */

function renderPagination() {
  const html = _ds.pages <= 1 ? '' : buildPaginationHtml();
  const top  = document.getElementById('ds-pagination-top');
  const bot  = document.getElementById('ds-pagination-bottom');
  if (top) top.innerHTML = html;
  if (bot) bot.innerHTML = html;
}

function buildPaginationHtml() {
  const { page, pages } = _ds;
  const prevDis = page <= 1    ? 'disabled' : '';
  const nextDis = page >= pages ? 'disabled' : '';

  // Show at most 7 page buttons with ellipsis
  const pageNums = buildPageNumbers(page, pages);
  const btns = pageNums.map(n =>
    n === '…'
      ? `<span class="ds-page-ellipsis">…</span>`
      : `<button class="ds-page-btn${n === page ? ' active' : ''}" onclick="dsGoPage(${n})">${n}</button>`
  ).join('');

  return `
    <button class="ds-page-btn" onclick="dsGoPage(${page - 1})" ${prevDis}>‹</button>
    ${btns}
    <button class="ds-page-btn" onclick="dsGoPage(${page + 1})" ${nextDis}>›</button>
  `;
}

function buildPageNumbers(current, total) {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const pages = new Set([1, total, current]);
  for (let d = -2; d <= 2; d++) {
    const n = current + d;
    if (n >= 1 && n <= total) pages.add(n);
  }
  const sorted = [...pages].sort((a, b) => a - b);
  const result = [];
  let prev = 0;
  for (const n of sorted) {
    if (n - prev > 1) result.push('…');
    result.push(n);
    prev = n;
  }
  return result;
}

function dsGoPage(n) {
  if (n < 1 || n > _ds.pages || n === _ds.page) return;
  _ds.page = n;
  loadGalleryPage();
  document.getElementById('page').scrollTo({ top: 0, behavior: 'smooth' });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ── Gallery filters ─────────────────────────────────────────────────────────── */

function dsFilterClass(cls) {
  _ds.cls  = cls || null;
  _ds.page = 1;
  loadGalleryPage();
}

/* ── Upload flow ─────────────────────────────────────────────────────────────── */

function dsOpenUpload() {
  const input = document.getElementById('ds-file-input');
  if (input) { input.value = ''; input.click(); }
}

async function dsHandleUpload(files) {
  if (!files || files.length === 0) return;
  const statusEl = document.getElementById('ds-upload-status');
  const total = files.length;
  let done = 0;
  let failed = 0;

  for (const file of files) {
    if (statusEl) statusEl.textContent = `Uploading ${done + 1}/${total}…`;
    const fd = new FormData();
    fd.append('files', file);
    const params = _ds.cls ? `?class=${encodeURIComponent(_ds.cls)}` : '';
    try {
      const res = await fetch(`/api/datasets/${encodeURIComponent(_ds.dirId)}/images${params}`, {
        method: 'POST',
        body: fd,
      });
      if (!res.ok) { failed++; } else { done++; }
    } catch (_) {
      failed++;
    }
  }

  if (statusEl) {
    statusEl.textContent = failed === 0
      ? `Uploaded ${done} image${done !== 1 ? 's' : ''}`
      : `${done} uploaded, ${failed} failed`;
    setTimeout(() => { statusEl.textContent = ''; }, 3000);
  }

  await loadGalleryPage();
}

/* ── Delete flow ─────────────────────────────────────────────────────────────── */

function dsStartDelete(btn) {
  // Cancel any existing confirm first
  document.querySelectorAll('.ds-tile--confirm').forEach(t => t.classList.remove('ds-tile--confirm'));
  btn.closest('.ds-tile').classList.add('ds-tile--confirm');
}

function dsCancelDelete(btn) {
  btn.closest('.ds-tile').classList.remove('ds-tile--confirm');
}

async function dsConfirmDelete(btn) {
  const tile = btn.closest('.ds-tile');
  const path = btn.dataset.path;

  btn.disabled = true;
  try {
    await apiFetch('DELETE', `/datasets/${encodeURIComponent(_ds.dirId)}/images/${path}`);
  } catch (e) {
    tile.classList.remove('ds-tile--confirm');
    alert(`Delete failed: ${e.message}`);
    return;
  }

  tile.classList.add('ds-tile--deleting');
  tile.addEventListener('animationend', () => {
    tile.remove();
    _ds.total = Math.max(0, _ds.total - 1);
    _ds.pages = Math.max(1, Math.ceil(_ds.total / _ds.pageSize));

    // If we just emptied the last page, go back one page and reload
    if (_ds.page > _ds.pages) {
      _ds.page = _ds.pages;
      loadGalleryPage();
      return;
    }

    const countEl = document.getElementById('ds-count');
    if (countEl) {
      const start = (_ds.page - 1) * _ds.pageSize + 1;
      const end   = Math.min(_ds.page * _ds.pageSize, _ds.total);
      countEl.textContent = _ds.total > 0
        ? `${start}–${end} of ${_ds.total.toLocaleString()} images`
        : '0 images';
    }
    renderPagination();
  }, { once: true });
}

// Clicking outside a confirm tile cancels it
document.addEventListener('click', e => {
  if (!e.target.closest('.ds-tile--confirm')) {
    document.querySelectorAll('.ds-tile--confirm').forEach(t => t.classList.remove('ds-tile--confirm'));
  }
});
