/* ── Datasets state ─────────────────────────────────────────────────────────── */

const _ds = {
  dirId:    null,
  dataset:  null,   // full dataset object from /api/datasets
  page:     1,
  pageSize: 60,
  cls:      null,
  total:    0,
  pages:    1,
  classes:  [],
};

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

    return `
      <tr class="ds-row" onclick="navigate('${rowHref}')">
        <td><strong>${escHtml(ds.name)}</strong></td>
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

  Object.assign(_ds, { dirId, dataset, page: 1, cls: null, total: 0, pages: 1, classes: [] });

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
      <span class="ds-upload-status" id="ds-upload-status"></span>
      <span class="ds-count" id="ds-count"></span>
    </div>
    <div id="ds-pagination-top" class="ds-pagination"></div>
    <div id="ds-grid" class="ds-grid"></div>
    <div id="ds-pagination-bottom" class="ds-pagination"></div>
  `;
}

async function loadGalleryPage() {
  const grid     = document.getElementById('ds-grid');
  const countEl  = document.getElementById('ds-count');
  if (!grid) return;

  grid.innerHTML = buildSkeletons(_ds.pageSize);

  const params = new URLSearchParams({ page: _ds.page, page_size: _ds.pageSize });
  if (_ds.cls) params.set('class', _ds.cls);

  let data;
  try {
    data = await api(`/datasets/${encodeURIComponent(_ds.dirId)}/images?${params}`);
  } catch (e) {
    grid.innerHTML = `<p class="error-msg">Failed to load images: ${escHtml(e.message)}</p>`;
    return;
  }

  _ds.total   = data.total;
  _ds.pages   = data.pages;
  _ds.classes = data.classes;

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

  renderPagination();
}

function buildSkeletons(n) {
  return Array.from({ length: n }, () =>
    `<div class="ds-tile ds-tile--skeleton"><div class="ds-tile-img-wrap"></div><div class="ds-tile-label"></div></div>`
  ).join('');
}

function buildTile(item) {
  const imgUrl  = `/api/datasets/${encodeURIComponent(_ds.dirId)}/file/${encodeURIComponent(item.path).replace(/%2F/g, '/')}`;
  const label   = escHtml(item.class || item.filename);
  const safeP   = escHtml(item.path);
  return `
    <div class="ds-tile" data-path="${safeP}">
      <div class="ds-tile-img-wrap">
        <img src="${imgUrl}" loading="lazy" alt="${escHtml(item.filename)}" onclick="openModal('${imgUrl}')">
        <button class="ds-delete-btn" title="Delete image" onclick="dsStartDelete(this)" data-path="${safeP}">×</button>
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
    const countEl = document.getElementById('ds-count');
    if (countEl) {
      const start = (_ds.page - 1) * _ds.pageSize + 1;
      const end   = Math.min(_ds.page * _ds.pageSize, _ds.total);
      countEl.textContent = _ds.total > 0
        ? `${start}–${end} of ${_ds.total.toLocaleString()} images`
        : '0 images';
    }
  }, { once: true });
}

// Clicking outside a confirm tile cancels it
document.addEventListener('click', e => {
  if (!e.target.closest('.ds-tile--confirm')) {
    document.querySelectorAll('.ds-tile--confirm').forEach(t => t.classList.remove('ds-tile--confirm'));
  }
});
