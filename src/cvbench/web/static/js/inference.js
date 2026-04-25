/* ── State ─────────────────────────────────────────────────────────────────── */

let _inferRuns       = [];
let _inferAugSchema  = [];
let _inferFile       = null;
let _inferAugments   = [];
let _inferAugCounter = 0;
let _inferOrigResult = null;
let _inferAugResult  = null;
let _inferOrigKey    = null;

/* ── Page entry ────────────────────────────────────────────────────────────── */

async function showInference() {
  setActive('nav-inference');
  const page = document.getElementById('page');
  page.innerHTML = '<p aria-busy="true">Loading…</p>';

  try {
    [_inferRuns, _inferAugSchema] = await Promise.all([
      api('/runs'),
      api('/augmentations'),
    ]);
  } catch (e) {
    page.innerHTML = `<p class="error-msg">Failed to load: ${e.message}</p>`;
    return;
  }

  _inferFile       = null;
  _inferAugments   = [];
  _inferAugCounter = 0;
  _inferOrigResult = null;
  _inferAugResult  = null;

  page.innerHTML = buildInferencePage();
  _initDropZone();
}

function buildInferencePage() {
  const runOptions = _inferRuns.length
    ? _inferRuns.map(r =>
        `<option value="${escHtml(r.name)}">${escHtml(r.name)} (${escHtml(r.backbone)}, ${fmtAcc(r.val_accuracy)})</option>`
      ).join('')
    : '<option value="">No runs available</option>';

  return `
    <div class="page-header"><h2>Inference</h2></div>
    <div class="inference-layout">

      <!-- LEFT PANEL -->
      <div class="inference-left">

        <div class="inference-section">
          <p class="section-label">Model</p>
          <select id="inference-run-select" style="margin:0">
            ${runOptions}
          </select>
        </div>

        <div class="inference-section">
          <p class="section-label">Image</p>
          <div class="drop-zone" id="inference-drop-zone" onclick="document.getElementById('inference-file-input').click()">
            <div class="drop-inner" id="drop-inner">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
              <span>Drop image or click to browse</span>
            </div>
            <input type="file" id="inference-file-input" accept="image/*" style="display:none"
                   onchange="onInferFileSelected(this.files[0])">
          </div>
        </div>

        <div class="inference-section">
          <div class="inference-aug-header">
            <p class="section-label" style="margin:0">Augmentations</p>
            <div style="position:relative">
              <button class="inference-add-btn outline" onclick="toggleAugDropdown(event)">+ Add</button>
              <div class="aug-dropdown" id="aug-dropdown" style="display:none">
                ${_inferAugSchema.map(a =>
                  `<div class="aug-dropdown-item" onclick="addAugmentation('${escHtml(a.name)}')">${escHtml(a.label)}</div>`
                ).join('')}
              </div>
            </div>
          </div>
          <div id="aug-stack"></div>
        </div>

        <button id="inference-run-btn" class="inference-run-btn" onclick="runInference()" disabled>
          ▶ Run
        </button>

      </div>

      <!-- RIGHT PANEL -->
      <div class="inference-right" id="inference-right">
        <div class="inference-placeholder">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3" aria-hidden="true">
            <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>
            <polyline points="21 15 16 10 5 21"/>
          </svg>
          <p>Drop an image and press Run to see predictions</p>
        </div>
      </div>

    </div>
  `;
}

/* ── Drop zone ─────────────────────────────────────────────────────────────── */

function _initDropZone() {
  const zone = document.getElementById('inference-drop-zone');
  if (!zone) return;

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) onInferFileSelected(f);
  });
}

function onInferFileSelected(file) {
  if (!file) return;
  _inferFile = file;

  const reader = new FileReader();
  reader.onload = ev => {
    const inner = document.getElementById('drop-inner');
    if (inner) inner.innerHTML = `
      <img src="${ev.target.result}" class="drop-preview" alt="preview">
      <span class="drop-filename">${escHtml(file.name)}</span>
    `;
  };
  reader.readAsDataURL(file);

  const btn = document.getElementById('inference-run-btn');
  if (btn) btn.disabled = false;

  _inferOrigResult = null;
  _inferAugResult  = null;
  _inferOrigKey    = null;
  _renderRight();
}

/* ── Augmentation stack ────────────────────────────────────────────────────── */

function toggleAugDropdown(e) {
  e.stopPropagation();
  const dd = document.getElementById('aug-dropdown');
  if (!dd) return;
  dd.style.display = dd.style.display === 'none' ? 'block' : 'none';
  const close = () => { dd.style.display = 'none'; document.removeEventListener('click', close); };
  if (dd.style.display === 'block') setTimeout(() => document.addEventListener('click', close), 0);
}

function addAugmentation(name) {
  const schema = _inferAugSchema.find(a => a.name === name);
  if (!schema) return;

  const id = ++_inferAugCounter;
  const params = {};
  schema.params.forEach(p => { params[p.name] = p.default; });
  _inferAugments.push({ id, name, label: schema.label, params });
  _renderAugStack();
}

function removeAugmentation(id) {
  _inferAugments = _inferAugments.filter(a => a.id !== id);
  _renderAugStack();
}

function updateAugParam(id, paramName, value) {
  const aug = _inferAugments.find(a => a.id === id);
  if (!aug) return;
  const schema = _inferAugSchema.find(a => a.name === aug.name);
  const pSchema = schema?.params.find(p => p.name === paramName);
  aug.params[paramName] = pSchema?.type === 'int' ? parseInt(value, 10) : pSchema?.type === 'float' ? parseFloat(value) : value;
  const label = document.getElementById(`pval-${id}-${paramName}`);
  if (label) label.textContent = aug.params[paramName];
}

function _renderAugStack() {
  const stack = document.getElementById('aug-stack');
  if (!stack) return;

  if (_inferAugments.length === 0) {
    stack.innerHTML = '<p class="aug-empty">No augmentations added.</p>';
    return;
  }

  stack.innerHTML = _inferAugments.map(aug => {
    const schema = _inferAugSchema.find(a => a.name === aug.name);
    const controls = (schema?.params || []).map(p => {
      const val = aug.params[p.name];
      if (p.type === 'choice') {
        const opts = p.choices.map(c => `<option value="${c}" ${c === val ? 'selected' : ''}>${c}</option>`).join('');
        return `
          <div class="aug-param-row">
            <label class="aug-param-label">${p.name}</label>
            <select class="aug-param-select"
                    onchange="updateAugParam(${aug.id}, '${p.name}', this.value)">
              ${opts}
            </select>
          </div>
        `;
      }
      return `
        <div class="aug-param-row">
          <label class="aug-param-label">${p.name}</label>
          <input type="range" class="aug-param-slider"
                 min="${p.min}" max="${p.max}" step="${p.step}" value="${val}"
                 oninput="updateAugParam(${aug.id}, '${p.name}', this.value)">
          <span class="aug-param-val" id="pval-${aug.id}-${p.name}">${val}</span>
        </div>
      `;
    }).join('');

    return `
      <div class="aug-card">
        <div class="aug-card-header">
          <span class="aug-card-label">${escHtml(aug.label)}</span>
          <button class="aug-remove-btn" title="Remove" onclick="removeAugmentation(${aug.id})">✕</button>
        </div>
        ${controls}
      </div>
    `;
  }).join('');
}

/* ── Run prediction ────────────────────────────────────────────────────────── */

async function runInference() {
  if (!_inferFile) return;

  const run = document.getElementById('inference-run-select')?.value;
  if (!run) return;

  const btn = document.getElementById('inference-run-btn');
  if (btn) { btn.disabled = true; btn.textContent = '…'; }

  const hasAug = _inferAugments.length > 0;
  const origKey = `${run}::${_inferFile.name}::${_inferFile.size}`;
  const origCached = _inferOrigResult && _inferOrigKey === origKey;

  _inferAugResult = null;
  _renderPendingGrid(hasAug, origCached);

  try {
    if (!origCached) {
      _inferOrigResult = null;
      const fd = new FormData();
      fd.append('file', _inferFile);
      fd.append('run', run);

      const origRes = await fetch('/api/predict/single', { method: 'POST', body: fd });
      if (!origRes.ok) {
        const err = await origRes.json().catch(() => ({ detail: origRes.statusText }));
        throw new Error(err.detail || origRes.statusText);
      }
      _inferOrigResult = await origRes.json();
      _inferOrigKey    = origKey;
      _updateOrigPanel();
    }

    if (hasAug) {
      const augPayload = _inferAugments.map(a => ({ name: a.name, params: a.params }));
      const fd2 = new FormData();
      fd2.append('file', _inferFile);
      fd2.append('run', run);
      fd2.append('augmentations', JSON.stringify(augPayload));

      const augRes = await fetch('/api/predict/augmented', { method: 'POST', body: fd2 });
      if (!augRes.ok) {
        const err = await augRes.json().catch(() => ({ detail: augRes.statusText }));
        throw new Error(err.detail || augRes.statusText);
      }
      _inferAugResult = await augRes.json();
      _updateAugPanel();
    }
  } catch (e) {
    document.getElementById('inference-right').innerHTML =
      `<p class="error-msg">${escHtml(e.message)}</p>`;
    if (btn) { btn.disabled = false; btn.textContent = '▶ Run'; }
    return;
  }

  if (btn) { btn.disabled = false; btn.textContent = '▶ Run'; }
}

/* ── Right panel rendering ─────────────────────────────────────────────────── */

function _renderRight() {
  const right = document.getElementById('inference-right');
  if (!right) return;
  right.innerHTML = `
    <div class="inference-placeholder">
      <p>Drop an image and press Run to see predictions</p>
    </div>`;
}

function _pendingSlot(label) {
  return `
    <div class="inference-result-panel">
      <p class="section-label">${label}</p>
      <div class="aug-processing-placeholder">
        <div class="aug-processing-spinner"></div>
        <span>Processing…</span>
      </div>
    </div>`;
}

function _renderPendingGrid(hasAug, origCached) {
  const right = document.getElementById('inference-right');
  if (!right) return;
  const origContent = origCached
    ? `<div class="inference-result-panel"><p class="section-label">Original</p>${buildResultPanel('orig', _inferOrigResult, null)}</div>`
    : _pendingSlot('Original');
  right.innerHTML = `
    <div class="inference-results-grid ${hasAug ? 'has-aug' : ''}">
      <div id="inference-orig-slot">${origContent}</div>
      ${hasAug ? `<div id="inference-aug-slot">${_pendingSlot('Augmented')}</div>` : '<div id="inference-aug-slot"></div>'}
    </div>`;
}

function _updateOrigPanel() {
  const slot = document.getElementById('inference-orig-slot');
  if (!slot || !_inferOrigResult) return;
  slot.innerHTML = `
    <div class="inference-result-panel">
      <p class="section-label">Original</p>
      ${buildResultPanel('orig', _inferOrigResult, null)}
    </div>`;
}

function _updateAugPanel() {
  if (!_inferAugResult) return;
  const grid = document.querySelector('.inference-results-grid');
  if (grid) grid.classList.add('has-aug');
  const flipped = _inferAugResult.class_name !== _inferOrigResult.class_name;
  const slot = document.getElementById('inference-aug-slot');
  if (!slot) return;
  slot.innerHTML = `
    <div class="inference-result-panel ${flipped ? 'result-flipped' : ''}">
      <p class="section-label">
        Augmented
        ${flipped ? '<span class="flip-badge">⚠ prediction changed</span>' : ''}
      </p>
      ${buildResultPanel('aug', _inferAugResult, _inferAugResult.augmented_image_b64)}
    </div>`;
}

function buildResultPanel(panelId, result, augImageB64) {
  const topK = (result.top_k || []).slice(0, 5);
  const maxConf = topK.length ? topK[0].confidence : 1;

  const bars = topK.map((item, i) => {
    const pct = (item.confidence / maxConf * 100).toFixed(1);
    const isTop = i === 0;
    return `
      <div class="topk-row">
        <span class="topk-label ${isTop ? 'topk-top' : ''}">${escHtml(item.class_name)}</span>
        <div class="topk-bar-wrap">
          <div class="topk-bar ${isTop ? 'topk-bar-top' : ''}" style="width:${pct}%"></div>
        </div>
        <span class="topk-pct">${(item.confidence * 100).toFixed(1)}%</span>
      </div>
    `;
  }).join('');

  const imgSrc = augImageB64
    ? `data:image/png;base64,${augImageB64}`
    : (_inferFile ? URL.createObjectURL(_inferFile) : '');

  return `
    <div class="result-image-wrap">
      <img src="${imgSrc}" class="result-image" alt="input image"
           onclick="openModal('${imgSrc}')" id="result-img-${panelId}">
    </div>
    <div class="result-topk">${bars}</div>
    <div class="result-explain-row">
      <button class="explain-btn outline"
              onclick="explainPrediction(event, '${panelId}', ${result.class_index})">
        Explain
      </button>
      <span class="explain-error" id="explain-err-${panelId}"></span>
    </div>
  `;
}

/* ── XAI for inference page ─────────────────────────────────────────────────── */

async function explainPrediction(event, panelId, classIndex) {
  const btn = event.target;
  const errEl = document.getElementById(`explain-err-${panelId}`);
  const imgEl = document.getElementById(`result-img-${panelId}`);
  if (!imgEl) return;

  if (btn.dataset.explained === '1') {
    imgEl.src = btn.dataset.origSrc;
    imgEl.onclick = () => openModal(btn.dataset.origSrc);
    btn.textContent = 'Explain';
    btn.dataset.explained = '0';
    return;
  }
  if (btn.dataset.heatmap) {
    imgEl.src = btn.dataset.heatmap;
    imgEl.onclick = () => openModal(btn.dataset.heatmap);
    btn.textContent = 'Original';
    btn.dataset.explained = '1';
    return;
  }

  btn.dataset.origSrc = imgEl.src;
  btn.textContent = '…';
  btn.disabled = true;
  if (errEl) errEl.textContent = '';

  const run = document.getElementById('inference-run-select')?.value;
  if (!run) { btn.textContent = 'Explain'; btn.disabled = false; return; }

  let fileToSend = _inferFile;
  if (panelId === 'aug' && _inferAugResult?.augmented_image_b64) {
    const binary = atob(_inferAugResult.augmented_image_b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    fileToSend = new Blob([bytes], { type: 'image/png' });
  }

  try {
    const fd = new FormData();
    fd.append('file', fileToSend, 'image.png');
    fd.append('run', run);
    fd.append('class_index', classIndex);

    const res = await fetch('/api/predict/xai', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const { heatmap_b64 } = await res.json();
    const heatSrc = `data:image/png;base64,${heatmap_b64}`;

    btn.dataset.heatmap   = heatSrc;
    btn.dataset.explained = '1';
    imgEl.src = heatSrc;
    imgEl.onclick = () => openModal(heatSrc);
    btn.textContent = 'Original';
    btn.disabled = false;
  } catch (e) {
    btn.textContent = 'Explain';
    btn.disabled = false;
    if (errEl) errEl.textContent = e.message;
  }
}
