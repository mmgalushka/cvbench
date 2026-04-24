/* ── State ─────────────────────────────────────────────────────────────────── */

let currentRun = null;
let activeTab = 'training';
let chartInstance = null;

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
  }
}

function navigate(hash) {
  location.hash = hash;
}

/* ── API helper ────────────────────────────────────────────────────────────── */

async function api(path) {
  const res = await fetch('/api' + path);
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

/* ── Runs list ─────────────────────────────────────────────────────────────── */

async function showRunsList() {
  setActive('nav-runs');
  const page = document.getElementById('page');
  page.innerHTML = '<p aria-busy="true">Loading experiments…</p>';
  try {
    const runs = await api('/runs');
    page.innerHTML = buildRunsList(runs);
  } catch (e) {
    page.innerHTML = `<p class="error-msg">Failed to load runs: ${e.message}</p>`;
  }
}

function buildRunsList(runs) {
  if (runs.length === 0) {
    return `
      <div class="page-header"><h2>Experiments</h2></div>
      <article><p>No experiments found. Run <kbd>train</kbd> to create one.</p></article>
    `;
  }

  const rows = runs.map(r => `
    <tr onclick="navigate('#/runs/${encodeURIComponent(r.name)}')">
      <td><strong>${r.name}</strong></td>
      <td>${r.backbone}</td>
      <td>${r.date || '—'}</td>
      <td><span class="badge badge-${r.status}">${r.status}</span></td>
      <td>${r.val_accuracy != null ? (r.val_accuracy * 100).toFixed(1) + '%' : '—'}</td>
      <td>${r.test_accuracy != null ? (r.test_accuracy * 100).toFixed(1) + '%' : '—'}</td>
      <td>${r.epochs_run} / ${r.epochs}</td>
    </tr>
  `).join('');

  return `
    <div class="page-header"><h2>Experiments <small style="font-size:0.85rem;font-weight:400;color:var(--muted-color)">${runs.length} run${runs.length !== 1 ? 's' : ''}</small></h2></div>
    <div class="overflow-x">
      <table class="runs-table">
        <thead>
          <tr>
            <th>Name</th><th>Backbone</th><th>Date</th><th>Status</th>
            <th>Val Acc</th><th>Test Acc</th><th>Epochs</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

/* ── Run detail ────────────────────────────────────────────────────────────── */

async function showRunDetail(name) {
  setActive('');
  const page = document.getElementById('page');
  page.innerHTML = '<p aria-busy="true">Loading run…</p>';
  try {
    const run = await api(`/runs/${encodeURIComponent(name)}`);
    currentRun = run;
    page.innerHTML = buildRunDetail(run);
    switchTab(activeTab);
  } catch (e) {
    page.innerHTML = `<p class="error-msg">Run '${name}' not found: ${e.message}</p>`;
  }
}

function buildRunDetail(run) {
  const hasEval = run.eval_report != null;
  return `
    <div class="page-header">
      <div>
        <a href="#/" class="back-link" onclick="setActive('nav-runs')">← Experiments</a>
        <h2>${run.name} <span class="badge badge-${run.status}">${run.status}</span></h2>
      </div>
    </div>

    <div class="metric-cards">
      ${card('Val Accuracy',  fmtAcc(run.val_accuracy))}
      ${card('Test Accuracy', fmtAcc(run.test_accuracy))}
      ${card('Backbone',      run.backbone)}
      ${card('Epochs',        run.epochs_run + ' / ' + run.epochs)}
      ${card('Learning Rate', run.lr)}
      ${card('Date',          run.date || '—')}
    </div>

    <div class="tabs">
      <button class="tab-btn" id="tab-training" onclick="switchTab('training')">Training</button>
      <button class="tab-btn" id="tab-eval"     onclick="switchTab('eval')"     ${!hasEval ? 'disabled title="Run evaluation first"' : ''}>Evaluation</button>
      <button class="tab-btn" id="tab-compare"  onclick="switchTab('compare')">Compare</button>
    </div>
    <div id="tab-content"></div>
  `;
}

function card(label, value) {
  return `
    <article class="metric-card">
      <small>${label}</small>
      <strong>${value}</strong>
    </article>
  `;
}

function fmtAcc(v) {
  return v != null ? (v * 100).toFixed(1) + '%' : '—';
}

/* ── Tab switcher ──────────────────────────────────────────────────────────── */

function switchTab(tab) {
  activeTab = tab;
  ['training', 'eval', 'compare'].forEach(t => {
    const btn = document.getElementById('tab-' + t);
    if (btn) btn.classList.toggle('active', t === tab);
  });

  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }

  const content = document.getElementById('tab-content');
  if (!content || !currentRun) return;

  if (tab === 'training') {
    content.innerHTML = buildTrainingTab(currentRun);
    renderChart(currentRun);
  } else if (tab === 'eval') {
    content.innerHTML = buildEvalTab(currentRun);
  } else if (tab === 'compare') {
    content.innerHTML = buildCompareTab();
    loadCompareOptions();
  }
}

/* ── Training tab ──────────────────────────────────────────────────────────── */

function buildTrainingTab(run) {
  const augs = run.config.augmentation;
  const augText = augs.length ? augs.map(t => t.name).join(', ') : 'None';
  const hasLog = run.training_log.length > 0;

  return `
    <div class="config-grid">
      <article>
        <h4>Model</h4>
        <dl>
          <dt>Backbone</dt>      <dd>${run.config.model.backbone}</dd>
          <dt>Input size</dt>    <dd>${run.config.data.input_size}px</dd>
          <dt>Dropout</dt>       <dd>${run.config.model.dropout}</dd>
          <dt>Fine-tune from</dt><dd>layer ${run.config.model.fine_tune_from_layer}</dd>
        </dl>
      </article>
      <article>
        <h4>Training</h4>
        <dl>
          <dt>Epochs</dt>         <dd>${run.epochs_run} / ${run.epochs}</dd>
          <dt>Learning rate</dt>  <dd>${run.config.training.learning_rate}</dd>
          <dt>Batch size</dt>     <dd>${run.config.data.batch_size}</dd>
          <dt>Class weight</dt>   <dd>${run.config.training.class_weight || 'none'}</dd>
          <dt>LR patience</dt>    <dd>${run.config.training.lr_scheduler_patience || 0}</dd>
          <dt>Checkpoints</dt>    <dd>${run.config.training.checkpoints_strategy}</dd>
        </dl>
      </article>
      <article>
        <h4>Data & Augmentation</h4>
        <dl>
          <dt>Data dir</dt>       <dd><code>${run.config.data.data_dir}</code></dd>
          <dt>Classes</dt>        <dd>${run.config.data.classes.join(', ')}</dd>
          <dt>Augmentations</dt>  <dd>${augText}</dd>
        </dl>
      </article>

      <article id="chart-container" class="config-grid-full">
        <h4>Training Curves</h4>
        ${hasLog ? '<canvas id="training-chart"></canvas>' : '<p style="color:var(--pico-muted-color,var(--muted-color))">No training log available.</p>'}
      </article>
    </div>
  `;
}

function renderChart(run) {
  if (!run.training_log.length) return;
  const ctx = document.getElementById('training-chart');
  if (!ctx) return;

  const labels   = run.training_log.map(r => `Ep ${r.epoch + 1}`);
  const trainAcc = run.training_log.map(r => +(r.accuracy   * 100).toFixed(2));
  const valAcc   = run.training_log.map(r => +(r.val_accuracy * 100).toFixed(2));
  const trainLoss= run.training_log.map(r => +r.loss.toFixed(4));
  const valLoss  = run.training_log.map(r => +r.val_loss.toFixed(4));

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Train Acc',  data: trainAcc,  borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.07)', yAxisID: 'yAcc', tension: 0.3, fill: true },
        { label: 'Val Acc',    data: valAcc,    borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.07)',  yAxisID: 'yAcc', tension: 0.3, fill: true },
        { label: 'Train Loss', data: trainLoss, borderColor: '#a78bfa', backgroundColor: 'transparent',           yAxisID: 'yLoss', tension: 0.3, borderDash: [5,3] },
        { label: 'Val Loss',   data: valLoss,   borderColor: '#4ade80', backgroundColor: 'transparent',           yAxisID: 'yLoss', tension: 0.3, borderDash: [5,3] },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'top' } },
      scales: {
        yAcc:  { type: 'linear', position: 'left',  title: { display: true, text: 'Accuracy (%)' }, min: 0, max: 100 },
        yLoss: { type: 'linear', position: 'right', title: { display: true, text: 'Loss' }, grid: { drawOnChartArea: false } },
      },
    },
  });
}

/* ── Evaluation tab ────────────────────────────────────────────────────────── */

function buildEvalTab(run) {
  const report = run.eval_report;
  if (!report) return '<p class="error-msg">No evaluation report. Run <kbd>evaluate</kbd> first.</p>';

  const classes = Object.keys(report.per_class);
  const perClassRows = classes.map(cls => {
    const m = report.per_class[cls];
    return `
      <tr>
        <td>${cls}</td>
        <td>${(m.precision * 100).toFixed(1)}%</td>
        <td>${(m.recall    * 100).toFixed(1)}%</td>
        <td>${(m.f1        * 100).toFixed(1)}%</td>
        <td>${m.support}</td>
      </tr>
    `;
  }).join('');

  return `
    <div class="metric-cards">
      ${card('Overall Accuracy', fmtAcc(report.overall_accuracy))}
      ${card('Top-3 Accuracy',   fmtAcc(report.top3_accuracy))}
      ${card('Test Images',      report.n_images)}
      ${card('Classes',          classes.length)}
    </div>

    <article class="eval-table-card">
      <h4>Per-Class Metrics</h4>
      <div class="overflow-x">
        <table>
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
          <tbody>${perClassRows}</tbody>
        </table>
      </div>
    </article>

    <div class="cm-gallery-layout">
      <article class="cm-card">
        <h4>Confusion Matrix <small class="cm-hint">— click a cell to explore images</small></h4>
        ${buildConfusionMatrix(report, run.name)}
      </article>
      <div id="gallery-panel" class="gallery-panel" style="display:none"></div>
    </div>
  `;
}

/* ── Confusion matrix ──────────────────────────────────────────────────────── */

function buildConfusionMatrix(report, runName) {
  if (!report.confusion_matrix) return '<p>No confusion matrix data.</p>';

  const { classes, matrix } = report.confusion_matrix;
  const n = classes.length;
  const maxVal = Math.max(...matrix.flat().filter(v => v > 0), 1);

  let cells = `<div class="cm-corner"></div>`;
  cells += classes.map(c => `<div class="cm-col-header">${c}</div>`).join('');

  for (let r = 0; r < n; r++) {
    cells += `<div class="cm-row-header">${classes[r]}</div>`;
    for (let c = 0; c < n; c++) {
      const val = matrix[r][c];
      const isDiag = r === c;
      const intensity = val / maxVal;
      // Single M3 blue scale — no green/red pre-judgment; diagonal stands out naturally
      const alpha = val === 0 ? 0 : intensity * 0.82 + 0.12;
      const bg = val === 0 ? '' : `rgba(25,118,210,${alpha.toFixed(2)})`;
      const textColor = intensity > 0.42 ? '#fff' : '';
      const zeroClass = val === 0 ? ' cm-zero' : '';

      cells += `
        <div class="cm-cell${zeroClass}"
             style="${bg ? 'background:' + bg + ';' : ''}${textColor ? 'color:' + textColor + ';' : ''}"
             title="${classes[r]} → ${classes[c]}: ${val}"
             onclick="showGallery('${escHtml(runName)}', '${escHtml(classes[r])}', '${escHtml(classes[c])}', ${val})">
          ${val}
        </div>
      `;
    }
  }

  return `
    <p class="cm-axis-label">Predicted →</p>
    <div class="cm-wrap">
      <div class="cm-grid" style="grid-template-columns:auto repeat(${n},1fr)">
        ${cells}
      </div>
    </div>
    <p class="cm-legend">Rows = Actual &nbsp;·&nbsp; Columns = Predicted</p>
  `;
}

/* ── Gallery panel ─────────────────────────────────────────────────────────── */

function showGallery(runName, trueClass, predClass, count) {
  const panel = document.getElementById('gallery-panel');
  if (!panel) return;

  if (count === 0) { panel.style.display = 'none'; return; }

  const samples = (currentRun.eval_report.samples || [])
    .filter(s => s.true_class === trueClass && s.predicted_class === predClass);

  const label = trueClass === predClass
    ? `<strong>${trueClass}</strong> — correct predictions (${count})`
    : `<strong>${trueClass}</strong> misclassified as <strong>${predClass}</strong> (${count})`;

  let body;
  if (samples.length === 0) {
    body = `<p class="gallery-empty">No sample images stored for this cell.</p>`;
  } else {
    const thumbs = samples.map((s, i) => {
      const imgSrc = `/api/runs/${encodeURIComponent(runName)}/images/${imgPath(s.path)}`;
      const thumbId = `thumb-${Date.now()}-${i}`;
      return `
        <figure class="gallery-thumb" id="${thumbId}">
          <div class="thumb-img-wrap">
            <img class="thumb-original" src="${imgSrc}" alt="${escHtml(s.path)}" loading="lazy"
                 onclick="openModal('${imgSrc}')" />
          </div>
          <figcaption>
            <span>${(s.confidence * 100).toFixed(1)}%</span>
            <button class="explain-btn outline" title="Show Grad-CAM explanation"
                    onclick="explainImage(event, '${escHtml(runName)}', '${escHtml(s.path)}', '${escHtml(s.predicted_class)}', '${thumbId}')">
              Explain
            </button>
          </figcaption>
        </figure>
      `;
    }).join('');
    body = `<div class="gallery-grid">${thumbs}</div>`;
  }

  panel.innerHTML = `
    <div class="gallery-header">
      <h5>${label}</h5>
      <button class="outline" style="padding:0.2rem 0.6rem;font-size:0.8rem"
              onclick="document.getElementById('gallery-panel').style.display='none'">✕</button>
    </div>
    ${body}
  `;
  panel.style.display = 'block';
}

/* ── Grad-CAM explanation ──────────────────────────────────────────────────── */

async function explainImage(event, runName, imagePath, predictedClass, thumbId) {
  event.stopPropagation();
  const figure = document.getElementById(thumbId);
  if (!figure) return;

  const btn = event.target;
  const wrap = figure.querySelector('.thumb-img-wrap');
  const img = wrap.querySelector('img');
  const origSrc = `/api/runs/${encodeURIComponent(runName)}/images/${imgPath(imagePath)}`;

  // Toggle: if heatmap is showing, revert to original
  if (figure.dataset.explained === '1') {
    img.src = origSrc;
    img.onclick = () => openModal(origSrc);
    btn.textContent = 'Explain';
    figure.dataset.explained = '0';
    return;
  }

  // If heatmap already fetched, just swap it back in
  if (figure.dataset.heatmap) {
    img.src = figure.dataset.heatmap;
    img.onclick = () => openModal(figure.dataset.heatmap);
    btn.textContent = 'Original';
    figure.dataset.explained = '1';
    return;
  }

  btn.textContent = '…';
  btn.disabled = true;

  const classes = currentRun?.eval_report?.confusion_matrix?.classes || [];
  const classIndex = classes.indexOf(predictedClass);
  if (classIndex === -1) {
    btn.textContent = 'Explain';
    btn.disabled = false;
    const existing = figure.querySelector('.explain-error');
    if (existing) existing.remove();
    figure.insertAdjacentHTML('beforeend', '<small class="explain-error">Cannot resolve class index.</small>');
    return;
  }

  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(runName)}/explain/gradcam`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_path: imagePath, class_index: classIndex }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const { heatmap_b64 } = await res.json();
    const heatSrc = `data:image/png;base64,${heatmap_b64}`;

    figure.dataset.heatmap = heatSrc;
    img.src = heatSrc;
    img.onclick = () => openModal(heatSrc);
    btn.textContent = 'Original';
    btn.disabled = false;
    figure.dataset.explained = '1';
  } catch (e) {
    btn.textContent = 'Explain';
    btn.disabled = false;
    const existing = figure.querySelector('.explain-error');
    if (existing) existing.remove();
    figure.insertAdjacentHTML('beforeend', `<small class="explain-error">${escHtml(e.message)}</small>`);
  }
}

function imgPath(path) {
  return path.split('/').map(encodeURIComponent).join('/');
}

/* ── Image modal ───────────────────────────────────────────────────────────── */

function openModal(src) {
  document.getElementById('modal-content').innerHTML = `<img src="${src}" />`;
  document.getElementById('modal-overlay').style.display = 'flex';
}

function closeModal() {
  document.getElementById('modal-overlay').style.display = 'none';
  document.getElementById('modal-content').innerHTML = '';
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

/* ── Compare tab ───────────────────────────────────────────────────────────── */

function buildCompareTab() {
  return `
    <article>
      <p class="section-label">Compare with another run</p>
      <div id="compare-picker"><p aria-busy="true">Loading runs…</p></div>
      <div id="compare-result"></div>
    </article>
  `;
}

async function loadCompareOptions() {
  const picker = document.getElementById('compare-picker');
  if (!picker) return;
  try {
    const runs = await api('/runs');
    const options = runs
      .filter(r => r.name !== currentRun.name)
      .map(r => `<option value="${escHtml(r.name)}">${r.name}</option>`)
      .join('');

    if (!options) {
      picker.innerHTML = '<p style="color:var(--muted-color)">No other runs to compare with.</p>';
      return;
    }
    picker.innerHTML = `
      <select id="compare-select" onchange="loadCompareRun(this.value)">
        <option value="">Select a run to compare…</option>
        ${options}
      </select>
    `;
  } catch (e) {
    picker.innerHTML = `<p class="error-msg">Failed to load runs: ${e.message}</p>`;
  }
}

async function loadCompareRun(name) {
  if (!name) return;
  const result = document.getElementById('compare-result');
  result.innerHTML = '<p aria-busy="true">Loading…</p>';
  try {
    const other = await api(`/runs/${encodeURIComponent(name)}`);
    result.innerHTML = buildCompareSideBySide(currentRun, other);
  } catch (e) {
    result.innerHTML = `<p class="error-msg">Failed to load run: ${e.message}</p>`;
  }
}

function buildCompareSideBySide(a, b) {
  const rows = [
    ['Status',         a.status,           b.status],
    ['Backbone',       a.backbone,         b.backbone],
    ['Val Accuracy',   fmtAcc(a.val_accuracy),  fmtAcc(b.val_accuracy)],
    ['Test Accuracy',  fmtAcc(a.test_accuracy), fmtAcc(b.test_accuracy)],
    ['Learning Rate',  a.lr,               b.lr],
    ['Epochs (done/total)', `${a.epochs_run}/${a.epochs}`, `${b.epochs_run}/${b.epochs}`],
    ['Batch Size',     a.config.data.batch_size,       b.config.data.batch_size],
    ['Input Size',     a.config.data.input_size + 'px',b.config.data.input_size + 'px'],
    ['Dropout',        a.config.model.dropout,         b.config.model.dropout],
    ['Classes',        a.config.data.classes.join(', '), b.config.data.classes.join(', ')],
    ['Augmentations',  a.config.augmentation.map(t=>t.name).join(', ') || 'None',
                       b.config.augmentation.map(t=>t.name).join(', ') || 'None'],
    ['Checkpoint Strategy', a.config.training.checkpoints_strategy, b.config.training.checkpoints_strategy],
    ['LR Patience',    a.config.training.lr_scheduler_patience, b.config.training.lr_scheduler_patience],
  ];

  const rowsHtml = rows.map(([label, av, bv]) => {
    const diff = String(av) !== String(bv);
    return `
      <tr class="${diff ? 'compare-diff' : ''}">
        <td>${label}</td>
        <td>${av}</td>
        <td>${bv}</td>
      </tr>
    `;
  }).join('');

  return `
    <div class="overflow-x" style="margin-top:1rem">
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>${escHtml(a.name)}</th>
            <th>${escHtml(b.name)}</th>
          </tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    </div>
    <p class="compare-diff-note">Highlighted rows differ between runs.</p>
  `;
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

/* ── Inference page ──────────────────────────────────────────────────────────── */

let _inferRuns       = [];
let _inferAugSchema  = [];
let _inferFile       = null;   // File object from drop / browse
let _inferAugments   = [];     // [{id, name, label, params: {name: value}}]
let _inferAugCounter = 0;
let _inferOrigResult = null;   // last original prediction result
let _inferAugResult  = null;   // last augmented prediction result
let _inferOrigKey    = null;   // "run::filename::size" — cache key for orig result

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

  // Clear previous results — new file invalidates the orig cache
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
  // Update displayed value label without full re-render
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

  // Toggle: if heatmap is showing, revert to original
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

  // For the augmented panel, XAI runs on the augmented image bytes (base64 → Blob)
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
