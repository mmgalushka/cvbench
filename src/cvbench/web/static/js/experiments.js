/* ── State ─────────────────────────────────────────────────────────────────── */

let currentRun = null;
let activeTab = 'training';
let chartInstance = null;

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
      <button class="tab-btn" id="tab-export"   onclick="switchTab('export')">Export</button>
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

/* ── Tab switcher ──────────────────────────────────────────────────────────── */

function switchTab(tab) {
  activeTab = tab;
  ['training', 'eval', 'compare', 'export'].forEach(t => {
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
  } else if (tab === 'export') {
    content.innerHTML = buildExportTab(currentRun);
    loadExports(currentRun.name);
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
      const intensity = val / maxVal;
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

  if (figure.dataset.explained === '1') {
    img.src = origSrc;
    img.onclick = () => openModal(origSrc);
    btn.textContent = 'Explain';
    figure.dataset.explained = '0';
    return;
  }

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

/* ── Export tab ────────────────────────────────────────────────────────────── */

function buildExportTab(run) {
  const hasCheckpoint = run.status === 'done' || run.status === 'evaluated' || run.epochs_run > 0;
  return `
    <div id="exports-list"><p aria-busy="true">Loading exports…</p></div>

    <article class="export-form-card">
      <h4>Generate New Export</h4>
      ${!hasCheckpoint ? '<p class="error-msg">No trained checkpoint available. Train the model first.</p>' : `
        <div class="export-form-row">
          <label>
            Format
            <select id="export-format" onchange="updateExportForm()">
              <option value="tflite">TFLite</option>
              <option value="onnx">ONNX</option>
            </select>
          </label>
          <label id="quantize-wrap">
            Quantization
            <select id="export-quantize">
              <option value="none">None (float32)</option>
              <option value="float16">Float16</option>
              <option value="int8">Int8</option>
            </select>
          </label>
        </div>
        <button id="export-generate-btn" onclick="generateExport('${escHtml(run.name)}')">Generate Export</button>
        <p id="export-error" class="error-msg" style="display:none;margin-top:0.5rem"></p>
      `}
    </article>
  `;
}

function updateExportForm() {
  const fmt = document.getElementById('export-format')?.value;
  const wrap = document.getElementById('quantize-wrap');
  if (wrap) wrap.style.display = fmt === 'tflite' ? '' : 'none';
}

async function loadExports(runName) {
  const el = document.getElementById('exports-list');
  if (!el) return;
  try {
    const exports = await api(`/runs/${encodeURIComponent(runName)}/exports`);
    el.innerHTML = renderExportsList(exports);
  } catch (e) {
    el.innerHTML = `<p class="error-msg">Failed to load exports: ${e.message}</p>`;
  }
}

function renderExportsList(exports) {
  if (exports.length === 0) {
    return `<p class="export-empty">No exports yet. Use the form below to generate one.</p>`;
  }

  const rows = exports.map(ex => {
    const fmt = ex.format ? ex.format.toUpperCase() : ex.subfolder.toUpperCase();
    const quant = ex.quantize || 'none';
    const size = ex.size_mb != null ? `${ex.size_mb} MB` : '—';
    const date = ex.exported_at ? ex.exported_at.replace('T', ' ') : '—';
    const dlUrl = `/api/runs/${encodeURIComponent(currentRun.name)}/exports/${encodeURIComponent(ex.subfolder)}/download`;
    return `
      <tr>
        <td><strong>${fmt}</strong></td>
        <td><span class="export-quant-badge">${quant}</span></td>
        <td>${size}</td>
        <td>${date}</td>
        <td><a href="${dlUrl}" class="export-download-btn" role="button">↓ .tar.gz</a></td>
      </tr>
    `;
  }).join('');

  return `
    <article class="export-list-card">
      <h4>Available Exports</h4>
      <div class="overflow-x">
        <table class="export-table">
          <thead><tr><th>Format</th><th>Quantization</th><th>Size</th><th>Exported</th><th></th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </article>
  `;
}

async function generateExport(runName) {
  const btn = document.getElementById('export-generate-btn');
  const errEl = document.getElementById('export-error');
  const format = document.getElementById('export-format')?.value;
  const quantize = document.getElementById('export-quantize')?.value || 'none';

  if (!btn || !format) return;

  btn.disabled = true;
  btn.textContent = 'Generating…';
  btn.setAttribute('aria-busy', 'true');
  if (errEl) errEl.style.display = 'none';

  try {
    const exports = await fetch(`/api/runs/${encodeURIComponent(runName)}/exports`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format, quantize }),
    });
    if (!exports.ok) {
      const err = await exports.json().catch(() => ({ detail: exports.statusText }));
      throw new Error(err.detail || exports.statusText);
    }
    const data = await exports.json();
    const listEl = document.getElementById('exports-list');
    if (listEl) listEl.innerHTML = renderExportsList(data);
  } catch (e) {
    if (errEl) { errEl.textContent = e.message; errEl.style.display = ''; }
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate Export';
    btn.removeAttribute('aria-busy');
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
