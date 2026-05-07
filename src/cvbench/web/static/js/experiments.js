/* ── State ─────────────────────────────────────────────────────────────────── */

let currentRun = null;
let activeTab = 'training';
let chartInstance = null;
let currentExports = [];

/* ── Helpers ────────────────────────────────────────────────────────────────── */

function fmtClassWeight(cw) {
  if (!cw) return 'none';
  if (typeof cw === 'string') return cw;
  return Object.entries(cw).map(([k, v]) => `${k}: ${v}`).join(', ');
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
      <div class="run-actions-menu" id="run-actions-menu">
        <button class="run-actions-trigger" onclick="toggleRunMenu(event)" title="More actions"><i class="fas fa-ellipsis-v"></i></button>
        <ul class="run-actions-dropdown" id="run-actions-dropdown">
          <li class="run-actions-item" onclick="renameRun('${escHtml(run.name)}'); closeRunMenu()"><i class="fas fa-pencil-alt"></i> Rename</li>
          <li class="run-actions-item run-actions-item--danger" onclick="deleteRun('${escHtml(run.name)}'); closeRunMenu()"><i class="fas fa-trash-alt"></i> Delete</li>
        </ul>
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
  const cliHtml = run.cli_command ? `
    <div class="cli-command-bar">
      <svg class="cli-label" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
      <code class="cli-code">${escHtml(run.cli_command)}</code>
      <button class="cli-copy-btn" data-cmd="${escHtml(run.cli_command)}" onclick="copyCliCommand(this)">Copy</button>
    </div>` : '';

  return `
    ${cliHtml}
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
          <dt>Seed</dt>           <dd>${run.config.training.seed != null ? run.config.training.seed : '—'}</dd>
          <dt>Class weight</dt>   <dd>${fmtClassWeight(run.config.training.class_weight)}</dd>
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
              <option value="plan">TensorRT Plan (Jetson)</option>
              <option value="hailo">Hailo HEF (hailo8l)</option>
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
          <label id="calib-samples-wrap" style="display:none">
            Samples per class
            <input id="export-calib-samples" type="number" min="1" step="1" placeholder="auto">
          </label>
        </div>
        <button id="export-generate-btn" onclick="generateExport('${escHtml(run.name)}')">Generate Export</button>
        <p id="export-error" class="error-msg" style="display:none;margin-top:0.5rem"></p>
      `}
    </article>
  `;
}

const _ALL_EXPORT_FORMATS = [
  { value: 'tflite', label: 'TFLite' },
  { value: 'onnx',   label: 'ONNX' },
  { value: 'plan',   label: 'TensorRT Plan (Jetson)' },
  { value: 'hailo',  label: 'Hailo HEF (hailo8l)' },
];

function updateFormatDropdown(exports) {
  const select = document.getElementById('export-format');
  if (!select) return;
  const existing = new Set(exports.map(ex => ex.format || ex.subfolder));
  const available = _ALL_EXPORT_FORMATS.filter(f => !existing.has(f.value));
  const prev = select.value;
  select.innerHTML = available.map(f => `<option value="${f.value}">${f.label}</option>`).join('');
  if (available.find(f => f.value === prev)) select.value = prev;
  const formCard = document.querySelector('.export-form-card');
  if (formCard) formCard.style.display = available.length === 0 ? 'none' : '';
  updateExportForm();
}

function updateExportForm() {
  const fmt = document.getElementById('export-format')?.value;
  const wrap = document.getElementById('quantize-wrap');
  if (wrap) wrap.style.display = fmt === 'tflite' ? '' : 'none';
  const calibWrap = document.getElementById('calib-samples-wrap');
  if (calibWrap) calibWrap.style.display = fmt === 'hailo' ? '' : 'none';
}

function buildPlanInstructions(runName, cardMode = false) {
  const onnxExists = currentExports.some(e => e.format === 'onnx' || e.subfolder === 'onnx');
  const scp = `scp experiments/${runName}/export/onnx/model.onnx user@jetson:/home/user/model.onnx`;
  const trtexec = `trtexec --onnx=model.onnx --saveEngine=model.plan --noTF32`;
  const cliCmd = `cvbench runs export ${runName} --format plan`;
  const termIcon = `<svg class="cli-label" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>`;

  const onnxWarning = onnxExists ? '' : `
    <div class="plan-warning">
      <strong>ONNX export not found.</strong> Generate it first:
      <div class="cli-command-bar plan-cmd-inline">
        ${termIcon}
        <code class="cli-code">cvbench runs export ${escHtml(runName)} --format onnx</code>
        <button class="cli-copy-btn" data-cmd="cvbench runs export ${escHtml(runName)} --format onnx" onclick="copyCliCommand(this)">Copy</button>
      </div>
    </div>`;

  const intro = cardMode
    ? `The <code>.plan</code> file must be built on the target Jetson device — it is compiled for a specific GPU architecture. Use the ONNX export from this run.`
    : `A TensorRT <code>.plan</code> file must be built on the target Jetson device itself — it is compiled for a specific GPU architecture.`;

  return `
    <div class="plan-instructions-box">
      <p class="plan-intro">${intro}</p>
      ${cardMode ? '' : onnxWarning}
      <div class="plan-step">
        <span class="plan-step-label">Step 1</span>
        <span class="plan-step-desc">Copy the ONNX model to your Jetson:</span>
        <div class="cli-command-bar">
          ${termIcon}
          <code class="cli-code">${escHtml(scp)}</code>
          <button class="cli-copy-btn" data-cmd="${escHtml(scp)}" onclick="copyCliCommand(this)">Copy</button>
        </div>
      </div>
      <div class="plan-step">
        <span class="plan-step-label">Step 2</span>
        <span class="plan-step-desc">On the Jetson, convert to TensorRT engine plan:</span>
        <div class="cli-command-bar">
          ${termIcon}
          <code class="cli-code">${escHtml(trtexec)}</code>
          <button class="cli-copy-btn" data-cmd="${escHtml(trtexec)}" onclick="copyCliCommand(this)">Copy</button>
        </div>
      </div>
      <div class="plan-step">
        <span class="plan-step-label">Step 3</span>
        <span class="plan-step-desc">Run inference using the TensorRT Python API or DeepStream.</span>
      </div>
      <div class="plan-step plan-cli-equivalent">
        <span class="plan-step-label">CLI</span>
        <span class="plan-step-desc">Equivalent command (prints these instructions):</span>
        <div class="cli-command-bar">
          ${termIcon}
          <code class="cli-code">${escHtml(cliCmd)}</code>
          <button class="cli-copy-btn" data-cmd="${escHtml(cliCmd)}" onclick="copyCliCommand(this)">Copy</button>
        </div>
      </div>
    </div>
  `;
}

function buildHailoInstructions(runName, cardMode = false) {
  const rel = `experiments/${runName}/export/hailo`;
  const termIcon = `<svg class="cli-label" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>`;
  const cliCmd = `cvbench runs export ${runName} --format hailo`;

  const step1cmd = `hailo parser tf model.tflite`;
  const step2cmd = `hailo optimize --hw-arch hailo8l --calib-set-path calib_set.npy --model-script model.alls --output-har-path model_optimized.har model.har`;
  const step3cmd = `hailo compiler --hw-arch hailo8l model_optimized.har`;

  const hailoIntro = cardMode
    ? `Files are ready in <code>${escHtml(rel)}/</code>. Run the commands below inside the Hailo Docker container.`
    : `Click <strong>Generate Export</strong> to prepare <code>model.tflite</code>, <code>calib_set.npy</code>, and <code>model.alls</code> in <code>${escHtml(rel)}/</code>. Then run the commands below inside the Hailo Docker container.`;

  return `
    <div class="plan-instructions-box">
      <p class="plan-intro">${hailoIntro}</p>
      <div class="plan-step">
        <span class="plan-step-label">Step 1</span>
        <span class="plan-step-desc">Parse TFLite to HAR:</span>
        <div class="cli-command-bar">
          ${termIcon}
          <code class="cli-code">${escHtml(step1cmd)}</code>
          <button class="cli-copy-btn" data-cmd="${escHtml(step1cmd)}" onclick="copyCliCommand(this)">Copy</button>
        </div>
      </div>
      <div class="plan-step">
        <span class="plan-step-label">Step 2</span>
        <span class="plan-step-desc">Optimize with calibration data:</span>
        <div class="cli-command-bar">
          ${termIcon}
          <code class="cli-code">${escHtml(step2cmd)}</code>
          <button class="cli-copy-btn" data-cmd="${escHtml(step2cmd)}" onclick="copyCliCommand(this)">Copy</button>
        </div>
      </div>
      <div class="plan-step">
        <span class="plan-step-label">Step 3</span>
        <span class="plan-step-desc">Compile to HEF:</span>
        <div class="cli-command-bar">
          ${termIcon}
          <code class="cli-code">${escHtml(step3cmd)}</code>
          <button class="cli-copy-btn" data-cmd="${escHtml(step3cmd)}" onclick="copyCliCommand(this)">Copy</button>
        </div>
      </div>
      <div class="plan-step plan-cli-equivalent">
        <span class="plan-step-label">CLI</span>
        <span class="plan-step-desc">Equivalent command (generates the package):</span>
        <div class="cli-command-bar">
          ${termIcon}
          <code class="cli-code">${escHtml(cliCmd)}</code>
          <button class="cli-copy-btn" data-cmd="${escHtml(cliCmd)}" onclick="copyCliCommand(this)">Copy</button>
        </div>
      </div>
    </div>
  `;
}

async function loadExports(runName) {
  const el = document.getElementById('exports-list');
  if (!el) return;
  try {
    const exports = await api(`/runs/${encodeURIComponent(runName)}/exports`);
    currentExports = exports;
    el.innerHTML = renderExportsList(exports);
    updateFormatDropdown(exports);
  } catch (e) {
    el.innerHTML = `<p class="error-msg">Failed to load exports: ${e.message}</p>`;
  }
}

function fmtDisplayName(fmt) {
  const map = { tflite: 'TFLite', onnx: 'ONNX', plan: 'TensorRT', hailo: 'Hailo HEF' };
  return map[fmt] || fmt.toUpperCase();
}

function toggleExportInstr(id, btn) {
  const el = document.getElementById(id);
  if (!el) return;
  const opening = el.style.display === 'none';
  el.style.display = opening ? '' : 'none';
  btn.classList.toggle('active', opening);
}

function toggleExportMenu(e, id) {
  e.stopPropagation();
  const dropdown = document.getElementById(id);
  if (!dropdown) return;
  const isOpen = dropdown.classList.contains('open');
  document.querySelectorAll('.run-actions-dropdown.open').forEach(d => d.classList.remove('open'));
  if (!isOpen) {
    dropdown.classList.add('open');
    document.addEventListener('click', () => closeExportMenu(id), { once: true });
  }
}

function closeExportMenu(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('open');
}

function renderExportsList(exports) {
  if (exports.length === 0) {
    return `<p class="export-empty">No exports yet. Use the form below to generate one.</p>`;
  }

  const cards = exports.map((ex, i) => {
    const fmt = ex.format || ex.subfolder;
    const quant = ex.quantize || 'none';
    const size = ex.size_mb != null ? `${ex.size_mb} MB` : '—';
    const date = ex.exported_at ? ex.exported_at.replace('T', ' ') : '—';
    const dlFilename = `${encodeURIComponent(currentRun.name)}_${encodeURIComponent(ex.subfolder)}.tar.gz`;
    const dlUrl = `/api/runs/${encodeURIComponent(currentRun.name)}/exports/${encodeURIComponent(ex.subfolder)}/download/${dlFilename}`;
    const dlPath = dlUrl;
    const curl = `curl -O http://<HOST>:<PORT>${dlPath}`;
    const wget = `wget http://<HOST>:<PORT>${dlPath}`;
    const hasInstr = fmt === 'hailo' || fmt === 'plan';
    const instrId = `export-instr-${i}`;
    const menuId = `export-menu-${i}`;

    return `
      <article class="export-card">
        <div class="export-card-header">
          <div class="export-card-meta">
            <span class="export-meta-name">${fmtDisplayName(fmt)}</span>
            ${quant && quant !== 'none' ? `<span class="export-meta-quant">${quant}</span>` : ''}
          </div>
          <div class="export-card-actions">
            ${hasInstr ? `<button class="export-instr-toggle" onclick="toggleExportInstr('${instrId}', this)"><i class="fas fa-chevron-down"></i> Instructions</button>` : ''}
            <a href="${dlUrl}" class="export-download-btn" role="button" title="Download archive"><i class="fas fa-download"></i> .tar.gz</a>
            <button class="cli-copy-btn" data-cmd="${escHtml(curl)}" onclick="copyCliCommand(this)" title="${escHtml(curl)}">curl</button>
            <button class="cli-copy-btn" data-cmd="${escHtml(wget)}" onclick="copyCliCommand(this)" title="${escHtml(wget)}">wget</button>
            <div class="run-actions-menu">
              <button class="run-actions-trigger" onclick="toggleExportMenu(event, '${menuId}')" title="More actions"><i class="fas fa-ellipsis-v"></i></button>
              <ul class="run-actions-dropdown" id="${menuId}">
                <li class="run-actions-item run-actions-item--danger" onclick="deleteExport('${escHtml(currentRun.name)}', '${escHtml(ex.subfolder)}'); closeExportMenu('${menuId}')"><i class="fas fa-trash-alt"></i> Delete</li>
              </ul>
            </div>
          </div>
        </div>
        ${hasInstr ? `
        <div class="export-card-instructions" id="${instrId}" style="display:none">
          ${fmt === 'hailo' ? buildHailoInstructions(currentRun.name, true) : buildPlanInstructions(currentRun.name, true)}
        </div>` : ''}
      </article>
    `;
  }).join('');

  return `
    <div class="export-cards">
      <h4 class="export-cards-title">Available Exports</h4>
      ${cards}
    </div>
  `;
}

async function generateExport(runName) {
  const btn = document.getElementById('export-generate-btn');
  const errEl = document.getElementById('export-error');
  const format = document.getElementById('export-format')?.value;
  const quantize = document.getElementById('export-quantize')?.value || 'none';
  const calibSamplesRaw = document.getElementById('export-calib-samples')?.value;
  const calib_samples_per_class = calibSamplesRaw ? parseInt(calibSamplesRaw, 10) : null;
  if (!btn || !format) return;

  btn.disabled = true;
  btn.textContent = 'Generating…';
  btn.setAttribute('aria-busy', 'true');
  if (errEl) errEl.style.display = 'none';

  try {
    const body = { format, quantize };
    if (calib_samples_per_class !== null) body.calib_samples_per_class = calib_samples_per_class;
    const exports = await fetch(`/api/runs/${encodeURIComponent(runName)}/exports`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!exports.ok) {
      const err = await exports.json().catch(() => ({ detail: exports.statusText }));
      throw new Error(err.detail || exports.statusText);
    }
    const data = await exports.json();
    currentExports = data;
    const listEl = document.getElementById('exports-list');
    if (listEl) listEl.innerHTML = renderExportsList(data);
    updateFormatDropdown(data);
  } catch (e) {
    if (errEl) { errEl.textContent = e.message; errEl.style.display = ''; }
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate Export';
    btn.removeAttribute('aria-busy');
  }
}

function toggleRunMenu(e) {
  e.stopPropagation();
  const dropdown = document.getElementById('run-actions-dropdown');
  if (!dropdown) return;
  const open = dropdown.classList.toggle('open');
  if (open) {
    document.addEventListener('click', closeRunMenu, { once: true });
  }
}

function closeRunMenu() {
  const dropdown = document.getElementById('run-actions-dropdown');
  if (dropdown) dropdown.classList.remove('open');
}

const _VALID_RUN_NAME = /^[a-zA-Z0-9][a-zA-Z0-9_\-]*$/;

async function renameRun(runName) {
  const newName = prompt(`Rename experiment "${runName}" to:`, runName);
  if (newName === null || newName === runName) return;
  if (!newName || !_VALID_RUN_NAME.test(newName) || newName.length > 100) {
    alert('Invalid name. Use only letters, digits, underscores, and hyphens (max 100 chars), starting with a letter or digit.');
    return;
  }
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(runName)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_name: newName }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      alert(`Failed to rename run: ${err.detail || res.statusText}`);
      return;
    }
    navigate(`#/runs/${encodeURIComponent(newName)}`);
  } catch (e) {
    alert(`Failed to rename run: ${e.message}`);
  }
}

async function deleteRun(runName) {
  if (!confirm(`Permanently delete run "${runName}" and all its contents? This cannot be undone.`)) return;
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(runName)}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      alert(`Failed to delete run: ${err.detail || res.statusText}`);
      return;
    }
    navigate('#/');
  } catch (e) {
    alert(`Failed to delete run: ${e.message}`);
  }
}

async function deleteExport(runName, subfolder) {
  if (!confirm(`Permanently delete export "${subfolder}" from run "${runName}"?`)) return;
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(runName)}/exports/${encodeURIComponent(subfolder)}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      alert(`Failed to delete export: ${err.detail || res.statusText}`);
      return;
    }
    const data = await res.json();
    currentExports = data;
    const listEl = document.getElementById('exports-list');
    if (listEl) listEl.innerHTML = renderExportsList(data);
    updateFormatDropdown(data);
  } catch (e) {
    alert(`Failed to delete export: ${e.message}`);
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
    ['Seed',           a.config.training.seed ?? '—',      b.config.training.seed ?? '—'],
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
