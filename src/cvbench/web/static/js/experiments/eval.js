/* ── Evaluation tab ────────────────────────────────────────────────────────── */

// Task-specific Evaluation tabs, keyed by the task name stored in the report. A task
// with no entry gets the generic tab, built purely from the server's formatted fields.
const EVAL_TAB_BUILDERS = {
  classification: buildClassificationEvalTab,
  detection: buildDetectionEvalTab,
};

// Where a confusion-matrix cell click goes; tasks not listed use the plain image gallery.
const CELL_GALLERY_HANDLERS = { detection: 'showDetectionCellGallery' };

function buildGenericEvalTab(run, report) {
  const d = run.eval_display;
  const facts = d.summary.map(([label, value]) => `<tr><th>${escHtml(label)}</th><td>${escHtml(value)}</td></tr>`).join('');
  const head = d.per_class.columns.map(c => `<th>${escHtml(c)}</th>`).join('');
  const rows = d.per_class.rows
    .map(r => `<tr>${r.map(c => `<td>${escHtml(c)}</td>`).join('')}</tr>`).join('');
  return `
    <article><table><tbody>${facts}</tbody></table></article>
    <article>
      <h3>${escHtml(d.sections.per_class)}</h3>
      <table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>
    </article>`;
}

function buildEvalTab(run) {
  const report = run.eval_report;
  if (!report) return '<p class="error-msg">No evaluation report. Run <kbd>evaluate</kbd> first.</p>';
  if (!run.eval_display) return '<p class="error-msg">This evaluation report predates the current format. Run <kbd>evaluate</kbd> again.</p>';

  const build = EVAL_TAB_BUILDERS[report.task] || buildGenericEvalTab;
  const body = build(run, report);
  return advisoriesHtml(run.advisories) + body;
}

// Same advisories `evaluate` and `runs show` print (core/present.py run_advisories).
function advisoriesHtml(advisories) {
  return (advisories || []).map(a => `<article class="advisory" role="note">⚠ ${escHtml(a)}</article>`).join('');
}

// A per-class table from the server's {columns, rows}.
function perClassTable(table) {
  const head = table.columns.map(c => `<th>${escHtml(c)}</th>`).join('');
  const body = table.rows.map(r => `<tr>${r.map(v => `<td>${escHtml(v)}</td>`).join('')}</tr>`).join('');
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function buildClassificationEvalTab(run, report) {
  const sections = _present.sections;
  return `
    <div class="metric-cards">
      ${summaryCards(run)}
    </div>

    <article class="eval-table-card">
      <h4>${escHtml(sections.per_class)}</h4>
      <div class="overflow-x">
        ${perClassTable(run.eval_display.per_class)}
      </div>
    </article>

    <div class="cm-gallery-layout">
      <article class="cm-card">
        <h4>${escHtml(sections.confusion)} <small class="cm-hint">— click a cell to explore images</small></h4>
        ${buildConfusionMatrix(report, run.name)}
      </article>
      <div id="gallery-panel" class="gallery-panel" style="display:none"></div>
    </div>
  `;
}

// Shared by every task's gallery thumb: reads the box/src/path data attributes
// a thumb was built with (see buildGalleryThumb) and opens the zoom modal.
// Classification thumbs carry no boxes, so openModal() falls back to its plain
// (no box-overlay, no metrics panel) rendering automatically.
function openGalleryModal(el) {
  let boxes = [];
  try { boxes = JSON.parse(el.dataset.boxes || '[]'); } catch (e) { /* noop */ }
  openModal(el.dataset.src, el.dataset.path, boxes);
}

// One thumb markup shared by every task's confusion-matrix gallery (currently
// classification and detection; extend here, not per-task, for future tasks).
// `boxes` (optional) draws an overlay on the thumb itself, dimmed/filtered per
// the caller's current layer selection; `modalBoxes` (optional, defaults to
// `boxes`) is what the zoom modal receives — detection passes every box
// undimmed so the modal's own GT/Prediction toggles work independent of the
// gallery's filter. `actions` is task-specific figcaption markup (e.g.
// classification's Explain button) — everything else renders identically.
function buildGalleryThumb({ id, imgSrc, path, boxes = [], modalBoxes, caption, actions = '' }) {
  const modalBoxesJson = escHtml(JSON.stringify(modalBoxes !== undefined ? modalBoxes : boxes));
  return `
    <figure class="gallery-thumb" id="${id}">
      <div class="thumb-img-wrap ds-modal-img-wrap"
           data-boxes="${modalBoxesJson}"
           data-src="${imgSrc}" data-path="${escHtml(path)}"
           onclick="openGalleryModal(this)">
        <img class="thumb-original ds-tile-img--contain" src="${imgSrc}" alt="${escHtml(path)}" loading="lazy"
             onload="fitBoxOverlay(this)" />
        ${boxes.length ? buildBoxLayer(boxes, false, { labels: false }) : ''}
      </div>
      <figcaption>
        <span title="${escHtml(path.split('/').pop())}">${caption}</span>
        ${actions}
      </figcaption>
    </figure>
  `;
}

/* ── Confusion matrix ──────────────────────────────────────────────────────── */

function buildConfusionMatrix(report, runName) {
  if (!report.confusion_matrix) return '<p>No confusion matrix data.</p>';

  const { classes, matrix } = report.confusion_matrix;
  const n = classes.length;
  const maxVal = Math.max(...matrix.flat().filter(v => v > 0), 1);
  const onCell = CELL_GALLERY_HANDLERS[report.task] || 'showGallery';
  const bgClass = c => (c === 'background' ? ' cm-bg-header' : '');

  let cells = `<div class="cm-corner"></div>`;
  cells += classes.map(c => `<div class="cm-col-header${bgClass(c)}">${c}</div>`).join('');

  for (let r = 0; r < n; r++) {
    cells += `<div class="cm-row-header${bgClass(classes[r])}">${classes[r]}</div>`;
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
             onclick="${onCell}('${escHtml(runName)}', '${escHtml(classes[r])}', '${escHtml(classes[c])}', ${val})">
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
      const explainBtn = `
        <button class="explain-btn outline" title="Show Grad-CAM explanation"
                onclick="explainImage(event, '${escHtml(runName)}', '${escHtml(s.path)}', '${escHtml(s.predicted_class)}', '${thumbId}')">
          Explain
        </button>`;
      return buildGalleryThumb({
        id: thumbId,
        imgSrc, path: s.path,
        caption: `${(s.confidence * 100).toFixed(1)}%`,
        actions: explainBtn,
      });
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
    wrap.dataset.src = origSrc;
    btn.textContent = 'Explain';
    figure.dataset.explained = '0';
    return;
  }

  if (figure.dataset.heatmap) {
    img.src = figure.dataset.heatmap;
    wrap.dataset.src = figure.dataset.heatmap;
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
    wrap.dataset.src = heatSrc;
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
