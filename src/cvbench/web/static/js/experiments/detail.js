/* ── Helpers ────────────────────────────────────────────────────────────────── */

function fmtClassWeight(cw) {
  if (!cw) return 'none';
  if (typeof cw === 'string') return cw;
  return Object.entries(cw).map(([k, v]) => `${k}: ${v}`).join(', ');
}

/* ── Run detail ────────────────────────────────────────────────────────────── */

async function showRunDetail(name) {
  setActive('');
  const page = document.getElementById('page');
  page.innerHTML = '<p aria-busy="true">Loading run…</p>';
  try {
    await loadPresent();
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

  // Same facts, labels and values as `runs show` (core/present.py run_fields);
  // the status is the badge in the title, not a card.
  const factCards = run.display.fields
    .filter(([label]) => label !== 'Status')
    .map(([label, value]) => card(label, escHtml(value)))
    .join('');

  return `
    <div class="page-header">
      <div>
        ${run.sweep
          ? `<a href="#/sweeps/${encodeURIComponent(run.sweep)}" class="back-link">← ${escHtml(run.sweep)}</a>`
          : `<a href="#/" class="back-link" onclick="setActive('nav-runs')">← Runs</a>`}
        <h2><span class="type-name">${RUN_ICON}<span>${run.name}</span></span> <span class="badge badge-${run.status}">${run.status}</span>${isDefaultTask(run.task) ? '' : ` <span class="badge badge-format">${escHtml(run.task)}</span>`}</h2>
      </div>
      <div class="run-actions-menu" id="run-actions-menu">
        <button class="run-actions-trigger" onclick="toggleRunMenu(event)" title="More actions"><i class="fas fa-ellipsis-v"></i></button>
        <ul class="run-actions-dropdown" id="run-actions-dropdown">
          ${run.sweep ? '' : `<li class="run-actions-item" onclick="renameRun('${escHtml(run.name)}'); closeRunMenu()"><i class="fas fa-pencil-alt"></i> Rename</li>`}
          <li class="run-actions-item run-actions-item--danger" onclick="deleteRun('${escHtml(run.name)}'${run.sweep ? `, 0, '${escHtml(run.sweep)}'` : ''}); closeRunMenu()"><i class="fas fa-trash-alt"></i> Delete</li>
        </ul>
      </div>
    </div>

    <div class="metric-cards">
      ${factCards}
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

function card(label, value, sub) {
  return `
    <article class="metric-card">
      <small>${label}</small>
      <strong>${value}</strong>
      ${sub ? `<small class="metric-card-sub">${sub}</small>` : ''}
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
    _detCell = null;
    _detTag = null;
  } else if (tab === 'compare') {
    content.innerHTML = buildCompareTab();
    loadCompareOptions();
  } else if (tab === 'export') {
    content.innerHTML = buildExportTab(currentRun);
    loadExports(currentRun.name);
  }
}

/* ── Training tab ──────────────────────────────────────────────────────────── */

/* Names of the augmentation transforms in a serialized config ("one_of" groups list their candidates). */
function augNames(config) {
  return config.augmentation.transforms
    .map(t => t.one_of ? `one_of(${t.one_of.candidates.map(c => c.name).join(', ')})` : t.name)
    .join(', ');
}

function fmtOptimizer(o) {
  const extra = o.type === 'sgd' ? `, momentum ${o.momentum}` : '';
  return `${o.type}${extra}${o.weight_decay ? `, weight decay ${o.weight_decay}` : ''}`;
}

function fmtLoss(l) {
  const extra = l.type === 'focal' ? `, gamma ${l.focal_gamma}` : '';
  return `${l.type}${extra}${l.label_smoothing ? `, label smoothing ${l.label_smoothing}` : ''}`;
}

function fmtScheduler(s) {
  return s.patience ? `patience ${s.patience}, factor ${s.factor}, min ${s.min_lr}` : 'disabled';
}

// Headline numbers of an eval report, preformatted by the server.
function summaryCards(run) {
  return run.eval_display.summary.map(([label, value]) => card(label, escHtml(value))).join('');
}

function fmtPct(v) { return v == null ? 'n/a' : (v * 100).toFixed(1) + '%'; }
