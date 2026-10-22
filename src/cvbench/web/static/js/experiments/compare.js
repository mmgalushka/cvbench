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
    [`Test ${a.primary_metric === b.primary_metric ? metricLabel(a.primary_metric) : 'score'}`,
                       fmtAcc(a.primary_value), fmtAcc(b.primary_value)],
    ['Learning rate',  a.display.lr,       b.display.lr],
    ['Epochs',          a.display.epochs,   b.display.epochs],
    ['Batch Size',     a.config.data.batch_size,       b.config.data.batch_size],
    ['Input Size',     a.config.model.input_size + 'px',b.config.model.input_size + 'px'],
    ['Dropout',        a.config.model.dropout,         b.config.model.dropout],
    ['Classes',        a.config.data.classes.join(', '), b.config.data.classes.join(', ')],
    ['Augmentations',  augNames(a.config) || 'None', augNames(b.config) || 'None'],
    ['Seed',           a.config.training.seed ?? '—',      b.config.training.seed ?? '—'],
    ['Checkpoint Strategy', a.config.training.checkpoints.strategy, b.config.training.checkpoints.strategy],
    ['Optimizer',      fmtOptimizer(a.config.training.optimizer), fmtOptimizer(b.config.training.optimizer)],
    ['Loss',           fmtLoss(a.config.training.loss),           fmtLoss(b.config.training.loss)],
    ['LR Scheduler',   fmtScheduler(a.config.training.lr_scheduler), fmtScheduler(b.config.training.lr_scheduler)],
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
