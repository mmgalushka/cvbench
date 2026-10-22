/* ── Runs list ─────────────────────────────────────────────────────────────── */

async function showRunsList() {
  setActive('nav-runs');
  const page = document.getElementById('page');
  page.innerHTML = '<p aria-busy="true">Loading runs…</p>';
  try {
    const [runs, sweeps] = await Promise.all([
      api('/runs'),
      api('/sweeps').catch(() => []),  // degrade to runs only
      loadPresent(),
    ]);
    page.innerHTML = buildRunsList(runs, sweeps);
  } catch (e) {
    page.innerHTML = `<p class="error-msg">Failed to load runs: ${e.message}</p>`;
  }
}

/* ── Type icons: one flask = a run, three flasks = a sweep ─────────────────── */

const RUN_ICON = '<i class="fas fa-flask type-icon" role="img" title="Run" aria-label="Run"></i>';
const SWEEP_ICON = `<svg class="type-icon" width="20" height="16" viewBox="0 0 30 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" role="img" aria-label="Sweep"><title>Sweep</title><path d="M12 3h6M13.5 3v6L9 19a1.5 1.5 0 0 0 1.4 2h9.2a1.5 1.5 0 0 0 1.4-2l-4.5-10V3"/><path d="M6 10h-3M4.5 10v3.5L2 19.5a1 1 0 0 0 1 1.5h3"/><path d="M24 10h3M25.5 10v3.5L28 19.5a1 1 0 0 1-1 1.5h-3"/></svg>`;
const trialCount = n => n ? ` <small class="trial-count" title="${n} trial${n !== 1 ? 's' : ''}">×${n}</small>` : '';

// One cell of a run or sweep row. Every value is preformatted by the server
// (`r.display`, from core/present.py) — the same strings `runs list` prints.
function runCell(col, r) {
  const d = r.display;
  const isSweep = !!r.is_sweep;
  switch (col.key) {
    case 'name':
      return isSweep
        ? `<span class="type-name">${SWEEP_ICON}<strong>${escHtml(d.name)}</strong>${trialCount(r.n_trials)}</span>`
        : `<span class="type-name">${RUN_ICON}<strong>${escHtml(d.name)}</strong></span>`;
    case 'task':
      return `<span class="badge badge-task-${r.task || ''}">${escHtml(d.task)}</span>`;
    case 'status':
      return `<span class="badge badge-${r.status}">${escHtml(d.status)}</span>`;
    case 'test':
      return escHtml(d.test) + (d.test_label && d.test !== '—' ? ` <small class="ds-metric-label">${escHtml(d.test_label)}</small>` : '');
    default:
      return escHtml(d[col.key] ?? '—');
  }
}

function buildRunsList(runs, sweeps = []) {
  if (runs.length === 0 && sweeps.length === 0) {
    return `
      <div class="page-header"><h2>Runs</h2></div>
      <article><p>${escHtml(emptyText('runs', 'experiments'))}</p></article>
    `;
  }

  const columns = _present.run_columns;
  const items = [
    ...runs.map(r => ({ date: r.date || '', r })),
    ...sweeps.map(r => ({ date: r.date || '', r })),
  ].sort((a, b) => b.date.localeCompare(a.date));  // newest first, like `runs list`

  const rows = items.map(({ r }) => {
    const href = r.is_sweep ? `#/sweeps/${encodeURIComponent(r.name)}` : `#/runs/${encodeURIComponent(r.name)}`;
    return `
    <tr onclick="navigate('${href}')">
      ${columns.map(col => `<td${alignStyle(col)}>${runCell(col, r)}</td>`).join('')}
    </tr>
  `;
  }).join('');

  return `
    <div class="page-header"><h2>Runs <small style="font-size:0.85rem;font-weight:400;color:var(--muted-color)">${runs.length} run${runs.length !== 1 ? 's' : ''}${sweeps.length ? ` · ${sweeps.length} sweep${sweeps.length !== 1 ? 's' : ''}` : ''}</small></h2></div>
    <div class="overflow-x">
      <table class="runs-table">
        <thead>
          <tr>${columns.map(col => `<th${alignStyle(col)}>${escHtml(col.label)}</th>`).join('')}</tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

/* ── Sweep detail ──────────────────────────────────────────────────────────── */

async function showSweepDetail(name) {
  setActive('nav-runs');
  const page = document.getElementById('page');
  page.innerHTML = '<p aria-busy="true">Loading sweep…</p>';
  try {
    const sweep = await api(`/sweeps/${encodeURIComponent(name)}`);
    page.innerHTML = buildSweepDetail(sweep);
  } catch (e) {
    page.innerHTML = `<p class="error-msg">Sweep '${escHtml(name)}' not found: ${escHtml(e.message)}</p>`;
  }
}

function sweepStatus(trials) {
  if (trials.some(t => t.status === 'running')) return 'running';
  return 'done';
}

function buildSweepDetail(sw) {
  const axisNames = Object.keys(sw.axes);
  const best = sw.trials.find(t => t.is_best);
  const status = sweepStatus(sw.trials);
  const testMetric = (sw.trials.find(t => t.test_metric) || {}).test_metric;
  const testHeader = testMetric ? `test ${testMetric}` : 'test';

  const rows = sw.trials.map(t => {
    const openable = t.status !== 'missing';
    const nameCell = openable
      ? `<a href="#/runs/${encodeURIComponent(t.dir)}">${escHtml(t.dir)}</a>`
      : escHtml(t.dir);
    return `
      <tr class="${t.is_best ? 'row-best' : ''}">
        <td>${t.index}</td>
        <td>${nameCell}${t.is_best ? ' <span class="badge badge-best">best</span>' : ''}</td>
        ${axisNames.map(a => `<td>${escHtml(t.params[a] ?? '—')}</td>`).join('')}
        <td>${escHtml(t.display.value)}${t.is_best ? ' <span class="best-star">★</span>' : ''}</td>
        <td title="${t.is_best_test ? 'Best test score (indicator only; the sweep picks on val)' : ''}">${escHtml(t.display.test)}${t.is_best_test ? ' <span class="best-star">★</span>' : ''}</td>
        <td><span class="badge badge-${t.status}">${t.status}</span></td>
      </tr>`;
  }).join('');

  const n = sw.trials.length;
  return `
    <div class="page-header">
      <div>
        <a href="#/" class="back-link" onclick="setActive('nav-runs')">← Runs</a>
        <h2><span class="type-name">${SWEEP_ICON}<span>${escHtml(sw.name)}</span>${trialCount(sw.trials.length)}</span> <span class="badge badge-${status}">${status}</span></h2>
      </div>
      <div class="run-actions-menu" id="run-actions-menu">
        <button class="run-actions-trigger" onclick="toggleRunMenu(event)" title="More actions"><i class="fas fa-ellipsis-v"></i></button>
        <ul class="run-actions-dropdown" id="run-actions-dropdown">
          <li class="run-actions-item run-actions-item--danger" onclick="deleteRun('${escHtml(sw.name)}', ${n}); closeRunMenu()"><i class="fas fa-trash-alt"></i> Delete</li>
        </ul>
      </div>
    </div>

    <div class="metric-cards">
      ${card('Trials', n)}
      ${card('Metric', escHtml(sw.metric), sw.direction === 'min' ? 'lower is better' : 'higher is better')}
      ${card('Best trial', best ? fmtVal(best.value) : '—', best ? escHtml(best.dir) : '')}
      ${card('Strategy', escHtml(sw.strategy))}
      ${card('Data', escHtml(sw.data_dir || '—'))}
      ${card('Date', sw.date || '—')}
    </div>

    <p class="section-label">Axes: ${axisNames.map(a => `${escHtml(a)} = ${escHtml(sw.axes[a].join(', '))}`).join(' · ')}</p>
    <div class="overflow-x">
      <table class="runs-table">
        <thead>
          <tr>
            <th>#</th><th>Trial</th>
            ${axisNames.map(a => `<th>${escHtml(a)}</th>`).join('')}
            <th>${escHtml(sw.metric)}</th><th>${escHtml(testHeader)}</th><th>Status</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}
