function buildTrainingTab(run) {
  const augText = augNames(run.config) || 'None';
  const tr = run.config.training;
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
          <dt>Input size</dt>    <dd>${run.config.model.input_size}px</dd>
          <dt>Dropout</dt>       <dd>${run.config.model.dropout}</dd>
          <dt>Fine-tune from</dt><dd>layer ${run.config.model.fine_tune_from_layer}</dd>
        </dl>
      </article>
      <article>
        <h4>Training</h4>
        <dl>
          <dt>Epochs</dt>         <dd>${run.display.epochs}</dd>
          <dt>Learning rate</dt>  <dd>${run.display.lr}</dd>
          <dt>Batch size</dt>     <dd>${run.config.data.batch_size}</dd>
          <dt>Seed</dt>           <dd>${run.config.training.seed != null ? run.config.training.seed : '—'}</dd>
          <dt>Class weight</dt>   <dd>${fmtClassWeight(run.config.training.class_weight)}</dd>
          <dt>Optimizer</dt>      <dd>${fmtOptimizer(tr.optimizer)}</dd>
          <dt>Loss</dt>           <dd>${fmtLoss(tr.loss)}</dd>
          <dt>LR scheduler</dt>   <dd>${fmtScheduler(tr.lr_scheduler)}</dd>
          <dt>Checkpoints</dt>    <dd>${tr.checkpoints.strategy}</dd>
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

  // Detection's training_log has no accuracy/val_accuracy columns (the model
  // isn't compiled with an "accuracy" metric) — fall back to a loss-only chart.
  const hasAccuracy = run.training_log[0].accuracy !== undefined;

  const labels   = run.training_log.map(r => `Ep ${r.epoch + 1}`);
  const trainLoss= run.training_log.map(r => +r.loss.toFixed(4));
  const valLoss  = run.training_log.map(r => +r.val_loss.toFixed(4));

  const datasets = [
    { label: 'Train Loss', data: trainLoss, borderColor: '#a78bfa', backgroundColor: 'transparent', yAxisID: 'yLoss', tension: 0.3, borderDash: hasAccuracy ? [5,3] : [] },
    { label: 'Val Loss',   data: valLoss,   borderColor: '#4ade80', backgroundColor: 'transparent', yAxisID: 'yLoss', tension: 0.3, borderDash: hasAccuracy ? [5,3] : [] },
  ];
  const scales = {
    yLoss: { type: 'linear', position: hasAccuracy ? 'right' : 'left', title: { display: true, text: 'Loss' }, grid: { drawOnChartArea: !hasAccuracy } },
  };

  if (hasAccuracy) {
    const trainAcc = run.training_log.map(r => +(r.accuracy   * 100).toFixed(2));
    const valAcc   = run.training_log.map(r => +(r.val_accuracy * 100).toFixed(2));
    datasets.unshift(
      { label: 'Train Acc', data: trainAcc, borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.07)', yAxisID: 'yAcc', tension: 0.3, fill: true },
      { label: 'Val Acc',   data: valAcc,   borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.07)',  yAxisID: 'yAcc', tension: 0.3, fill: true },
    );
    scales.yAcc = { type: 'linear', position: 'left', title: { display: true, text: 'Accuracy (%)' }, min: 0, max: 100 };
  }

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'top' } },
      scales,
    },
  });
}
