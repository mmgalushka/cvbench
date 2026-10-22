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
            <select id="export-quantize" onchange="updateExportForm()">
              <option value="none">None (float32)</option>
              <option value="float16">Float16</option>
              <option value="int8">Int8</option>
            </select>
          </label>
          <label id="calib-strategy-wrap" style="display:none">
            Calib strategy
            <select id="export-calib-strategy" onchange="updateExportForm()">
              <option value="stratified">Stratified (recommended)</option>
              <option value="proportional">Proportional</option>
              <option value="equal">Equal per class</option>
              <option value="diverse">Diverse (k-means)</option>
            </select>
          </label>
          <label id="calib-total-wrap" style="display:none">
            Calib total
            <input id="export-calib-total" type="number" min="1" step="1" value="1024" oninput="updateExportForm()">
          </label>
        </div>
        <button id="export-generate-btn" onclick="generateExport('${escHtml(run.name)}')">Generate Export</button>
        <p id="export-error" class="error-msg" style="display:none;margin-top:0.5rem"></p>
        <div class="export-cli-alt">
          <span class="export-cli-alt-label">or run from terminal</span>
          <div class="cli-command-bar" id="export-cli-cmd">
            <svg class="cli-label" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
            <code class="cli-code">runs export ${escHtml(run.name)} --format tflite</code>
            <button class="cli-copy-btn" data-cmd="runs export ${escHtml(run.name)} --format tflite" onclick="copyCliCommand(this)">Copy</button>
          </div>
        </div>
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

function buildExportCliCmd(runName) {
  const fmt = document.getElementById('export-format')?.value || 'tflite';
  let cmd = `runs export ${runName} --format ${fmt}`;
  if (fmt === 'tflite') {
    const q = document.getElementById('export-quantize')?.value || 'none';
    if (q !== 'none') cmd += ` --quantize ${q}`;
  } else if (fmt === 'hailo') {
    const strategy = document.getElementById('export-calib-strategy')?.value || 'stratified';
    const total = document.getElementById('export-calib-total')?.value || '1024';
    cmd += ` --calib-strategy ${strategy} --calib-total ${total}`;
  }
  return cmd;
}

function updateExportForm() {
  const fmt = document.getElementById('export-format')?.value;
  const wrap = document.getElementById('quantize-wrap');
  if (wrap) wrap.style.display = fmt === 'tflite' ? '' : 'none';
  const calibStrategyWrap = document.getElementById('calib-strategy-wrap');
  if (calibStrategyWrap) calibStrategyWrap.style.display = fmt === 'hailo' ? '' : 'none';
  const calibTotalWrap = document.getElementById('calib-total-wrap');
  if (calibTotalWrap) calibTotalWrap.style.display = fmt === 'hailo' ? '' : 'none';

  const cliBar = document.getElementById('export-cli-cmd');
  if (cliBar && currentRun) {
    const cmd = buildExportCliCmd(currentRun.name);
    const code = cliBar.querySelector('.cli-code');
    const btn = cliBar.querySelector('.cli-copy-btn');
    if (code) code.textContent = cmd;
    if (btn) btn.dataset.cmd = cmd;
  }
}

function buildPlanInstructions(runName, cardMode = false) {
  const onnxExists = currentExports.some(e => e.format === 'onnx' || e.subfolder === 'onnx');
  const scp = `scp experiments/${runName}/export/plan/model.onnx user@jetson:/home/user/model.onnx`;
  const trtexec = `trtexec --onnx=model.onnx --saveEngine=model.plan --noTF32`;

  const termIcon = `<svg class="cli-label" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>`;

  const onnxWarning = onnxExists ? '' : `
    <div class="plan-warning">
      <strong>ONNX export not found.</strong> Generate it first:
      <div class="cli-command-bar plan-cmd-inline">
        ${termIcon}
        <code class="cli-code">runs export ${escHtml(runName)} --format onnx</code>
        <button class="cli-copy-btn" data-cmd="runs export ${escHtml(runName)} --format onnx" onclick="copyCliCommand(this)">Copy</button>
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
    </div>
  `;
}

function buildHailoInstructions(runName, cardMode = false) {
  const rel = `experiments/${runName}/export/hailo`;
  const termIcon = `<svg class="cli-label" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>`;


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
  const calib_strategy = document.getElementById('export-calib-strategy')?.value || 'proportional';
  const calibTotalRaw = document.getElementById('export-calib-total')?.value;
  const calib_total = calibTotalRaw ? parseInt(calibTotalRaw, 10) : 1024;
  if (!btn || !format) return;

  btn.disabled = true;
  btn.textContent = 'Generating…';
  btn.setAttribute('aria-busy', 'true');
  if (errEl) errEl.style.display = 'none';

  try {
    const body = { format, quantize };
    if (format === 'hailo') { body.calib_total = calib_total; body.calib_strategy = calib_strategy; }
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
