/* ── Detection Evaluation tab ──────────────────────────────────────────────── */
//
// Detection evaluation is framed as two decomposed sub-problems:
//   * localization — mean IoU of matched boxes, AP@50/@75, recall-vs-IoU
//   * classification — a per-class outcome table (`class_breakdown`): for each
//     GT class, how many boxes were matched / class-confused / mis-located /
//     missed, plus duplicate/spurious strips keyed by predicted class. The
//     (N+1)×(N+1) confusion matrix (buildConfusionMatrix(), shared with the
//     classification tab) still renders below the table — it is the honest
//     view when class-for-class confusion is non-trivial, near-empty otherwise.
// Clicking a breakdown cell filters a gallery of GT-vs-prediction box overlays
// (solid vs dashed, via shared buildBoxLayer()/fitBoxOverlay()) to the samples
// tagged with that `<class>:<outcome>`.

// Which box layers the detection gallery draws. Toggled via the checkboxes in
// the gallery card header; persisted so the choice survives tab/run switches.
const _DET_LAYERS = { gt: true, pred: true };
try {
  const saved = JSON.parse(localStorage.getItem('cvbench.detLayers') || 'null');
  if (saved && typeof saved === 'object') Object.assign(_DET_LAYERS, saved);
} catch (e) { /* noop */ }
let _detCell = null;  // {t, p, n} of the open confusion-matrix cell, or
let _detTag = null;   // {tag, label, n} of the open breakdown-table cell

function toggleDetLayer(which, on) {
  _DET_LAYERS[which] = on;
  try { localStorage.setItem('cvbench.detLayers', JSON.stringify(_DET_LAYERS)); } catch (e) { /* noop */ }
  if (_detCell) showDetectionCellGallery(currentRun.name, _detCell.t, _detCell.p, _detCell.n);
  else if (_detTag) showDetectionTagGallery(currentRun.name, _detTag.tag, _detTag.label, _detTag.n);
}

function buildDetectionEvalTab(run, report) {
  const det = report.detection || {};
  const counts = det.counts || { tp: 0, fp: 0, fn: 0 };
  const loc = det.localization || {};
  const sweep = loc.recall_sweep || {};
  const rawSamples = report.samples || [];
  const staleSamples = rawSamples.length > 0 && !rawSamples.some(s => Array.isArray(s.cells));
  // The per-class breakdown galleries filter on sample `tags` and the `duplicate`
  // bucket — both added together. A report predating them still renders the
  // table but every cell click comes back empty, so fall back to the matrix.
  const breakdownReady = (det.class_breakdown || {}).duplicate !== undefined
    && rawSamples.some(s => Array.isArray(s.tags));

  let galleryCard;
  if (staleSamples) {
    galleryCard = `
      <article class="cm-card">
        <h4>Detections</h4>
        <p class="cm-hint">Re-run <kbd>evaluate</kbd> to use the detection sample browser.</p>
      </article>`;
  } else if (breakdownReady) {
    galleryCard = `
      <div class="cm-gallery-layout">
        <article class="cm-card">
          <h4>${escHtml(_present.sections.outcomes)}
            <small class="cm-hint">— what happened to every ground-truth box; click a number to browse those images</small>
          </h4>
          ${buildDetClassBreakdown(report, run.name)}
          <div class="det-cm-block">
            <h5 class="det-cm-heading">${escHtml(_present.sections.class_confusion)}
              <small class="cm-hint">— rows = true, columns = predicted; <code>background</code> = missed GT / spurious prediction</small>
            </h5>
            ${buildConfusionMatrix(report, run.name)}
          </div>
        </article>
        <div id="gallery-panel" class="gallery-panel" style="display:none"></div>
      </div>`;
  } else {
    galleryCard = `
      <div class="cm-gallery-layout">
        <article class="cm-card">
          <h4>Classification confusion
            <small class="cm-hint">— rows = true, columns = predicted;</small>
          </h4>
          <p class="cm-hint cm-hint--sub"><code>background</code> = missed GT / spurious prediction; click a cell to browse images</p>
          ${buildConfusionMatrix(report, run.name)}
        </article>
        <div id="gallery-panel" class="gallery-panel" style="display:none"></div>
      </div>`;
  }

  return `
    <div class="metric-cards">
      ${summaryCards(run)}
      ${card('conf ≥ / IoU ≥', `${det.conf_threshold} / ${det.iou_threshold}`)}
    </div>

    <article class="eval-table-card">
      <h4>${escHtml(_present.sections.localization)} <small class="cm-hint">— recall vs IoU threshold (are the boxes tight?)</small></h4>
      <div class="det-recall-sweep">
        <span>IoU ≥ 0.5 <strong>${fmtPct(sweep['0.5'])}</strong></span>
        <span>IoU ≥ 0.75 <strong>${fmtPct(sweep['0.75'])}</strong></span>
        <span>IoU ≥ 0.9 <strong>${fmtPct(sweep['0.9'])}</strong></span>
      </div>
    </article>

    <article class="eval-table-card">
      <h4>${escHtml(_present.sections.detection_quality)}
        <small class="cm-hint">— AP / precision / recall per class · ${counts.tp} TP · ${counts.fp} FP · ${counts.fn} FN, class-agnostic at IoU ≥ ${det.iou_threshold} (the outcome table below splits FP / FN by cause)</small>
      </h4>
      <div class="overflow-x">
        ${perClassTable(run.eval_display.per_class)}
      </div>
    </article>

    ${galleryCard}
  `;
}

// Does a gallery box belong to the clicked confusion-matrix cell (t → p)?
// `background` is the sentinel for the missed-GT column / spurious-pred row.
function _detBoxInCell(box, isPred, t, p) {
  if (isPred) {
    if (p === 'background') return false;                 // pred boxes never sit in the background column
    if (box.class !== p) return false;
    if (t === 'background') return box.match === 'background';
    return box.match === 'matched' && box.counterpart === t;
  }
  if (t === 'background') return false;                   // gt boxes never sit in the background row
  if (box.class !== t) return false;
  if (p === 'background') return box.match === 'background';
  return box.match === 'matched' && box.counterpart === p;
}

// IoU of two gallery boxes ({x, y, w, h}, normalized, top-left origin). Used to
// tell a mis-located prediction apart from an outright miss on the UI side —
// the server only stores a box's best IoU, not which cell GT it lines up with.
function _boxIoU(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = ix * iy;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

// Min IoU for a sub-threshold prediction to read as "the model found this, it's
// just badly localized" rather than "the model never fired here".
const _DET_LOC_FLOOR = 0.1;

// Splits an ambiguous `class → background` (missed GT) or `background → class`
// (spurious pred) sample into a sub-bucket, so the gallery can group the
// "detected but IoU < threshold" images apart from the "not detected" / "empty
// space" ones. Returns { kind, iou, hi }: `kind` is 'loc'|'gone' for a missed
// column and 'near'|'empty' for a spurious row; `iou` is the best pairing IoU;
// `hi` is the Set of prediction boxes to render undimmed on the thumb.
function detMissSplit(s, t, p) {
  const hi = new Set();
  let best = 0;
  if (p === 'background') {
    const cellGts = (s.gt || []).filter(b => _detBoxInCell(b, false, t, p));
    for (const pred of (s.pred || [])) {
      if (pred.class !== t) continue;
      for (const g of cellGts) {
        const v = _boxIoU(pred, g);
        if (v >= _DET_LOC_FLOOR) { hi.add(pred); best = Math.max(best, v); }
      }
    }
    return { kind: hi.size ? 'loc' : 'gone', iou: best, hi };
  }
  for (const pred of (s.pred || []).filter(b => _detBoxInCell(b, true, t, p))) {
    for (const g of (s.gt || [])) {
      if (g.class !== p) continue;
      const v = _boxIoU(pred, g);
      if (v >= _DET_LOC_FLOOR) { hi.add(pred); best = Math.max(best, v); }
    }
  }
  return { kind: hi.size ? 'near' : 'empty', iou: best, hi };
}

// Per-image metric shown under a detection gallery thumb, analogous to the
// classification gallery's confidence caption. Summarises the boxes in this
// image that belong to the clicked confusion-matrix cell (t → p).
function detSampleMetric(s, t, p) {
  if (t === 'background' || p === 'background') {
    const { kind, iou } = detMissSplit(s, t, p);
    if (kind === 'loc')  return `mis-located · IoU ${iou.toFixed(2)}`;
    if (kind === 'gone') return 'not detected';
    if (kind === 'near') return `near a real ${p} · IoU ${iou.toFixed(2)}`;
    if (kind === 'empty') {
      const preds = (s.pred || []).filter(b => _detBoxInCell(b, true, t, p));
      const c = preds.length ? preds.reduce((a, b) => a + (b.confidence || 0), 0) / preds.length : 0;
      return `spurious · conf ${(c * 100).toFixed(0)}%`;
    }
  }
  return detSampleMetricCell(s, t, p);
}

// The on-diagonal / class-confusion case (neither side is `background`): a
// matched prediction whose class the gallery is showing against its GT class.
function detSampleMetricCell(s, t, p) {
  const preds = (s.pred || []).filter(b => _detBoxInCell(b, true, t, p));
  if (preds.length === 0) return '—';
  const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
  const conf = mean(preds.map(b => b.confidence || 0)) * 100;
  const iou = mean(preds.map(b => b.iou || 0));
  return `conf ${conf.toFixed(0)}% · IoU ${iou.toFixed(2)}`;
}

function showDetectionCellGallery(runName, trueClass, predClass, count) {
  _detCell = { t: trueClass, p: predClass, n: count };
  const panel = document.getElementById('gallery-panel');
  if (!panel) return;
  if (count === 0) { panel.style.display = 'none'; _detCell = null; return; }

  const cellKey = JSON.stringify([trueClass, predClass]);
  const samples = (currentRun.eval_report.samples || [])
    .filter(s => Array.isArray(s.cells) && s.cells.some(c => JSON.stringify(c) === cellKey));

  let headLabel;
  if (trueClass === 'background')      headLabel = `Spurious <strong>${predClass}</strong> predictions (background → ${predClass})`;
  else if (predClass === 'background') headLabel = `Missed <strong>${trueClass}</strong> (${trueClass} → background)`;
  else if (trueClass === predClass)    headLabel = `<strong>${trueClass}</strong> — located &amp; classified correctly`;
  else                                 headLabel = `<strong>${trueClass}</strong> boxes predicted as <strong>${predClass}</strong>`;
  const label = `${headLabel} (${samples.length}${samples.length >= 20 ? '+' : ''})`;

  // A missed-GT / spurious-pred cell mixes two failure modes — the model
  // localised the object badly (IoU below threshold) vs. it never fired here at
  // all. Split those into labelled sub-groups; the near-miss prediction is kept
  // undimmed (`hi`) so the loose box reads against its GT at a glance.
  const isSplit = trueClass === 'background' || predClass === 'background';
  const splitOrder = predClass === 'background'
    ? [['loc', 'Detected — localization below threshold'], ['gone', 'Not detected']]
    : [['near', `Overlapping a real ${predClass}`], ['empty', 'In empty space']];

  const makeThumb = (s, i, hi) => {
    const imgSrc = `/api/runs/${encodeURIComponent(runName)}/images/${imgPath(s.path)}`;
    const overlayBoxes = [
      ...(_DET_LAYERS.gt ? (s.gt || []).map(b => ({ ...b, dim: !_detBoxInCell(b, false, trueClass, predClass) })) : []),
      ...(_DET_LAYERS.pred ? (s.pred || []).map(b => ({
        ...b, variant: 'pred',
        dim: hi.has(b) ? false : !_detBoxInCell(b, true, trueClass, predClass),
      })) : []),
    ];
    // The zoom modal carries every box (both layers, undimmed) and toggles
    // them itself, independent of the gallery's current layer filter.
    const modalBoxes = [
      ...(s.gt || []).map(b => ({ ...b })),
      ...(s.pred || []).map(b => ({ ...b, variant: 'pred' })),
    ];
    return buildGalleryThumb({
      id: `thumb-${Date.now()}-${i}`,
      imgSrc, path: s.path,
      boxes: overlayBoxes, modalBoxes,
      caption: escHtml(detSampleMetric(s, trueClass, predClass)),
    });
  };

  let body;
  if (samples.length === 0) {
    body = `<p class="gallery-empty">No sample images stored for this cell.</p>`;
  } else if (isSplit) {
    const tagged = samples.map(s => ({ s, split: detMissSplit(s, trueClass, predClass) }));
    body = splitOrder.map(([kind, title]) => {
      const group = tagged.filter(t => t.split.kind === kind);
      if (group.length === 0) return '';
      const thumbs = group.map(({ s, split }, i) => makeThumb(s, i, split.hi)).join('');
      return `<h6 class="gallery-subhead">${title} (${group.length})</h6>
              <div class="gallery-grid">${thumbs}</div>`;
    }).join('');
  } else {
    body = `<div class="gallery-grid">${samples.map((s, i) => makeThumb(s, i, new Set())).join('')}</div>`;
  }

  panel.innerHTML = `
    <div class="gallery-header">
      <h5>${label}</h5>
      <button class="outline" style="padding:0.2rem 0.6rem;font-size:0.8rem"
              onclick="document.getElementById('gallery-panel').style.display='none';_detCell=null;document.querySelectorAll('.cm-cell').forEach(c=>c.classList.remove('cm-cell--active'))">✕</button>
    </div>
    ${buildBoxLegend()}
    <div class="det-layer-toggles">
      <label><input type="checkbox" ${_DET_LAYERS.gt ? 'checked' : ''}
             onchange="toggleDetLayer('gt', this.checked)"> Ground truth</label>
      <label><input type="checkbox" ${_DET_LAYERS.pred ? 'checked' : ''}
             onchange="toggleDetLayer('pred', this.checked)"> Predictions</label>
    </div>
    ${body}
  `;
  panel.style.display = 'block';
  requestAnimationFrame(refitBoxOverlays);
}

/* ── Per-class outcome gallery (breakdown-table cell click) ────────────────── */

// Is this box the subject of the clicked breakdown cell `<cls>:<outcome>`?
// (`matched`/`mislocated`/`missed` key off the box's own class; `confused` off
// the GT class — `box.class` for a GT box, `box.counterpart` for a prediction;
// `duplicate`/`spurious` are prediction-only.)
function _detBoxInTag(box, isPred, cls, outcome) {
  if (box.outcome !== outcome) return false;
  if (outcome === 'spurious' || outcome === 'duplicate') return isPred && box.class === cls;
  if (outcome === 'confused') return isPred ? box.counterpart === cls : box.class === cls;
  return box.class === cls;
}

// Per-thumb caption for the outcome gallery, mirroring detSampleMetric.
function detTagCaption(s, cls, outcome) {
  const gt = (s.gt || []).filter(b => _detBoxInTag(b, false, cls, outcome));
  const pr = (s.pred || []).filter(b => _detBoxInTag(b, true, cls, outcome));
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  if (outcome === 'missed') return `${gt.length} not detected`;
  if (outcome === 'spurious') {
    return pr.length ? `spurious · conf ${(mean(pr.map(b => b.confidence || 0)) * 100).toFixed(0)}%` : '—';
  }
  if (outcome === 'duplicate') {
    const b = pr[0] || {};
    const parts = [];
    if (b.iou != null) parts.push(`IoU ${b.iou.toFixed(2)} vs object`);
    if (b.rival_iou != null) parts.push(`${b.rival_iou.toFixed(2)} vs kept box`);
    return `duplicate${parts.length ? ` · ${parts.join(' · ')}` : ' box'}`;
  }
  if (outcome === 'confused') {
    const as = [...new Set(pr.map(b => b.class))].join(', ');
    return as ? `predicted ${as}` : `${gt.length} confused`;
  }
  const ious = pr.map(b => b.iou).filter(v => v != null);
  const iouStr = ious.length ? ` · IoU ${mean(ious).toFixed(2)}` : '';
  return `${gt.length} ${outcome === 'matched' ? 'matched' : 'mis-located'}${iouStr}`;
}

function showDetectionTagGallery(runName, tag, label, count) {
  _detCell = null;
  _detTag = { tag, label, n: count };
  const panel = document.getElementById('gallery-panel');
  if (!panel) return;
  if (!count) { panel.style.display = 'none'; _detTag = null; return; }

  const [cls, outcome] = tag.split(':');
  const samples = (currentRun.eval_report.samples || [])
    .filter(s => Array.isArray(s.tags) && s.tags.includes(tag));

  const makeThumb = (s, i) => {
    const hi = new Set();
    for (const b of (s.gt || []))   if (_detBoxInTag(b, false, cls, outcome)) hi.add(b);
    for (const b of (s.pred || [])) if (_detBoxInTag(b, true, cls, outcome)) hi.add(b);
    // duplicate: also surface the object it's redundant with (the matched
    // same-class GT + prediction) so "already covered" is visible.
    if (outcome === 'duplicate') {
      for (const b of (s.gt || []))   if (b.class === cls && b.outcome === 'matched') hi.add(b);
      for (const b of (s.pred || [])) if (b.class === cls && b.outcome === 'matched') hi.add(b);
    }
    const imgSrc = `/api/runs/${encodeURIComponent(runName)}/images/${imgPath(s.path)}`;
    const overlayBoxes = [
      ...(_DET_LAYERS.gt ? (s.gt || []).map(b => ({ ...b, dim: !hi.has(b) })) : []),
      ...(_DET_LAYERS.pred ? (s.pred || []).map(b => ({ ...b, variant: 'pred', dim: !hi.has(b) })) : []),
    ];
    const modalBoxes = [
      ...(s.gt || []).map(b => ({ ...b })),
      ...(s.pred || []).map(b => ({ ...b, variant: 'pred' })),
    ];
    return buildGalleryThumb({
      id: `thumb-${Date.now()}-${i}`, imgSrc, path: s.path,
      boxes: overlayBoxes, modalBoxes,
      caption: escHtml(detTagCaption(s, cls, outcome)),
    });
  };

  let body;
  if (samples.length === 0) {
    body = `<p class="gallery-empty">No sample images stored for this outcome.</p>`;
  } else if (outcome === 'duplicate') {
    // Split by whether NMS would have removed the redundant box.
    const thr = (((currentRun.eval_report.detection || {}).class_breakdown || {}).dup_suppress_iou) ?? 0.5;
    const isSup = s => (s.pred || []).some(
      b => b.outcome === 'duplicate' && b.class === cls && (b.rival_iou ?? 0) >= thr);
    body = [
      ['NMS would remove these', `box overlaps the kept box ≥ ${thr}`, samples.filter(isSup)],
      ['Separate second box', 'low overlap with the kept box — a regression problem', samples.filter(s => !isSup(s))],
    ].map(([title, sub, grp]) => grp.length
      ? `<h6 class="gallery-subhead">${title} (${grp.length}) <span class="cm-hint">— ${sub}</span></h6>
         <div class="gallery-grid">${grp.map(makeThumb).join('')}</div>`
      : '').join('');
  } else {
    body = `<div class="gallery-grid">${samples.map(makeThumb).join('')}</div>`;
  }

  const shown = samples.length;
  const head = `${escHtml(label)} — <strong>${count}</strong> box${count === 1 ? '' : 'es'}`
    + `, ${shown} sample image${shown === 1 ? '' : 's'}`;
  panel.innerHTML = `
    <div class="gallery-header">
      <h5>${head}</h5>
      <button class="outline" style="padding:0.2rem 0.6rem;font-size:0.8rem"
              onclick="document.getElementById('gallery-panel').style.display='none';_detTag=null">✕</button>
    </div>
    ${buildBoxLegend()}
    <div class="det-layer-toggles">
      <label><input type="checkbox" ${_DET_LAYERS.gt ? 'checked' : ''}
             onchange="toggleDetLayer('gt', this.checked)"> Ground truth</label>
      <label><input type="checkbox" ${_DET_LAYERS.pred ? 'checked' : ''}
             onchange="toggleDetLayer('pred', this.checked)"> Predictions</label>
    </div>
    ${body}
  `;
  panel.style.display = 'block';
  requestAnimationFrame(refitBoxOverlays);
}

/* ── Per-class outcome breakdown ───────────────────────────────────────────── */

// The detection tab's headline classification view: one row per GT class, one
// column per outcome (`class_breakdown` from the backend), then two strips for
// predictions that belong to no GT — `duplicate` (a redundant box on an object
// already detected) and `spurious` (a box in the background). Every non-zero
// number is a gallery filter — clicking it opens the images tagged
// `<class>:<outcome>` / `duplicate:<class>` / `spurious:<class>`.
function buildDetClassBreakdown(report, runName) {
  const bd = (report.detection || {}).class_breakdown;
  if (!bd) return '<p class="cm-hint">No breakdown data — re-run <kbd>evaluate</kbd>.</p>';

  const cols = [
    ['matched',    'Matched',     'located & classified'],
    ['mislocated', 'Mis-located', `same class, IoU < threshold (≥ ${bd.loc_floor})`],
    ['confused',   'Confused',    'right box, wrong class'],
    ['missed',     'Missed',      'no prediction'],
  ];
  const rn = escHtml(runName);

  const cell = (cls, kind, v, extra = '') => {
    if (!v) return `<td class="det-bd-cell det-bd-zero">0</td>`;
    const lbl = escHtml(`${cls} — ${kind}`);
    return `<td class="det-bd-cell det-bd-${kind}"
      onclick="showDetectionTagGallery('${rn}', '${escHtml(cls + ':' + kind)}', '${lbl}', ${v})">
      <span class="det-bd-n">${v}</span>${extra}</td>`;
  };

  const rows = bd.classes.map((cls, i) => {
    const r = bd.rows[cls] || { instances: 0 };
    const inst = r.instances || 0;
    const confExtra = Object.keys(r.confused_as || {}).length
      ? `<small class="det-bd-sub">${Object.entries(r.confused_as)
          .map(([k, n]) => `→${escHtml(k)} ${n}`).join(', ')}</small>`
      : '';
    return `<tr>
      <th><span class="det-cls-dot" style="background:${boxColor(i)}"></span>${escHtml(cls)}</th>
      <td class="det-bd-inst">${inst}</td>
      ${cell(cls, 'matched', r.matched || 0)}
      ${cell(cls, 'mislocated', r.mislocated || 0)}
      ${cell(cls, 'confused', r.confused || 0, confExtra)}
      ${cell(cls, 'missed', r.missed || 0)}
    </tr>`;
  }).join('');

  const sum = src => bd.classes.reduce((a, c) => a + ((src || {})[c] || 0), 0);

  const strip = (kind, src, note) => {
    const items = bd.classes.map((cls, i) => {
      const v = (src || {})[cls] || 0;
      const dot = `<span class="det-cls-dot" style="background:${boxColor(i)}"></span>`;
      if (!v) return `<span class="det-spur-item det-bd-zero">${dot}${escHtml(cls)} 0</span>`;
      return `<span class="det-spur-item" style="border-color:${boxColor(i)}"
        onclick="showDetectionTagGallery('${rn}', '${escHtml(cls + ':' + kind)}', '${escHtml(kind + ' ' + cls)}', ${v})">
        ${dot}${escHtml(cls)} <strong>${v}</strong></span>`;
    }).join('');
    return `<li class="det-spur-strip">
      <span class="det-spur-label"><strong>${escHtml(kind)} ${sum(src)}</strong>
        <small class="cm-hint">— ${note}</small></span>
      <span class="det-spur-items">${items}</span>
    </li>`;
  };

  // How the headline FP / FN split by cause (answers "why don't these match
  // the numbers above?"). mis-located is on both lines: it is at once an
  // unmatched prediction and the unmatched GT it sits on.
  const counts = (report.detection || {}).counts || {};
  const b = k => bd.classes.reduce((a, c) => a + ((bd.rows[c] || {})[k] || 0), 0);
  const dupSup = sum(bd.duplicate_suppressible);
  const recon = (counts.fp != null && counts.fn != null) ? `
    <p class="det-reconcile">
      <strong>FP ${counts.fp}</strong> = ${b('mislocated')} mis-located + ${sum(bd.duplicate)} duplicate + ${sum(bd.spurious)} spurious
      &nbsp;·&nbsp;
      <strong>FN ${counts.fn}</strong> = ${b('missed')} missed + ${b('mislocated')} mis-located
    </p>` : '';

  const dupNote = bd.duplicate_suppressible !== undefined
    ? `a second box on an object already detected — ${dupSup} of ${sum(bd.duplicate)} would be removed by NMS at IoU ${bd.dup_suppress_iou}, the rest are separate boxes`
    : 'a second box on an object already detected';

  return `
    <div class="overflow-x">
      <table class="det-breakdown">
        <thead><tr>
          <th>Class</th><th>Instances</th>
          ${cols.map(([, h, tip]) => `<th title="${escHtml(tip)}">${h}</th>`).join('')}
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    ${recon}
    <div class="det-extra-preds">
      <p class="cm-hint cm-hint--sub">Extra predictions — boxes matched to no ground truth:</p>
      <ul class="det-extra-list">
        ${strip('duplicate', bd.duplicate, dupNote)}
        ${strip('spurious', bd.spurious, 'fired where there is no object of that class')}
      </ul>
    </div>
  `;
}
