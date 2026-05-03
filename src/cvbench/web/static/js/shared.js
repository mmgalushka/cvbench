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
  } else if (hash === '#/datasets') {
    showDatasetsList();
  } else if (hash.startsWith('#/datasets/')) {
    showDatasetGallery(decodeURIComponent(hash.slice(11)));
  }
}

function navigate(hash) {
  location.hash = hash;
}

/* ── API helper ────────────────────────────────────────────────────────────── */

async function api(path, { signal } = {}) {
  const res = await fetch('/api' + path, signal ? { signal } : {});
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

async function apiFetch(method, path, body) {
  const opts = { method };
  if (body !== undefined) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch('/api' + path, opts);
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

/* ── Shared formatters ─────────────────────────────────────────────────────── */

function fmtAcc(v) {
  return v != null ? (v * 100).toFixed(1) + '%' : '—';
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

function copyCliCommand(btn) {
  const cmd = btn.dataset.cmd;
  const flash = () => {
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = orig; }, 1500);
  };

  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(cmd).then(flash).catch(() => fallbackCopy(cmd, flash));
  } else {
    fallbackCopy(cmd, flash);
  }
}

function fallbackCopy(text, onSuccess) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.cssText = 'position:fixed;top:-9999px;left:-9999px;opacity:0';
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {
    if (document.execCommand('copy')) onSuccess();
  } finally {
    document.body.removeChild(ta);
  }
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
