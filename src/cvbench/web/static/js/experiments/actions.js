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
  const newName = prompt(`Rename run "${runName}" to:`, runName);
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

// nTrials > 0 marks a sweep; parentSweep is set when deleting one of its trials.
async function deleteRun(runName, nTrials = 0, parentSweep = null) {
  const what = nTrials ? `sweep "${runName}" and its ${nTrials} trial(s)` : `run "${runName}" and all its contents`;
  if (!confirm(`Permanently delete ${what}? This cannot be undone.`)) return;
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(runName)}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      alert(`Failed to delete run: ${err.detail || res.statusText}`);
      return;
    }
    navigate(parentSweep ? `#/sweeps/${encodeURIComponent(parentSweep)}` : '#/');
  } catch (e) {
    alert(`Failed to delete run: ${e.message}`);
  }
}
