"""
Minimal Flask app for human validation of LLM judge verdicts.

Usage:
    pip install flask
    python scripts/human_validation_ui.py [--csv PATH] [--port PORT]

Opens at http://localhost:5050
"""

import argparse
import csv
import os
from flask import Flask, jsonify, request, send_from_directory

_parser = argparse.ArgumentParser()
_parser.add_argument("--csv", default=os.path.join(
    os.path.dirname(__file__), "..", "data", "results", "human_validation_sample.csv"
))
_parser.add_argument("--port", type=int, default=5050)
_args, _ = _parser.parse_known_args()

CSV_PATH = os.path.abspath(_args.csv)
PORT = _args.port

app = Flask(__name__)


def read_csv():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows):
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@app.route("/")
def index():
    return HTML


@app.route("/api/rows")
def api_rows():
    return jsonify(read_csv())


@app.route("/api/verdict", methods=["POST"])
def api_verdict():
    data = request.json
    row_id = str(data["id"])
    verdict = data["verdict"]
    rows = read_csv()
    for row in rows:
        if row["id"] == row_id:
            row["human_verdict"] = verdict
            break
    write_csv(rows)
    return jsonify({"ok": True})


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MemLeak Human Validation</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f5; color: #1a1a1a; }

  /* Top bar */
  .topbar { background: #fff; border-bottom: 1px solid #ddd; padding: 12px 24px;
            display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 10; }
  .topbar h1 { font-size: 18px; font-weight: 600; }
  .progress { font-size: 14px; color: #666; }
  .progress-bar { width: 200px; height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; }
  .progress-bar-fill { height: 100%; background: #2563eb; transition: width 0.3s; }
  .filter-group { margin-left: auto; display: flex; gap: 8px; align-items: center; font-size: 13px; }
  .filter-group select { padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }

  /* Main layout */
  .container { max-width: 960px; margin: 0 auto; padding: 16px; }

  /* Card */
  .card { background: #fff; border: 2px solid #e0e0e0; border-radius: 8px;
          padding: 20px; margin-bottom: 12px; transition: border-color 0.2s; }
  .card.leaked-human { border-color: #ef4444; }
  .card.not-leaked-human { border-color: #22c55e; }
  .card.active { box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.3); }

  /* Card header */
  .card-header { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
  .badge { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px; text-transform: uppercase; }
  .badge-same_fact { background: #fef3c7; color: #92400e; }
  .badge-retained { background: #dbeafe; color: #1e40af; }
  .badge-negative { background: #f3e8ff; color: #6b21a8; }
  .badge-cat { background: #f0fdf4; color: #166534; }
  .badge-ensemble { font-size: 11px; padding: 2px 8px; border-radius: 10px; }
  .badge-ensemble.leaked { background: #fee2e2; color: #991b1b; }
  .badge-ensemble.not_leaked { background: #dcfce7; color: #166534; }
  .card-id { font-size: 12px; color: #999; margin-left: auto; }

  /* Fact */
  .fact-label { font-size: 11px; font-weight: 600; color: #666; text-transform: uppercase;
                letter-spacing: 0.05em; margin-bottom: 4px; }
  .fact-text { font-size: 15px; font-weight: 600; color: #1e40af; margin-bottom: 12px;
               padding: 8px 12px; background: #eff6ff; border-radius: 6px; }

  /* VLM response */
  .vlm-response { font-size: 13px; line-height: 1.6; color: #333; white-space: pre-wrap;
                  max-height: 200px; overflow-y: auto; padding: 12px; background: #fafafa;
                  border: 1px solid #eee; border-radius: 6px; margin-bottom: 12px; }
  .vlm-response.expanded { max-height: none; }

  .expand-btn { font-size: 12px; color: #2563eb; cursor: pointer; border: none;
                background: none; padding: 4px 0; margin-bottom: 12px; }

  /* Verdict buttons */
  .verdict-row { display: flex; gap: 8px; align-items: center; }
  .verdict-btn { padding: 6px 16px; border: 2px solid #ddd; border-radius: 6px;
                 font-size: 13px; font-weight: 600; cursor: pointer; background: #fff;
                 transition: all 0.15s; }
  .verdict-btn:hover { background: #f5f5f5; }
  .verdict-btn.leaked-active { background: #fee2e2; border-color: #ef4444; color: #991b1b; }
  .verdict-btn.not-leaked-active { background: #dcfce7; border-color: #22c55e; color: #166534; }
  .vote-info { font-size: 12px; color: #999; }

  /* Keyboard hint */
  .kbd { display: inline-block; padding: 1px 5px; font-size: 11px; font-family: monospace;
         background: #f0f0f0; border: 1px solid #ccc; border-radius: 3px; }

  /* Stats bar */
  .stats { display: flex; gap: 16px; font-size: 13px; color: #666; padding: 8px 0; }
</style>
</head>
<body>

<div class="topbar">
  <h1>MemLeak Human Validation</h1>
  <div class="progress" id="progress-text">0 / 100</div>
  <div class="progress-bar"><div class="progress-bar-fill" id="progress-fill"></div></div>
  <div class="filter-group">
    <label>Show:</label>
    <select id="filter-status">
      <option value="all">All</option>
      <option value="pending">Pending</option>
      <option value="done">Done</option>
      <option value="disagree">Disagreements</option>
    </select>
    <select id="filter-mode">
      <option value="all">All modes</option>
      <option value="same_fact">same_fact</option>
      <option value="retained">retained</option>
      <option value="negative">negative</option>
    </select>
    <span style="color:#999">
      <span class="kbd">L</span> leaked &nbsp;
      <span class="kbd">N</span> not leaked &nbsp;
      <span class="kbd">&uarr;&darr;</span> nav
    </span>
  </div>
</div>

<div class="container">
  <div class="stats" id="stats"></div>
  <div id="cards"></div>
</div>

<script>
let rows = [];
let activeIdx = 0;
let filteredIds = [];

async function load() {
  try {
    const res = await fetch('/api/rows');
    rows = await res.json();
    document.getElementById('cards').innerHTML = '<p>Loaded ' + rows.length + ' rows, rendering...</p>';
    render();
  } catch(e) {
    document.getElementById('cards').innerHTML = '<p style="color:red">Error loading data: ' + e.message + '</p>';
    console.error('Load error:', e);
  }
}

function getFiltered() {
  const status = document.getElementById('filter-status').value;
  const mode = document.getElementById('filter-mode').value;
  return rows.filter(r => {
    if (mode !== 'all' && r.mode !== mode) return false;
    if (status === 'pending' && r.human_verdict) return false;
    if (status === 'done' && !r.human_verdict) return false;
    if (status === 'disagree' && (!r.human_verdict || r.human_verdict === r.ensemble_verdict)) return false;
    return true;
  });
}

function render() {
  try { _render(); } catch(e) {
    document.getElementById('cards').innerHTML = '<p style="color:red">Render error: ' + e.message + '<br><pre>' + e.stack + '</pre></p>';
    console.error('Render error:', e);
  }
}
function _render() {
  const filtered = getFiltered();
  filteredIds = filtered.map(r => r.id);

  // Progress
  const done = rows.filter(r => r.human_verdict).length;
  document.getElementById('progress-text').textContent = done + ' / ' + rows.length;
  document.getElementById('progress-fill').style.width = (done / rows.length * 100) + '%';

  // Stats
  const humanLeaked = rows.filter(r => r.human_verdict === 'leaked').length;
  const humanNotLeaked = rows.filter(r => r.human_verdict === 'not_leaked').length;
  const agree = rows.filter(r => r.human_verdict && r.human_verdict === r.ensemble_verdict).length;
  const disagree = rows.filter(r => r.human_verdict && r.human_verdict !== r.ensemble_verdict).length;
  document.getElementById('stats').innerHTML =
    '<span>Validated: <b>' + done + '</b></span>' +
    '<span>Leaked: <b>' + humanLeaked + '</b></span>' +
    '<span>Not leaked: <b>' + humanNotLeaked + '</b></span>' +
    '<span>Agree: <b>' + agree + '</b></span>' +
    '<span>Disagree: <b style="color:#ef4444">' + disagree + '</b></span>' +
    (done > 0 ? '<span>Agreement: <b>' + (agree / done * 100).toFixed(1) + '%</b></span>' : '');

  // Cards
  const container = document.getElementById('cards');
  container.innerHTML = filtered.map((r, i) => {
    const isActive = filteredIds[activeIdx] === r.id;
    const humanClass = r.human_verdict === 'leaked' ? 'leaked-human' :
                       r.human_verdict === 'not_leaked' ? 'not-leaked-human' : '';
    return '<div class="card ' + humanClass + (isActive ? ' active' : '') + '" data-id="' + r.id + '" id="card-' + r.id + '">' +
      '<div class="card-header">' +
        '<span class="badge badge-' + r.mode + '">' + r.mode + '</span>' +
        '<span class="badge badge-cat">' + r.category + '</span>' +
        '<span class="badge badge-ensemble ' + r.ensemble_verdict + '">Ensemble: ' + r.ensemble_verdict + ' (' + r.vote_count + ')</span>' +
        '<span class="card-id">#' + r.id + ' &middot; ' + r.profile_id + '</span>' +
      '</div>' +
      '<div class="fact-label">Deleted fact</div>' +
      '<div class="fact-text">' + escHtml(r.fact_text) + '</div>' +
      '<div class="fact-label">VLM response</div>' +
      '<div class="vlm-response" id="vlm-' + r.id + '">' + escHtml(r.vlm_response) + '</div>' +
      '<button class="expand-btn" data-expand="' + r.id + '">Show more / less</button>' +
      '<div class="verdict-row">' +
        '<button class="verdict-btn' + (r.human_verdict === 'leaked' ? ' leaked-active' : '') + '" data-verdict="leaked" data-row="' + r.id + '">Leaked</button>' +
        '<button class="verdict-btn' + (r.human_verdict === 'not_leaked' ? ' not-leaked-active' : '') + '" data-verdict="not_leaked" data-row="' + r.id + '">Not Leaked</button>' +
        (r.human_verdict ? '<span class="vote-info">&check;</span>' : '') +
      '</div>' +
    '</div>';
  }).join('');
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function toggleExpand(id) {
  document.getElementById('vlm-' + id).classList.toggle('expanded');
}

async function setVerdict(id, verdict) {
  const row = rows.find(r => r.id === id);
  // Toggle off if clicking the same verdict
  if (row.human_verdict === verdict) {
    row.human_verdict = '';
    verdict = '';
  } else {
    row.human_verdict = verdict;
  }
  await fetch('/api/verdict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id, verdict})
  });
  render();
  scrollToActive();
}

function scrollToActive() {
  const id = filteredIds[activeIdx];
  const el = document.getElementById('card-' + id);
  if (el) el.scrollIntoView({behavior: 'smooth', block: 'center'});
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const id = filteredIds[activeIdx];
  if (!id) return;

  if (e.key === 'l' || e.key === 'L') {
    e.preventDefault();
    setVerdict(id, 'leaked');
    // Auto-advance to next pending
    advanceToNext();
  } else if (e.key === 'n' || e.key === 'N') {
    e.preventDefault();
    setVerdict(id, 'not_leaked');
    advanceToNext();
  } else if (e.key === 'ArrowDown' || e.key === 'j') {
    e.preventDefault();
    if (activeIdx < filteredIds.length - 1) { activeIdx++; render(); scrollToActive(); }
  } else if (e.key === 'ArrowUp' || e.key === 'k') {
    e.preventDefault();
    if (activeIdx > 0) { activeIdx--; render(); scrollToActive(); }
  } else if (e.key === 'e' || e.key === 'E') {
    e.preventDefault();
    toggleExpand(id);
  }
});

function advanceToNext() {
  // Move to next card after a brief delay so the UI updates
  setTimeout(() => {
    if (activeIdx < filteredIds.length - 1) {
      activeIdx++;
      render();
      scrollToActive();
    }
  }, 100);
}

document.getElementById('filter-status').addEventListener('change', () => { activeIdx = 0; render(); });
document.getElementById('filter-mode').addEventListener('change', () => { activeIdx = 0; render(); });

document.getElementById('cards').addEventListener('click', e => {
  const expandBtn = e.target.closest('[data-expand]');
  if (expandBtn) {
    toggleExpand(expandBtn.dataset.expand);
    return;
  }
  const verdictBtn = e.target.closest('[data-verdict]');
  if (verdictBtn) {
    setVerdict(verdictBtn.dataset.row, verdictBtn.dataset.verdict);
    return;
  }
});

load();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(f"Loading CSV from: {CSV_PATH}")
    print(f"Rows: {len(read_csv())}")
    print(f"Starting at http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
