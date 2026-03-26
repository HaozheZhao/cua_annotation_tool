#!/usr/bin/env python3
"""
Lightweight export viewer — a simple web page to browse exported CUA annotation data.
Reads exported zip files or directories and serves them in a clean read-only web UI.

Usage:
    python export_viewer.py [--port 8080] [--data ./exports]

The --data directory should contain either:
  - Exported .zip files from the annotation tool
  - Extracted directories with export.json + events.jsonl + step_*.png
"""

import argparse
import json
import io
import zipfile
import base64
from pathlib import Path
from flask import Flask, Response, jsonify

app = Flask(__name__)
DATA_DIR = Path('./exports')

VIEWER_HTML = '''<!DOCTYPE html>
<html>
<head>
<title>CUA Export Viewer</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#0a0a14; color:#e0e0e0; }
.header { background:linear-gradient(135deg,#1a1a2e,#16213e); padding:14px 24px; border-bottom:2px solid #00d9ff; display:flex; justify-content:space-between; align-items:center; }
.header h1 { font-size:1.2em; color:#00d9ff; }
.header .stats { color:#888; font-size:0.85em; }
.sidebar { position:fixed; left:0; top:50px; bottom:0; width:280px; background:#12121e; border-right:1px solid #252542; overflow-y:auto; }
.case-item { padding:10px 14px; border-bottom:1px solid #1a1a2e; cursor:pointer; transition:all 0.2s; }
.case-item:hover { background:#1a1a2e; }
.case-item.active { background:#16213e; border-left:3px solid #00d9ff; }
.case-item .name { font-size:0.82em; color:#ccc; word-break:break-all; }
.case-item .meta { font-size:0.72em; color:#888; margin-top:3px; }
.case-item .badge { display:inline-block; padding:1px 6px; border-radius:3px; font-size:0.68em; font-weight:bold; margin-right:4px; }
.badge-pass { background:rgba(76,175,80,0.2); color:#4caf50; }
.badge-fail { background:rgba(244,67,54,0.2); color:#f44336; }
.badge-none { background:rgba(136,136,136,0.2); color:#888; }
.content { margin-left:280px; padding:20px; }
.info-bar { background:#16213e; border:1px solid #252542; border-radius:8px; padding:14px 18px; margin-bottom:16px; }
.info-bar .query { color:#00d9ff; font-size:1em; font-weight:bold; margin-bottom:8px; }
.info-bar .details { display:flex; gap:16px; flex-wrap:wrap; font-size:0.82em; color:#888; }
.info-bar .details b { color:#ccc; }
.step-card { background:#16213e; border:1px solid #252542; border-radius:8px; margin-bottom:12px; overflow:hidden; }
.step-header { padding:10px 14px; background:#1a1a2e; display:flex; justify-content:space-between; align-items:center; }
.step-header h3 { font-size:0.9em; color:#00d9ff; }
.step-header .badges { display:flex; gap:6px; }
.step-body { display:flex; gap:14px; padding:14px; }
.step-img { flex-shrink:0; }
.step-img img { max-width:500px; max-height:350px; border-radius:4px; border:1px solid #333; }
.step-info { flex:1; min-width:200px; }
.step-info .field { margin-bottom:8px; }
.step-info .field label { display:block; color:#888; font-size:0.72em; font-weight:bold; text-transform:uppercase; margin-bottom:2px; }
.step-info .field .val { font-size:0.85em; color:#ccc; }
.step-info .code { background:#0d1117; padding:6px 10px; border-radius:4px; font-family:monospace; font-size:0.82em; color:#7ee787; }
.step-info .justification { background:#1a1a2e; padding:6px 10px; border-radius:4px; font-size:0.82em; color:#ccc; border-left:3px solid #7c4dff; }
.empty { text-align:center; padding:60px; color:#555; }
.empty h2 { margin-bottom:10px; }
.search { width:100%; padding:8px 12px; background:#0d1117; border:1px solid #333; border-radius:4px; color:#e0e0e0; font-size:0.85em; margin-bottom:8px; }
.search:focus { border-color:#00d9ff; outline:none; }
</style>
</head>
<body>
<div class="header">
    <h1>CUA Export Viewer</h1>
    <div class="stats" id="stats"></div>
</div>
<div class="sidebar">
    <div style="padding:8px;"><input class="search" placeholder="Search cases..." oninput="filterCases(this.value)" /></div>
    <div id="caseList"></div>
</div>
<div class="content" id="content">
    <div class="empty"><h2>Select a case</h2><p>Choose a case from the sidebar to view its steps</p></div>
</div>

<script>
let allCases = [];
let currentCase = null;

async function init() {
    const resp = await fetch('/api/cases');
    allCases = await resp.json();
    document.getElementById('stats').textContent = allCases.length + ' cases loaded';
    renderCaseList(allCases);
    if (allCases.length > 0) loadCase(allCases[0].id);
}

function filterCases(q) {
    q = q.toLowerCase();
    const filtered = allCases.filter(c =>
        (c.query||'').toLowerCase().includes(q) ||
        (c.folder_name||'').toLowerCase().includes(q) ||
        (c.annotator||'').toLowerCase().includes(q)
    );
    renderCaseList(filtered);
}

function renderCaseList(cases) {
    let html = '';
    cases.forEach(c => {
        const badge = c.mark === 'pass' ? '<span class="badge badge-pass">PASS</span>' :
                      c.mark === 'fail' ? '<span class="badge badge-fail">FAIL</span>' :
                      '<span class="badge badge-none">-</span>';
        const active = currentCase === c.id ? ' active' : '';
        html += '<div class="case-item' + active + '" onclick="loadCase(\\''+c.id+'\\')">' +
            '<div class="name">' + (c.query || c.folder_name || c.id).substring(0,80) + '</div>' +
            '<div class="meta">' + badge + (c.annotator||'') + ' | ' + (c.step_count||0) + ' steps</div>' +
            '</div>';
    });
    document.getElementById('caseList').innerHTML = html || '<div class="empty" style="padding:20px;">No cases found</div>';
}

async function loadCase(id) {
    currentCase = id;
    renderCaseList(allCases.filter(c => document.querySelector('.search').value ? true : true));
    // Re-render sidebar to update active state
    const q = document.querySelector('.search').value.toLowerCase();
    const filtered = allCases.filter(c =>
        !q || (c.query||'').toLowerCase().includes(q) || (c.folder_name||'').toLowerCase().includes(q)
    );
    renderCaseList(filtered);

    const resp = await fetch('/api/case/' + encodeURIComponent(id));
    const data = await resp.json();
    renderCase(data);
}

function renderCase(data) {
    const info = data.export || {};
    let html = '<div class="info-bar">';
    html += '<div class="query">' + (info.query || info.instruction || 'No query') + '</div>';
    html += '<div class="details">';
    html += '<span><b>Annotator:</b> ' + (info.annotator || '-') + '</span>';
    html += '<span><b>Task ID:</b> ' + (info.task_id || '-') + '</span>';
    html += '<span><b>Mark:</b> ' + (info.mark || 'unreviewed') + '</span>';
    html += '<span><b>Steps:</b> ' + (data.steps || []).length + '</span>';
    if (info.pass_reason) html += '<span><b>Reason:</b> ' + info.pass_reason + '</span>';
    html += '</div></div>';

    (data.steps || []).forEach((step, i) => {
        html += '<div class="step-card">';
        html += '<div class="step-header"><h3>Step ' + i + ': ' + (step.action || '') + '</h3></div>';
        html += '<div class="step-body">';
        // Image
        if (step.has_image) {
            html += '<div class="step-img"><img src="/api/image/' + encodeURIComponent(data.id) + '/' + step.index + '" loading="lazy" /></div>';
        }
        // Info
        html += '<div class="step-info">';
        if (step.code) html += '<div class="field"><label>Code</label><div class="code">' + step.code + '</div></div>';
        if (step.description) html += '<div class="field"><label>Description</label><div class="val">' + step.description + '</div></div>';
        if (step.justification) html += '<div class="field"><label>Justification</label><div class="justification">' + step.justification + '</div></div>';
        if (step.coordinate) html += '<div class="field"><label>Coordinate</label><div class="val">(' + (step.coordinate.x||0) + ', ' + (step.coordinate.y||0) + ')</div></div>';
        html += '</div></div></div>';
    });

    document.getElementById('content').innerHTML = html;
}

init();
</script>
</body>
</html>'''


def scan_cases():
    """Scan data directory for exported cases (zips and directories)."""
    cases = []
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return cases

    # Scan zip files
    for zf_path in DATA_DIR.glob('*.zip'):
        try:
            with zipfile.ZipFile(zf_path) as zf:
                for name in zf.namelist():
                    if name.endswith('/export.json'):
                        folder = name.rsplit('/export.json', 1)[0]
                        export_data = json.loads(zf.read(name))
                        traj = export_data.get('traj', [])
                        cases.append({
                            'id': f'zip:{zf_path.name}:{folder}',
                            'source': str(zf_path),
                            'folder_name': folder,
                            'query': export_data.get('query', ''),
                            'annotator': export_data.get('annotator', ''),
                            'task_id': export_data.get('task_id', ''),
                            'mark': export_data.get('mark'),
                            'step_count': len(traj),
                        })
        except Exception as e:
            print(f"Error reading {zf_path}: {e}")

    # Scan extracted directories
    for export_json in DATA_DIR.glob('*/export.json'):
        try:
            with open(export_json) as f:
                export_data = json.load(f)
            folder = export_json.parent.name
            traj = export_data.get('traj', [])
            cases.append({
                'id': f'dir:{folder}',
                'source': str(export_json.parent),
                'folder_name': folder,
                'query': export_data.get('query', ''),
                'annotator': export_data.get('annotator', ''),
                'task_id': export_data.get('task_id', ''),
                'mark': export_data.get('mark'),
                'step_count': len(traj),
            })
        except Exception as e:
            print(f"Error reading {export_json}: {e}")

    return cases


def load_case_data(case_id):
    """Load full case data including steps."""
    cases = scan_cases()
    case = next((c for c in cases if c['id'] == case_id), None)
    if not case:
        return None

    if case_id.startswith('zip:'):
        _, zip_name, folder = case_id.split(':', 2)
        zf_path = DATA_DIR / zip_name
        with zipfile.ZipFile(zf_path) as zf:
            export_data = json.loads(zf.read(f'{folder}/export.json'))
            # Check which step images exist
            image_names = set(zf.namelist())
            traj = export_data.get('traj', [])
            steps = []
            for i, t in enumerate(traj):
                img_name = f"{folder}/step_{t.get('index', i)}.png"
                steps.append({
                    'index': t.get('index', i),
                    'action': t.get('action_type', ''),
                    'code': t.get('code', ''),
                    'description': t.get('description', ''),
                    'justification': t.get('justification', ''),
                    'coordinate': t.get('coordinate', {}),
                    'has_image': img_name in image_names,
                })
            return {'id': case_id, 'export': export_data, 'steps': steps}
    else:
        folder = case_id.split(':', 1)[1]
        case_dir = DATA_DIR / folder
        with open(case_dir / 'export.json') as f:
            export_data = json.load(f)
        traj = export_data.get('traj', [])
        steps = []
        for i, t in enumerate(traj):
            img_file = case_dir / f"step_{t.get('index', i)}.png"
            steps.append({
                'index': t.get('index', i),
                'action': t.get('action_type', ''),
                'code': t.get('code', ''),
                'description': t.get('description', ''),
                'justification': t.get('justification', ''),
                'coordinate': t.get('coordinate', {}),
                'has_image': img_file.exists(),
            })
        return {'id': case_id, 'export': export_data, 'steps': steps}


def get_step_image(case_id, step_index):
    """Get step screenshot as bytes."""
    if case_id.startswith('zip:'):
        _, zip_name, folder = case_id.split(':', 2)
        zf_path = DATA_DIR / zip_name
        with zipfile.ZipFile(zf_path) as zf:
            try:
                return zf.read(f'{folder}/step_{step_index}.png')
            except KeyError:
                return None
    else:
        folder = case_id.split(':', 1)[1]
        img_file = DATA_DIR / folder / f'step_{step_index}.png'
        if img_file.exists():
            return img_file.read_bytes()
    return None


@app.route('/')
def index():
    return VIEWER_HTML


@app.route('/api/cases')
def api_cases():
    return jsonify(scan_cases())


@app.route('/api/case/<path:case_id>')
def api_case(case_id):
    data = load_case_data(case_id)
    if not data:
        return jsonify({'error': 'Case not found'}), 404
    return jsonify(data)


@app.route('/api/image/<path:case_id>/<int:step_index>')
def api_image(case_id, step_index):
    img = get_step_image(case_id, step_index)
    if not img:
        return 'Image not found', 404
    return Response(img, mimetype='image/png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CUA Export Viewer')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--data', type=str, default='./exports', help='Directory with exported zips/folders')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    DATA_DIR = Path(args.data)
    print(f"Scanning {DATA_DIR} for exported cases...")
    cases = scan_cases()
    print(f"Found {len(cases)} cases")
    print(f"Starting viewer at http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
