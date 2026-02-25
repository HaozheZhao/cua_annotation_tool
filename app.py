#!/usr/bin/env python3
"""
CUA Annotation Tool - Human verification system for CUA-SFT data.
"""

import json
import os
import re
import csv
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, send_from_directory, jsonify, request, Response

app = Flask(__name__)

# Configuration - can be overridden via environment variables
DATA_DIR = Path(os.environ.get('CUA_DATA_DIR', './data'))
CSV_FILE = Path(os.environ.get('CUA_CSV_FILE', './task_assignments.csv'))
OUTPUT_DIR = Path(os.environ.get('CUA_OUTPUT_DIR', './output'))
ANNOTATIONS_FILE = Path(os.environ.get('CUA_ANNOTATIONS_FILE', './annotations.json'))
OSS_CACHE_DIR = Path(os.environ.get('CUA_OSS_CACHE', './oss_cache'))
REVIEW_STATUS_FILE = Path(os.environ.get('CUA_REVIEW_STATUS', './review_status.json'))

def parse_list_field(value):
    """Parse a comma-separated or JSON list field."""
    if not value:
        return []
    value = value.strip()
    if value.startswith('['):
        try:
            return json.loads(value)
        except:
            pass
    # Split by comma or semicolon
    items = [item.strip() for item in value.replace(';', ',').split(',')]
    return [item for item in items if item]

def load_tasks():
    """Load tasks from CSV file."""
    tasks = {}
    if CSV_FILE.exists():
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id = str(row.get('task_id', '')).strip()
                if task_id:
                    tasks[task_id] = {
                        'task_id': task_id,
                        'instruction': row.get('instruction', ''),
                        'worker_id': row.get('worker_id', ''),
                        'worker_name': row.get('worker_name', ''),
                        # New fields
                        'osworld_overlap': parse_list_field(row.get('osworld_overlap', '')),
                        'custom_nodes': parse_list_field(row.get('custom_nodes', '')),
                        'related_apps': parse_list_field(row.get('related_apps', ''))
                    }
    return tasks

def load_annotations():
    """Load existing annotations."""
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_annotations(annotations):
    """Save annotations to file."""
    ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

def get_step_annotations_file():
    return ANNOTATIONS_FILE.parent / 'step_annotations.json'

def load_step_annotations():
    """Load existing step annotations."""
    step_file = get_step_annotations_file()
    if step_file.exists():
        with open(step_file, 'r') as f:
            return json.load(f)
    return {}

def save_step_annotations(step_annotations):
    """Save step annotations to file."""
    step_file = get_step_annotations_file()
    step_file.parent.mkdir(parents=True, exist_ok=True)
    with open(step_file, 'w') as f:
        json.dump(step_annotations, f, indent=2, ensure_ascii=False)

def load_task_data(task_id):
    """Load data for a specific task."""
    task_dir = DATA_DIR / str(task_id)
    if not task_dir.exists():
        return None

    # Load metadata
    metadata_file = task_dir / 'metadata.json'
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

    video_start_ts = metadata.get('video_start_timestamp', 0)

    # Find video file and get resolution
    video_file = None
    video_path = None
    video_width = 1920  # Default
    video_height = 1080  # Default
    for f in task_dir.glob('*.mp4'):
        if 'video_clips' not in str(f):
            video_file = f.name
            video_path = f
            break

    # Get actual video resolution
    if video_path:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        cap.release()

    # Load events
    vis_file = task_dir / 'reduced_events_vis.jsonl'
    complete_file = task_dir / 'reduced_events_complete.jsonl'

    if not complete_file.exists():
        return None

    vis_events = []
    if vis_file.exists():
        with open(vis_file) as f:
            vis_events = [json.loads(line) for line in f if line.strip()]

    complete_events = []
    with open(complete_file) as f:
        complete_events = [json.loads(line) for line in f if line.strip()]

    # Build steps
    steps = []
    for i, ce in enumerate(complete_events):
        ve = vis_events[i] if i < len(vis_events) else {}

        # Get coordinates
        coord = ce.get('coordinate', {})
        x, y = coord.get('x', 0), coord.get('y', 0)

        # Calculate video time for screenshot
        start_time = ce.get('start_time', 0)
        end_time = ce.get('end_time', start_time)
        pre_move = ce.get('pre_move', {})

        if pre_move and pre_move.get('start_time') and pre_move.get('end_time'):
            pm_start = pre_move['start_time']
            pm_end = pre_move['end_time']
            pm_duration = pm_end - pm_start
            capture_time = pm_start + (pm_duration * 0.8)
        elif pre_move and pre_move.get('end_time'):
            capture_time = max(0, pre_move['end_time'] - 0.3)
        else:
            capture_time = max(0, start_time - 0.1)

        video_time = capture_time - video_start_ts

        # Build pyautogui code
        action = ce.get('action', '')
        description = ve.get('description', ce.get('description', ''))
        code = build_pyautogui_code(action, ce, description)

        # Only show coordinate marker for actions that have meaningful coordinates
        has_coordinate = action in ('click', 'drag') and (x != 0 or y != 0)

        steps.append({
            'index': i,
            'action': action,
            'description': description,
            'justification': ce.get('justification', ''),
            'code': code,
            'coordinate': {'x': x, 'y': y},
            'has_coordinate': has_coordinate,
            'video_time': video_time
        })

    return {
        'task_id': task_id,
        'video_file': video_file,
        'video_start_ts': video_start_ts,
        'video_width': video_width,
        'video_height': video_height,
        'steps': steps
    }

def build_pyautogui_code(action, event, description):
    """Build pyautogui code from event."""
    coord = event.get('coordinate', {})
    x, y = coord.get('x', 0), coord.get('y', 0)
    desc_lower = description.lower()

    if action == 'click':
        if 'double' in desc_lower:
            return f"pyautogui.doubleClick({x}, {y})"
        elif 'right' in desc_lower:
            return f"pyautogui.rightClick({x}, {y})"
        elif 'triple' in desc_lower:
            return f"pyautogui.click({x}, {y}, clicks=3)"
        return f"pyautogui.click({x}, {y})"
    elif action == 'drag':
        match = re.search(r'Drag from \((\d+),\s*(\d+)\) to \((\d+),\s*(\d+)\)', description)
        if match:
            return f"pyautogui.moveTo({match.group(1)}, {match.group(2)}); pyautogui.dragTo({match.group(3)}, {match.group(4)})"
        return f"pyautogui.drag({x}, {y})"
    elif action == 'scroll':
        down = re.search(r'⬇️×(\d+)', description)
        up = re.search(r'⬆️×(\d+)', description)
        total = 0
        if down: total -= int(down.group(1))
        if up: total += int(up.group(1))
        return f"pyautogui.scroll({total})"
    elif action == 'type':
        match = re.search(r'⌨️ Type: (.+)', description)
        if match:
            text = match.group(1)[:30] + '...' if len(match.group(1)) > 30 else match.group(1)
            return f"pyautogui.write('{text}')"
        return "pyautogui.write('...')"
    elif action == 'press':
        match = re.search(r'⌨️ Press: (.+)', description)
        if match:
            return f"pyautogui.hotkey({match.group(1)})"
        return "pyautogui.press('...')"
    return f"# {action}"

# ============================================================================
# OSS Review Status Persistence
# ============================================================================

def load_review_status():
    """Load review status from JSON file."""
    if REVIEW_STATUS_FILE.exists():
        with open(REVIEW_STATUS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_review_status(status):
    """Save review status to JSON file."""
    REVIEW_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)

def load_oss_task_data(local_dir):
    """Load task data from locally cached OSS recording files.
    Similar to load_task_data but reads from an arbitrary directory."""
    local_dir = Path(local_dir)
    if not local_dir.exists():
        return None

    # Load metadata
    metadata_file = local_dir / 'metadata.json'
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

    video_start_ts = metadata.get('video_start_timestamp', 0)

    # Find video file and get resolution
    video_file = None
    video_path = None
    video_width = 1920
    video_height = 1080
    for f in local_dir.glob('*.mp4'):
        if 'video_clips' not in str(f):
            video_file = f.name
            video_path = f
            break

    if video_path:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        cap.release()

    # Load events
    complete_file = local_dir / 'reduced_events_complete.jsonl'
    if not complete_file.exists():
        return None

    vis_file = local_dir / 'reduced_events_vis.jsonl'
    vis_events = []
    if vis_file.exists():
        with open(vis_file) as f:
            vis_events = [json.loads(line) for line in f if line.strip()]

    complete_events = []
    with open(complete_file) as f:
        complete_events = [json.loads(line) for line in f if line.strip()]

    steps = []
    for i, ce in enumerate(complete_events):
        ve = vis_events[i] if i < len(vis_events) else {}
        coord = ce.get('coordinate', {})
        x, y = coord.get('x', 0), coord.get('y', 0)

        start_time = ce.get('start_time', 0)
        pre_move = ce.get('pre_move', {})

        if pre_move and pre_move.get('start_time') and pre_move.get('end_time'):
            pm_start = pre_move['start_time']
            pm_end = pre_move['end_time']
            pm_duration = pm_end - pm_start
            capture_time = pm_start + (pm_duration * 0.8)
        elif pre_move and pre_move.get('end_time'):
            capture_time = max(0, pre_move['end_time'] - 0.3)
        else:
            capture_time = max(0, start_time - 0.1)

        video_time = capture_time - video_start_ts

        action = ce.get('action', '')
        description = ve.get('description', ce.get('description', ''))
        code = build_pyautogui_code(action, ce, description)
        has_coordinate = action in ('click', 'drag') and (x != 0 or y != 0)

        steps.append({
            'index': i,
            'action': action,
            'description': description,
            'justification': ce.get('justification', ''),
            'code': code,
            'coordinate': {'x': x, 'y': y},
            'has_coordinate': has_coordinate,
            'video_time': video_time
        })

    # Load annotator info
    annotator_info = {}
    ai_file = local_dir / 'annotator_info.json'
    if ai_file.exists():
        with open(ai_file) as f:
            annotator_info = json.load(f)

    # Load task name
    task_name = ""
    tn_file = local_dir / 'task_name.json'
    if tn_file.exists():
        with open(tn_file) as f:
            tn_data = json.load(f)
            task_name = tn_data.get('task_name', '')

    return {
        'video_file': video_file,
        'video_start_ts': video_start_ts,
        'video_width': video_width,
        'video_height': video_height,
        'steps': steps,
        'annotator_info': annotator_info,
        'task_name': task_name,
    }

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>CUA Annotation Tool</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #00d9ff;
        }
        .header h1 { font-size: 1.4em; color: #00d9ff; }
        .header-info { display: flex; gap: 20px; align-items: center; }
        .header-info span { color: #888; }
        .main-container {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        .sidebar {
            width: 280px;
            background: #16213e;
            overflow-y: auto;
            padding: 10px;
            border-right: 1px solid #333;
        }
        .task-item {
            padding: 10px 12px;
            margin: 4px 0;
            background: #1a1a2e;
            border-radius: 6px;
            cursor: pointer;
            border-left: 4px solid transparent;
            transition: all 0.2s;
        }
        .task-item:hover { background: #252542; }
        .task-item.active { border-left-color: #00d9ff; background: #252542; }
        .task-item.passed { border-left-color: #4caf50; }
        .task-item.failed { border-left-color: #f44336; }
        .task-item .task-id { color: #00d9ff; font-weight: bold; font-size: 0.9em; }
        .task-item .task-status {
            font-size: 0.75em;
            padding: 2px 6px;
            border-radius: 3px;
            margin-left: 8px;
        }
        .task-item .task-status.passed { background: #4caf50; color: white; }
        .task-item .task-status.failed { background: #f44336; color: white; }
        .task-item .task-status.pending { background: #666; color: white; }
        .content {
            flex: 1;
            padding: 15px 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }
        .instruction-box {
            background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #00d9ff;
        }
        .instruction-box h3 { color: #00d9ff; margin-bottom: 8px; font-size: 0.95em; }
        .instruction-box p { line-height: 1.5; font-size: 0.9em; }
        .annotation-panel {
            background: #1a1a2e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #333;
        }
        .annotation-panel h4 { color: #ffc107; margin-bottom: 12px; }
        .mark-buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 12px;
        }
        .mark-btn {
            padding: 10px 25px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
            flex: 1;
        }
        .mark-btn.pass { background: #2e7d32; color: white; }
        .mark-btn.pass:hover { background: #4caf50; }
        .mark-btn.pass.active { background: #4caf50; box-shadow: 0 0 10px #4caf50; }
        .mark-btn.fail { background: #c62828; color: white; }
        .mark-btn.fail:hover { background: #f44336; }
        .mark-btn.fail.active { background: #f44336; box-shadow: 0 0 10px #f44336; }
        .mark-btn.unclear { background: #555; color: white; }
        .mark-btn.unclear:hover { background: #777; }
        .mark-btn.unclear.active { background: #777; box-shadow: 0 0 10px #777; }
        .fail-reason {
            width: 100%;
            padding: 10px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #0f0f1a;
            color: #e0e0e0;
            resize: vertical;
            min-height: 60px;
        }
        .case-evaluation {
            background: linear-gradient(135deg, #1a2e1a 0%, #1a1a2e 100%);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #2e7d32;
        }
        .case-evaluation h4 { color: #4caf50; margin-bottom: 12px; }
        .case-evaluation.fail-mode {
            background: linear-gradient(135deg, #2e1a1a 0%, #1a1a2e 100%);
            border-color: #c62828;
        }
        .case-evaluation.fail-mode h4 { color: #f44336; }
        .rating-section {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 15px;
        }
        .rating-item {
            background: #0d1117;
            padding: 12px;
            border-radius: 6px;
        }
        .rating-item label {
            display: block;
            color: #888;
            font-size: 0.85em;
            margin-bottom: 8px;
        }
        .rating-item .label-cn {
            color: #666;
            font-size: 0.75em;
        }
        .rating-item .label-primary {
            color: #ffc107;
            font-size: 0.7em;
            margin-left: 5px;
        }
        .rating-stars {
            display: flex;
            gap: 4px;
        }
        .rating-star {
            width: 32px;
            height: 32px;
            border: none;
            background: #333;
            color: #666;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .rating-star:hover { background: #444; color: #aaa; }
        .rating-star.active { background: #ffc107; color: #000; }
        .pass-reason-input {
            width: 100%;
            padding: 10px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #0d1117;
            color: #e0e0e0;
            resize: vertical;
            min-height: 80px;
            font-family: inherit;
        }
        .pass-reason-input::placeholder { color: #555; }
        .knowledge-panel {
            background: linear-gradient(135deg, #1a1a3e 0%, #16213e 100%);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #4a4a8a;
        }
        .knowledge-panel h4 { color: #9c88ff; margin-bottom: 12px; }
        .knowledge-section {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 12px;
        }
        .knowledge-item {
            background: #0d1117;
            padding: 12px;
            border-radius: 6px;
        }
        .knowledge-item label {
            display: block;
            color: #9c88ff;
            font-size: 0.85em;
            margin-bottom: 8px;
            font-weight: bold;
        }
        .knowledge-item .label-hint {
            color: #666;
            font-size: 0.75em;
            font-weight: normal;
        }
        .knowledge-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 8px;
            min-height: 28px;
        }
        .knowledge-tag {
            display: inline-flex;
            align-items: center;
            background: #4a4a8a;
            color: white;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.8em;
        }
        .knowledge-tag .remove-tag {
            margin-left: 6px;
            cursor: pointer;
            opacity: 0.7;
        }
        .knowledge-tag .remove-tag:hover { opacity: 1; }
        .knowledge-input-row {
            display: flex;
            gap: 6px;
        }
        .knowledge-input {
            flex: 1;
            padding: 6px 10px;
            border: 1px solid #444;
            border-radius: 4px;
            background: #1a1a2e;
            color: #e0e0e0;
            font-size: 0.85em;
        }
        .knowledge-add-btn {
            padding: 6px 12px;
            background: #4a4a8a;
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
            font-size: 0.85em;
        }
        .knowledge-add-btn:hover { background: #5a5a9a; }
        .step-instructions-input {
            width: 100%;
            padding: 10px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #0d1117;
            color: #e0e0e0;
            resize: vertical;
            min-height: 60px;
            font-family: inherit;
            margin-top: 8px;
        }
        .step-nav {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }
        .step-nav button {
            padding: 8px 16px;
            background: #00d9ff;
            border: none;
            border-radius: 5px;
            color: #0f0f1a;
            font-weight: bold;
            cursor: pointer;
        }
        .step-nav button:hover { background: #00b8d4; }
        .step-nav button:disabled { background: #444; color: #888; cursor: not-allowed; }
        .step-info {
            background: #252542;
            padding: 6px 12px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        .step-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 12px;
        }
        .detail-box {
            background: #0d1117;
            padding: 10px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.85em;
        }
        .detail-box.code { color: #7ee787; }
        .detail-box.justification {
            color: #ffc107;
            grid-column: 1 / -1;
            border-left: 3px solid #ffc107;
        }
        .detail-box label {
            display: block;
            color: #888;
            font-size: 0.8em;
            margin-bottom: 5px;
            font-family: sans-serif;
        }
        .image-container {
            position: relative;
            display: inline-block;
            border: 2px solid #333;
            border-radius: 8px;
            overflow: hidden;
            max-width: 100%;
        }
        .image-container img {
            display: block;
            max-width: 100%;
            height: auto;
        }
        .coord-marker {
            position: absolute;
            width: 36px;
            height: 36px;
            border: 3px solid #ff0040;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            animation: pulse 1.5s infinite;
            box-shadow: 0 0 15px rgba(255, 0, 64, 0.5);
        }
        .coord-marker::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 10px;
            height: 10px;
            background: #ff0040;
            border-radius: 50%;
            transform: translate(-50%, -50%);
        }
        .coord-marker::after {
            content: attr(data-label);
            position: absolute;
            top: -28px;
            left: 50%;
            transform: translateX(-50%);
            background: #ff0040;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            white-space: nowrap;
        }
        @keyframes pulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
            50% { transform: translate(-50%, -50%) scale(1.2); opacity: 0.8; }
        }
        .progress-bar-container {
            background: #16213e;
            padding: 10px 20px;
            border-top: 1px solid #333;
        }
        .progress-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .progress-stats { font-size: 0.85em; color: #888; }
        .progress-stats span { margin-right: 15px; }
        .progress-stats .passed { color: #4caf50; }
        .progress-stats .failed { color: #f44336; }
        .progress-stats .pending { color: #888; }
        .finish-btn {
            padding: 8px 20px;
            background: #4caf50;
            border: none;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            cursor: pointer;
        }
        .finish-btn:hover { background: #66bb6a; }
        .finish-btn:disabled { background: #444; cursor: not-allowed; }
        .progress-bar {
            display: flex;
            height: 25px;
            border-radius: 4px;
            overflow: hidden;
            background: #333;
        }
        .progress-segment {
            height: 100%;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            color: white;
            min-width: 20px;
        }
        .progress-segment:hover {
            filter: brightness(1.2);
            transform: scaleY(1.1);
        }
        .progress-segment.passed { background: #4caf50; }
        .progress-segment.failed { background: #f44336; }
        .progress-segment.pending { background: #555; }
        .progress-segment.active {
            box-shadow: inset 0 0 0 2px #00d9ff;
        }
        .image-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding: 8px 12px;
            background: #1a1a2e;
            border-radius: 6px;
            border: 1px solid #333;
        }
        .resolution-info {
            color: #00d9ff;
            font-size: 0.85em;
            font-family: monospace;
        }
        .coord-adjust {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .coord-adjust label {
            color: #888;
            font-size: 0.85em;
        }
        .coord-input {
            width: 70px;
            padding: 5px 8px;
            border: 1px solid #444;
            border-radius: 4px;
            background: #0d1117;
            color: #e0e0e0;
            font-family: monospace;
            font-size: 0.9em;
            text-align: center;
        }
        .coord-input:focus {
            border-color: #00d9ff;
            outline: none;
        }
        .finetune-btn {
            padding: 5px 12px;
            background: #ffc107;
            border: none;
            border-radius: 4px;
            color: #000;
            font-weight: bold;
            cursor: pointer;
            font-size: 0.85em;
        }
        .finetune-btn:hover { background: #ffca28; }
        .finetune-btn.save { background: #4caf50; color: white; }
        .finetune-btn.save:hover { background: #66bb6a; }
        .finetune-btn.cancel { background: #666; color: white; }
        .finetune-btn.cancel:hover { background: #888; }
        .coord-adjust-panel {
            display: none;
            background: #252542;
            padding: 10px 15px;
            border-radius: 6px;
            margin-bottom: 10px;
            border: 1px solid #ffc107;
        }
        .coord-adjust-panel.show { display: block; }
        .coord-adjust-panel h5 {
            color: #ffc107;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        .coord-adjust-controls {
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        .coord-field {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .coord-field label {
            color: #888;
            font-size: 0.85em;
        }
        .adjusted-badge {
            display: inline-block;
            background: #ff9800;
            color: #000;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: bold;
            margin-left: 8px;
        }
        .original-coord {
            color: #888;
            font-size: 0.8em;
            text-decoration: line-through;
            margin-left: 8px;
        }
        .current-coord {
            color: #4caf50;
            font-weight: bold;
        }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .modal.show { display: flex; }
        .modal-content {
            background: #1a1a2e;
            padding: 30px;
            border-radius: 12px;
            max-width: 500px;
            text-align: center;
        }
        .modal-content h2 { color: #4caf50; margin-bottom: 20px; }
        .modal-content p { margin-bottom: 20px; color: #aaa; }
        .modal-buttons { display: flex; gap: 15px; justify-content: center; }
        .modal-btn {
            padding: 12px 30px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
        }
        .modal-btn.primary { background: #4caf50; color: white; }
        .modal-btn.secondary { background: #444; color: white; }
        .export-progress {
            margin: 20px 0;
        }
        .export-progress-bar {
            height: 20px;
            background: #333;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        .export-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            border-radius: 10px;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }
        .export-status {
            color: #888;
            font-size: 0.9em;
        }
        .export-path {
            background: #0d1117;
            padding: 12px;
            border-radius: 6px;
            margin: 15px 0;
            text-align: left;
            font-family: monospace;
            font-size: 0.85em;
            word-break: break-all;
        }
        .export-path label {
            color: #888;
            font-size: 0.8em;
            display: block;
            margin-bottom: 5px;
        }
        .export-path .path {
            color: #4caf50;
        }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #333;
            border-top: 3px solid #4caf50;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
            vertical-align: middle;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 CUA Annotation Tool</h1>
        <div class="header-info">
            <span id="current-task-info"></span>
        </div>
    </div>

    <div class="main-container">
        <div class="sidebar" id="sidebar"></div>
        <div class="content" id="content">
            <div style="text-align:center;padding:50px;color:#666;">
                Select a task from the sidebar to begin
            </div>
        </div>
    </div>

    <div class="progress-bar-container">
        <div class="progress-info">
            <div class="progress-stats">
                <span class="passed">✓ Passed: <b id="passed-count">0</b></span>
                <span class="failed">✗ Failed: <b id="failed-count">0</b></span>
                <span class="pending">○ Pending: <b id="pending-count">0</b></span>
            </div>
            <button class="finish-btn" id="finish-btn" onclick="showFinishModal()" disabled>
                Export Passed Cases
            </button>
        </div>
        <div class="progress-bar" id="progress-bar"></div>
    </div>

    <div class="modal" id="finish-modal">
        <div class="modal-content" style="max-width:600px;">
            <h2 id="export-title"><span class="spinner"></span>Exporting...</h2>
            <div class="export-progress" id="export-progress">
                <div class="export-progress-bar">
                    <div class="export-progress-fill" id="export-progress-fill" style="width:0%">0%</div>
                </div>
                <div class="export-status" id="export-status">Preparing export...</div>
            </div>
            <p id="export-message" style="display:none;"></p>
            <div class="export-path" id="export-paths" style="display:none;">
                <label>📁 Output Directory:</label>
                <div class="path" id="output-dir-path"></div>
                <label style="margin-top:10px;">📦 ZIP File:</label>
                <div class="path" id="zip-path"></div>
                <label style="margin-top:10px;">📄 JSON File:</label>
                <div class="path" id="json-path"></div>
            </div>
            <div class="modal-buttons" id="export-buttons" style="display:none;">
                <button class="modal-btn secondary" onclick="closeModal()">Close</button>
                <button class="modal-btn primary" onclick="downloadExport()">Download ZIP</button>
            </div>
        </div>
    </div>

    <script>
        let tasks = {};
        let taskOrder = [];
        let annotations = {};
        let stepAnnotations = {};  // Per-step annotations
        let coordAdjustments = {};  // Coordinate adjustments
        let originalCoords = {};  // Original coordinates before any adjustment
        let currentTaskId = null;
        let currentStepIndex = 0;
        let taskData = {};

        async function init() {
            const resp = await fetch('/api/init');
            const data = await resp.json();
            tasks = data.tasks;
            taskOrder = data.task_order;
            annotations = data.annotations;
            stepAnnotations = data.step_annotations || {};
            coordAdjustments = data.coord_adjustments || {};
            renderSidebar();
            renderProgressBar();
            if (taskOrder.length > 0) {
                selectTask(taskOrder[0]);
            }
        }

        function renderSidebar() {
            const sidebar = document.getElementById('sidebar');
            sidebar.innerHTML = taskOrder.map(tid => {
                const task = tasks[tid];
                const ann = annotations[tid] || {};
                let statusClass = 'pending';
                let statusText = 'Pending';
                if (ann.mark === 'pass') { statusClass = 'passed'; statusText = 'Passed'; }
                else if (ann.mark === 'fail') { statusClass = 'failed'; statusText = 'Failed'; }

                return `
                    <div class="task-item ${statusClass} ${tid === currentTaskId ? 'active' : ''}"
                         onclick="selectTask('${tid}')" id="task-item-${tid}">
                        <span class="task-id">Task ${tid}</span>
                        <span class="task-status ${statusClass}">${statusText}</span>
                        <div style="font-size:0.75em;color:#666;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                            ${task.instruction.substring(0, 40)}...
                        </div>
                    </div>
                `;
            }).join('');
        }

        function renderProgressBar() {
            const bar = document.getElementById('progress-bar');
            let passed = 0, failed = 0, pending = 0;

            bar.innerHTML = taskOrder.map(tid => {
                const ann = annotations[tid] || {};
                let cls = 'pending';
                if (ann.mark === 'pass') { cls = 'passed'; passed++; }
                else if (ann.mark === 'fail') { cls = 'failed'; failed++; }
                else { pending++; }

                return `<div class="progress-segment ${cls} ${tid === currentTaskId ? 'active' : ''}"
                            style="flex:1" onclick="selectTask('${tid}')" title="Task ${tid}">${tid}</div>`;
            }).join('');

            document.getElementById('passed-count').textContent = passed;
            document.getElementById('failed-count').textContent = failed;
            document.getElementById('pending-count').textContent = pending;

            document.getElementById('finish-btn').disabled = pending > 0;
        }

        async function selectTask(taskId) {
            currentTaskId = taskId;
            currentStepIndex = 0;

            document.querySelectorAll('.task-item').forEach(el => el.classList.remove('active'));
            document.getElementById(`task-item-${taskId}`)?.classList.add('active');

            if (!taskData[taskId]) {
                const resp = await fetch(`/api/task/${taskId}`);
                const data = await resp.json();
                taskData[taskId] = data;

                // Store original coordinates for each step (from coordAdjustments if exists)
                for (const step of data.steps) {
                    const key = `${taskId}_${step.index}`;
                    if (coordAdjustments[key]) {
                        // There was an adjustment, store original from adjustment record
                        if (!originalCoords[key]) {
                            originalCoords[key] = coordAdjustments[key].original || { x: step.coordinate.x, y: step.coordinate.y };
                        }
                    }
                }
            }

            document.getElementById('current-task-info').textContent =
                `Task ${taskId} | Worker: ${tasks[taskId].worker_name || 'N/A'}`;

            renderContent();
            renderProgressBar();
        }

        function renderContent() {
            if (!currentTaskId || !taskData[currentTaskId]) return;

            const task = tasks[currentTaskId];
            const data = taskData[currentTaskId];
            const ann = annotations[currentTaskId] || {};
            const step = data.steps[currentStepIndex];
            const totalSteps = data.steps.length;

            // Get actual video resolution
            const videoWidth = data.video_width || 1920;
            const videoHeight = data.video_height || 1080;

            const displayWidth = Math.min(1200, window.innerWidth - 350);
            const scale = displayWidth / videoWidth;
            const displayHeight = Math.round(videoHeight * scale);
            const markerX = step.coordinate.x * scale;
            const markerY = step.coordinate.y * scale;

            // Check if this step has been adjusted
            const adjKey = `${currentTaskId}_${currentStepIndex}`;
            const isAdjusted = !!coordAdjustments[adjKey];
            const origCoord = isAdjusted ? (coordAdjustments[adjKey].original || { x: 0, y: 0 }) : step.coordinate;

            // Get scores from annotation
            const scores = ann.scores || {};

            document.getElementById('content').innerHTML = `
                <div class="instruction-box">
                    <h3>Task ${currentTaskId} - Instruction:</h3>
                    <p>${task.instruction}</p>
                </div>

                <div class="annotation-panel">
                    <h4>📋 Case Verdict (案例结论)</h4>
                    <div class="mark-buttons">
                        <button class="mark-btn pass ${ann.mark === 'pass' ? 'active' : ''}" onclick="setMark('pass')">
                            ✓ PASS (通过)
                        </button>
                        <button class="mark-btn fail ${ann.mark === 'fail' ? 'active' : ''}" onclick="setMark('fail')">
                            ✗ FAIL (失败)
                        </button>
                        <button class="mark-btn unclear ${!ann.mark ? 'active' : ''}" onclick="setMark(null)">
                            ? UNCLEAR (待定)
                        </button>
                    </div>
                </div>

                <div class="case-evaluation ${ann.mark === 'fail' ? 'fail-mode' : ''}">
                    <h4>📝 Case Evaluation (案例评估) ${ann.mark === 'fail' ? '- Fail Reason' : '- Pass Reason'}</h4>

                    <div class="rating-section">
                        <div class="rating-item">
                            <label>Correctness (正确性) <span class="label-primary">★ Primary Focus</span></label>
                            <div class="rating-stars">
                                ${[1,2,3,4,5].map(n => `<button class="rating-star ${(scores.correctness || 0) >= n ? 'active' : ''}"
                                    onclick="setCaseRating('correctness', ${n})">${n}</button>`).join('')}
                            </div>
                        </div>
                        <div class="rating-item">
                            <label>Difficulty (难度)</label>
                            <div class="rating-stars">
                                ${[1,2,3,4,5].map(n => `<button class="rating-star ${(scores.difficulty || 0) >= n ? 'active' : ''}"
                                    onclick="setCaseRating('difficulty', ${n})">${n}</button>`).join('')}
                            </div>
                        </div>
                        <div class="rating-item">
                            <label>Knowledge Richness (知识丰富程度)</label>
                            <div class="rating-stars">
                                ${[1,2,3,4,5].map(n => `<button class="rating-star ${(scores.knowledge_richness || 0) >= n ? 'active' : ''}"
                                    onclick="setCaseRating('knowledge_richness', ${n})">${n}</button>`).join('')}
                            </div>
                        </div>
                        <div class="rating-item">
                            <label>Task Value (题目价值)</label>
                            <div class="rating-stars">
                                ${[1,2,3,4,5].map(n => `<button class="rating-star ${(scores.task_value || 0) >= n ? 'active' : ''}"
                                    onclick="setCaseRating('task_value', ${n})">${n}</button>`).join('')}
                            </div>
                        </div>
                    </div>

                    <label style="color:#888;font-size:0.85em;display:block;margin-bottom:8px;">
                        ${ann.mark === 'fail' ? 'Fail Reason (失败理由):' : 'Pass Reason (通过理由):'}
                    </label>
                    <textarea class="pass-reason-input" id="pass-reason"
                              placeholder="${ann.mark === 'fail' ? 'Describe why this case failed...' : 'e.g., Step-by-step aligns with screenshots; actions valid; final state verified.'}"
                              onchange="updatePassReason(this.value)">${ann.pass_reason || ''}</textarea>
                </div>

                <div class="knowledge-panel">
                    <h4>📚 Knowledge Points & Related Apps (知识点与相关应用)</h4>
                    <div class="knowledge-section">
                        <div class="knowledge-item">
                            <label>OSWorld Overlap <span class="label-hint">(与OSWorld重叠)</span></label>
                            <div class="knowledge-tags" id="osworld-tags">
                                ${(ann.osworld_overlap || task.osworld_overlap || []).map(tag =>
                                    `<span class="knowledge-tag">${tag}<span class="remove-tag" onclick="removeTag('osworld_overlap', '${tag}')">×</span></span>`
                                ).join('')}
                            </div>
                            <div class="knowledge-input-row">
                                <input type="text" class="knowledge-input" id="osworld-input" placeholder="Add tag...">
                                <button class="knowledge-add-btn" onclick="addTag('osworld_overlap', 'osworld-input')">+</button>
                            </div>
                        </div>
                        <div class="knowledge-item">
                            <label>Custom Nodes <span class="label-hint">(自定义节点)</span></label>
                            <div class="knowledge-tags" id="custom-tags">
                                ${(ann.custom_nodes || task.custom_nodes || []).map(tag =>
                                    `<span class="knowledge-tag">${tag}<span class="remove-tag" onclick="removeTag('custom_nodes', '${tag}')">×</span></span>`
                                ).join('')}
                            </div>
                            <div class="knowledge-input-row">
                                <input type="text" class="knowledge-input" id="custom-input" placeholder="Add tag...">
                                <button class="knowledge-add-btn" onclick="addTag('custom_nodes', 'custom-input')">+</button>
                            </div>
                        </div>
                        <div class="knowledge-item">
                            <label>Related Apps <span class="label-hint">(相关应用)</span></label>
                            <div class="knowledge-tags" id="apps-tags">
                                ${(ann.related_apps || task.related_apps || []).map(tag =>
                                    `<span class="knowledge-tag">${tag}<span class="remove-tag" onclick="removeTag('related_apps', '${tag}')">×</span></span>`
                                ).join('')}
                            </div>
                            <div class="knowledge-input-row">
                                <input type="text" class="knowledge-input" id="apps-input" placeholder="Add app...">
                                <button class="knowledge-add-btn" onclick="addTag('related_apps', 'apps-input')">+</button>
                            </div>
                        </div>
                    </div>
                    <label style="color:#888;font-size:0.85em;display:block;margin-bottom:4px;">
                        Step-by-Step Instructions (分步说明):
                    </label>
                    <textarea class="step-instructions-input" id="step-instructions"
                              placeholder="Describe the step-by-step instructions for this task..."
                              onchange="updateStepInstructions(this.value)">${ann.step_by_step_instructions || ''}</textarea>
                </div>

                <div class="step-nav">
                    <button onclick="prevStep()" ${currentStepIndex === 0 ? 'disabled' : ''}>← Prev</button>
                    <span class="step-info">Step ${currentStepIndex + 1} / ${totalSteps}</span>
                    <button onclick="nextStep()" ${currentStepIndex >= totalSteps - 1 ? 'disabled' : ''}>Next →</button>
                    <button onclick="currentStepIndex=0;renderContent()">⏮ First</button>
                    <button onclick="currentStepIndex=${totalSteps-1};renderContent()">Last ⏭</button>
                </div>

                <div class="step-details">
                    <div class="detail-box code">
                        <label>Action Code:</label>
                        <span id="action-code-display">${step.code}</span>
                        ${isAdjusted ? '<span class="adjusted-badge">已微调 Adjusted</span>' : ''}
                    </div>
                    <div class="detail-box">
                        <label>Coordinates:</label>
                        ${isAdjusted ?
                            `<span class="current-coord" id="coord-display">(${step.coordinate.x}, ${step.coordinate.y})</span>
                             <span class="original-coord">原始 (${origCoord.x}, ${origCoord.y})</span>` :
                            `<span id="coord-display">(${step.coordinate.x}, ${step.coordinate.y})</span>`
                        }
                        ${step.has_coordinate ? `<button class="finetune-btn" onclick="toggleFinetune()" style="margin-left:10px;">🎯 微调 Fine-tune</button>` : ''}
                    </div>
                    <div class="detail-box justification">
                        <label>💡 Original Justification (原始理由):</label>
                        ${step.justification || '<em style="color:#666">No justification provided</em>'}
                    </div>
                </div>

                ${step.has_coordinate ? `
                <div class="coord-adjust-panel" id="finetune-panel">
                    <h5>🎯 Coordinate Fine-tuning (坐标微调)</h5>
                    <div class="coord-adjust-controls">
                        <div class="coord-field">
                            <label>X:</label>
                            <input type="number" class="coord-input" id="coord-x" value="${step.coordinate.x}"
                                   min="0" max="${videoWidth}" oninput="updateMarkerPreview()">
                        </div>
                        <div class="coord-field">
                            <label>Y:</label>
                            <input type="number" class="coord-input" id="coord-y" value="${step.coordinate.y}"
                                   min="0" max="${videoHeight}" oninput="updateMarkerPreview()">
                        </div>
                        <button class="finetune-btn save" onclick="saveCoordinate()">✓ Save</button>
                        <button class="finetune-btn cancel" onclick="cancelFinetune()">✗ Cancel</button>
                        <span style="color:#888;font-size:0.8em;margin-left:10px;">Click on image to set position</span>
                    </div>
                </div>
                ` : ''}

                <div class="image-info">
                    <div class="resolution-info">
                        📐 Resolution: <b>${videoWidth} × ${videoHeight}</b> | Display: <b>${displayWidth} × ${displayHeight}</b> | Scale: <b>${(scale * 100).toFixed(1)}%</b>
                    </div>
                </div>

                <div class="image-container" id="image-container" style="max-width:${displayWidth}px"
                     onclick="handleImageClick(event)" data-scale="${scale}" data-width="${videoWidth}" data-height="${videoHeight}">
                    <img src="/frame/${currentTaskId}/${step.video_time}" style="width:100%">
                    ${step.has_coordinate ? `<div class="coord-marker" id="coord-marker" style="left:${markerX}px;top:${markerY}px"
                         data-label="${step.action} (${step.coordinate.x},${step.coordinate.y})"></div>` : ''}
                </div>
            `;

            // Store original coordinates and video dimensions for cancel
            window.originalCoords = { x: step.coordinate.x, y: step.coordinate.y };
            window.videoDimensions = { width: videoWidth, height: videoHeight };
        }

        async function setMark(mark) {
            if (!currentTaskId) return;

            if (!annotations[currentTaskId]) annotations[currentTaskId] = {};
            annotations[currentTaskId].mark = mark;

            await saveAnnotation();

            renderContent();
            renderSidebar();
            renderProgressBar();
        }

        async function setCaseRating(dimension, value) {
            if (!currentTaskId) return;
            if (!annotations[currentTaskId]) annotations[currentTaskId] = {};
            if (!annotations[currentTaskId].scores) annotations[currentTaskId].scores = {};
            annotations[currentTaskId].scores[dimension] = value;
            await saveAnnotation();
            renderContent();
        }

        async function updatePassReason(reason) {
            if (!currentTaskId) return;
            if (!annotations[currentTaskId]) annotations[currentTaskId] = {};
            annotations[currentTaskId].pass_reason = reason;
            await saveAnnotation();
        }

        async function updateStepInstructions(text) {
            if (!currentTaskId) return;
            if (!annotations[currentTaskId]) annotations[currentTaskId] = {};
            annotations[currentTaskId].step_by_step_instructions = text;
            await saveAnnotation();
        }

        async function addTag(field, inputId) {
            if (!currentTaskId) return;
            const input = document.getElementById(inputId);
            const value = input.value.trim();
            if (!value) return;

            if (!annotations[currentTaskId]) annotations[currentTaskId] = {};
            if (!annotations[currentTaskId][field]) {
                // Initialize from task data if available
                const task = tasks[currentTaskId];
                annotations[currentTaskId][field] = [...(task[field] || [])];
            }

            // Add if not already exists
            if (!annotations[currentTaskId][field].includes(value)) {
                annotations[currentTaskId][field].push(value);
                await saveAnnotation();
                renderContent();
            }
            input.value = '';
        }

        async function removeTag(field, value) {
            if (!currentTaskId) return;
            if (!annotations[currentTaskId]) annotations[currentTaskId] = {};
            if (!annotations[currentTaskId][field]) {
                const task = tasks[currentTaskId];
                annotations[currentTaskId][field] = [...(task[field] || [])];
            }

            const idx = annotations[currentTaskId][field].indexOf(value);
            if (idx > -1) {
                annotations[currentTaskId][field].splice(idx, 1);
                await saveAnnotation();
                renderContent();
            }
        }

        async function saveAnnotation() {
            if (!currentTaskId) return;
            const ann = annotations[currentTaskId] || {};
            const task = tasks[currentTaskId] || {};
            await fetch('/api/annotate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    task_id: currentTaskId,
                    mark: ann.mark,
                    pass_reason: ann.pass_reason || '',
                    scores: ann.scores || {},
                    step_by_step_instructions: ann.step_by_step_instructions || '',
                    osworld_overlap: ann.osworld_overlap || task.osworld_overlap || [],
                    custom_nodes: ann.custom_nodes || task.custom_nodes || [],
                    related_apps: ann.related_apps || task.related_apps || []
                })
            });
        }

        // Coordinate fine-tuning functions
        function toggleFinetune() {
            const panel = document.getElementById('finetune-panel');
            if (panel) {
                panel.classList.toggle('show');
            }
        }

        function updateMarkerPreview() {
            const xInput = document.getElementById('coord-x');
            const yInput = document.getElementById('coord-y');
            const marker = document.getElementById('coord-marker');
            const container = document.getElementById('image-container');

            if (!xInput || !yInput || !marker || !container) return;

            const x = parseInt(xInput.value) || 0;
            const y = parseInt(yInput.value) || 0;
            const scale = parseFloat(container.dataset.scale) || 1;

            // Update marker position
            marker.style.left = (x * scale) + 'px';
            marker.style.top = (y * scale) + 'px';

            // Update marker label
            const data = taskData[currentTaskId];
            const step = data.steps[currentStepIndex];
            marker.setAttribute('data-label', `${step.action} (${x},${y})`);

            // Update coordinate display
            const coordDisplay = document.getElementById('coord-display');
            if (coordDisplay) {
                coordDisplay.textContent = `(${x}, ${y})`;
            }

            // Update action code display
            updateActionCodeDisplay(x, y);
        }

        function updateActionCodeDisplay(x, y) {
            const data = taskData[currentTaskId];
            const step = data.steps[currentStepIndex];
            const codeDisplay = document.getElementById('action-code-display');
            if (!codeDisplay) return;

            // Update the code based on action type
            let newCode = step.code;
            if (step.action === 'click') {
                if (step.code.includes('doubleClick')) {
                    newCode = `pyautogui.doubleClick(${x}, ${y})`;
                } else if (step.code.includes('rightClick')) {
                    newCode = `pyautogui.rightClick(${x}, ${y})`;
                } else if (step.code.includes('clicks=3')) {
                    newCode = `pyautogui.click(${x}, ${y}, clicks=3)`;
                } else {
                    newCode = `pyautogui.click(${x}, ${y})`;
                }
            } else if (step.action === 'drag') {
                // Keep the dragTo part, update moveTo
                const match = step.code.match(/dragTo\((\d+),\s*(\d+)\)/);
                if (match) {
                    newCode = `pyautogui.moveTo(${x}, ${y}); pyautogui.dragTo(${match[1]}, ${match[2]})`;
                }
            }
            codeDisplay.textContent = newCode;
        }

        async function saveCoordinate() {
            const xInput = document.getElementById('coord-x');
            const yInput = document.getElementById('coord-y');
            if (!xInput || !yInput) return;

            const x = parseInt(xInput.value) || 0;
            const y = parseInt(yInput.value) || 0;

            const adjKey = `${currentTaskId}_${currentStepIndex}`;
            const step = taskData[currentTaskId].steps[currentStepIndex];

            // Get original coordinates (before any adjustment)
            let origX, origY;
            if (coordAdjustments[adjKey] && coordAdjustments[adjKey].original) {
                // Already adjusted before, keep original
                origX = coordAdjustments[adjKey].original.x;
                origY = coordAdjustments[adjKey].original.y;
            } else {
                // First time adjusting, current is original
                origX = window.originalCoords.x;
                origY = window.originalCoords.y;
            }

            // Save to server with original coordinates
            await fetch('/api/update_coordinate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    task_id: currentTaskId,
                    step_index: currentStepIndex,
                    x: x,
                    y: y,
                    original_x: origX,
                    original_y: origY
                })
            });

            // Update local coordAdjustments
            coordAdjustments[adjKey] = {
                task_id: currentTaskId,
                step_index: currentStepIndex,
                x: x,
                y: y,
                original: { x: origX, y: origY }
            };

            // Update local task data
            step.coordinate = { x, y };

            // Update the code in local data
            let newCode = step.code;
            if (step.action === 'click') {
                if (step.code.includes('doubleClick')) {
                    newCode = `pyautogui.doubleClick(${x}, ${y})`;
                } else if (step.code.includes('rightClick')) {
                    newCode = `pyautogui.rightClick(${x}, ${y})`;
                } else if (step.code.includes('clicks=3')) {
                    newCode = `pyautogui.click(${x}, ${y}, clicks=3)`;
                } else {
                    newCode = `pyautogui.click(${x}, ${y})`;
                }
            } else if (step.action === 'drag') {
                const match = step.code.match(/dragTo\((\d+),\s*(\d+)\)/);
                if (match) {
                    newCode = `pyautogui.moveTo(${x}, ${y}); pyautogui.dragTo(${match[1]}, ${match[2]})`;
                }
            }
            step.code = newCode;

            // Hide panel and refresh
            toggleFinetune();
            renderContent();
        }

        function cancelFinetune() {
            // Restore original coordinates
            const data = taskData[currentTaskId];
            const step = data.steps[currentStepIndex];

            const xInput = document.getElementById('coord-x');
            const yInput = document.getElementById('coord-y');
            if (xInput) xInput.value = window.originalCoords.x;
            if (yInput) yInput.value = window.originalCoords.y;

            updateMarkerPreview();
            toggleFinetune();
        }

        function handleImageClick(event) {
            const panel = document.getElementById('finetune-panel');
            if (!panel || !panel.classList.contains('show')) return;

            const container = document.getElementById('image-container');
            const rect = container.getBoundingClientRect();
            const scale = parseFloat(container.dataset.scale) || 1;
            const videoWidth = parseInt(container.dataset.width) || 1920;
            const videoHeight = parseInt(container.dataset.height) || 1080;

            // Calculate click position relative to image
            const clickX = event.clientX - rect.left;
            const clickY = event.clientY - rect.top;

            // Convert to original resolution coordinates
            const x = Math.round(clickX / scale);
            const y = Math.round(clickY / scale);

            // Clamp values to actual video dimensions
            const clampedX = Math.max(0, Math.min(videoWidth, x));
            const clampedY = Math.max(0, Math.min(videoHeight, y));

            // Update inputs
            const xInput = document.getElementById('coord-x');
            const yInput = document.getElementById('coord-y');
            if (xInput) xInput.value = clampedX;
            if (yInput) yInput.value = clampedY;

            // Update preview
            updateMarkerPreview();
        }

        function prevStep() {
            if (currentStepIndex > 0) {
                currentStepIndex--;
                renderContent();
            }
        }

        function nextStep() {
            const data = taskData[currentTaskId];
            if (currentStepIndex < data.steps.length - 1) {
                currentStepIndex++;
                renderContent();
            }
        }

        async function showFinishModal() {
            // Reset modal to loading state
            document.getElementById('export-title').innerHTML = '<span class="spinner"></span>Exporting...';
            document.getElementById('export-progress').style.display = 'block';
            document.getElementById('export-progress-fill').style.width = '0%';
            document.getElementById('export-progress-fill').textContent = '0%';
            document.getElementById('export-status').textContent = 'Preparing export...';
            document.getElementById('export-message').style.display = 'none';
            document.getElementById('export-paths').style.display = 'none';
            document.getElementById('export-buttons').style.display = 'none';

            // Show modal
            document.getElementById('finish-modal').classList.add('show');

            // Simulate progress while exporting
            let progress = 0;
            const progressInterval = setInterval(() => {
                if (progress < 90) {
                    progress += Math.random() * 15;
                    progress = Math.min(90, progress);
                    document.getElementById('export-progress-fill').style.width = progress + '%';
                    document.getElementById('export-progress-fill').textContent = Math.round(progress) + '%';
                    if (progress < 30) {
                        document.getElementById('export-status').textContent = 'Extracting screenshots...';
                    } else if (progress < 60) {
                        document.getElementById('export-status').textContent = 'Building JSON files...';
                    } else {
                        document.getElementById('export-status').textContent = 'Creating ZIP archive...';
                    }
                }
            }, 200);

            try {
                const resp = await fetch('/api/export', {method: 'POST'});
                const result = await resp.json();

                clearInterval(progressInterval);

                // Complete progress
                document.getElementById('export-progress-fill').style.width = '100%';
                document.getElementById('export-progress-fill').textContent = '100%';
                document.getElementById('export-status').textContent = 'Export complete!';

                // Show success state
                setTimeout(() => {
                    document.getElementById('export-title').innerHTML = '✅ Export Complete!';
                    document.getElementById('export-progress').style.display = 'none';
                    document.getElementById('export-message').textContent = result.message;
                    document.getElementById('export-message').style.display = 'block';

                    // Show paths
                    document.getElementById('output-dir-path').textContent = result.output_dir;
                    document.getElementById('zip-path').textContent = result.zip_path;
                    document.getElementById('json-path').textContent = result.json_path;
                    document.getElementById('export-paths').style.display = 'block';

                    document.getElementById('export-buttons').style.display = 'flex';
                }, 500);
            } catch (err) {
                clearInterval(progressInterval);
                document.getElementById('export-title').innerHTML = '❌ Export Failed';
                document.getElementById('export-status').textContent = 'Error: ' + err.message;
                document.getElementById('export-buttons').style.display = 'flex';
            }
        }

        function closeModal() {
            document.getElementById('finish-modal').classList.remove('show');
        }

        function downloadExport() {
            window.location.href = '/api/download';
        }

        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'TEXTAREA') return;
            if (e.key === 'ArrowLeft') prevStep();
            if (e.key === 'ArrowRight') nextStep();
            if (e.key === '1') setMark('pass');
            if (e.key === '2') setMark('fail');
            if (e.key === '3') setMark(null);
        });

        window.addEventListener('resize', () => {
            if (currentTaskId) renderContent();
        });

        init();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/init')
def api_init():
    tasks = load_tasks()
    annotations = load_annotations()
    step_annotations = load_step_annotations()
    task_order = sorted(tasks.keys(), key=lambda x: int(x) if x.isdigit() else x)

    # Load coordinate adjustments
    coord_file = OUTPUT_DIR / 'coordinate_adjustments.json'
    coord_adjustments = {}
    if coord_file.exists():
        with open(coord_file, 'r') as f:
            coord_adjustments = json.load(f)

    return jsonify({
        'tasks': tasks,
        'task_order': task_order,
        'annotations': annotations,
        'step_annotations': step_annotations,
        'coord_adjustments': coord_adjustments
    })

@app.route('/api/task/<task_id>')
def api_task(task_id):
    data = load_task_data(task_id)
    if not data:
        return jsonify({'error': 'Task not found'}), 404

    # Apply coordinate adjustments
    coord_file = OUTPUT_DIR / 'coordinate_adjustments.json'
    if coord_file.exists():
        with open(coord_file, 'r') as f:
            coord_adjustments = json.load(f)

        for step in data['steps']:
            adj_key = f"{task_id}_{step['index']}"
            if adj_key in coord_adjustments:
                adj = coord_adjustments[adj_key]
                step['coordinate'] = {'x': adj['x'], 'y': adj['y']}
                # Update code with adjusted coordinates
                x, y = adj['x'], adj['y']
                code = step['code']
                if 'doubleClick' in code:
                    step['code'] = f"pyautogui.doubleClick({x}, {y})"
                elif 'rightClick' in code:
                    step['code'] = f"pyautogui.rightClick({x}, {y})"
                elif 'clicks=3' in code:
                    step['code'] = f"pyautogui.click({x}, {y}, clicks=3)"
                elif 'click' in code.lower() and 'pyautogui.click' in code:
                    step['code'] = f"pyautogui.click({x}, {y})"
                elif 'moveTo' in code and 'dragTo' in code:
                    match = re.search(r'dragTo\((\d+),\s*(\d+)\)', code)
                    if match:
                        step['code'] = f"pyautogui.moveTo({x}, {y}); pyautogui.dragTo({match.group(1)}, {match.group(2)})"

    return jsonify(data)

@app.route('/api/annotate', methods=['POST'])
def api_annotate():
    data = request.json
    task_id = data.get('task_id')
    mark = data.get('mark')
    pass_reason = data.get('pass_reason', '')
    scores = data.get('scores', {})
    step_by_step_instructions = data.get('step_by_step_instructions', '')
    osworld_overlap = data.get('osworld_overlap', [])
    custom_nodes = data.get('custom_nodes', [])
    related_apps = data.get('related_apps', [])

    annotations = load_annotations()
    annotations[task_id] = {
        'mark': mark,
        'pass_reason': pass_reason,
        'scores': scores,
        'step_by_step_instructions': step_by_step_instructions,
        'osworld_overlap': osworld_overlap,
        'custom_nodes': custom_nodes,
        'related_apps': related_apps
    }
    save_annotations(annotations)

    return jsonify({'success': True})

@app.route('/api/step_annotate', methods=['POST'])
def api_step_annotate():
    data = request.json
    step_key = data.get('step_key')
    annotation = data.get('annotation', {})

    step_annotations = load_step_annotations()
    step_annotations[step_key] = annotation
    save_step_annotations(step_annotations)

    return jsonify({'success': True})

@app.route('/api/update_coordinate', methods=['POST'])
def api_update_coordinate():
    """Update coordinate for a specific step."""
    data = request.json
    task_id = str(data.get('task_id'))
    step_index = data.get('step_index')
    new_x = data.get('x')
    new_y = data.get('y')
    original_x = data.get('original_x')
    original_y = data.get('original_y')

    # Load and update the coordinate adjustments file
    coord_file = OUTPUT_DIR / 'coordinate_adjustments.json'
    coord_file.parent.mkdir(parents=True, exist_ok=True)

    coord_adjustments = {}
    if coord_file.exists():
        with open(coord_file, 'r') as f:
            coord_adjustments = json.load(f)

    # Store the adjustment with original coordinates
    key = f"{task_id}_{step_index}"

    # Preserve original coordinates if this step was adjusted before
    if key in coord_adjustments and 'original' in coord_adjustments[key]:
        original = coord_adjustments[key]['original']
    else:
        original = {'x': original_x, 'y': original_y}

    coord_adjustments[key] = {
        'task_id': task_id,
        'step_index': step_index,
        'x': new_x,
        'y': new_y,
        'original': original
    }

    with open(coord_file, 'w') as f:
        json.dump(coord_adjustments, f, indent=2)

    return jsonify({'success': True})

@app.route('/frame/<task_id>/<float:video_time>')
def serve_frame(task_id, video_time):
    """Extract and serve a frame from the video at the specified time."""
    import cv2
    import tempfile

    task_dir = DATA_DIR / str(task_id)
    video_file = None
    for f in task_dir.glob('*.mp4'):
        if 'video_clips' not in str(f):
            video_file = f
            break

    if not video_file:
        return 'Video not found', 404

    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_num = int(video_time * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = max(0, min(frame_num, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return 'Frame extraction failed', 500

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/api/export', methods=['POST'])
def api_export():
    """Export passed cases to the required format."""
    annotations = load_annotations()
    tasks = load_tasks()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load coordinate adjustments
    coord_file = OUTPUT_DIR / 'coordinate_adjustments.json'
    coord_adjustments = {}
    if coord_file.exists():
        with open(coord_file, 'r') as f:
            coord_adjustments = json.load(f)

    passed_tasks = [tid for tid, ann in annotations.items() if ann.get('mark') == 'pass']

    import cv2

    all_data = []
    for task_id in passed_tasks:
        task_info = tasks.get(task_id, {})
        task_data = load_task_data(task_id)
        ann = annotations.get(task_id, {})
        if not task_data:
            continue

        task_output_dir = OUTPUT_DIR / f'task_{task_id}'
        task_output_dir.mkdir(exist_ok=True)

        # Extract video file
        task_dir = DATA_DIR / str(task_id)
        video_file = None
        for f in task_dir.glob('*.mp4'):
            if 'video_clips' not in str(f):
                video_file = f
                break

        if not video_file:
            continue

        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        traj = []
        for step in task_data['steps']:
            frame_num = int(step['video_time'] * fps)
            frame_num = max(0, min(frame_num, total_frames - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            screenshot_name = f"step_{step['index']}.png"
            if ret:
                cv2.imwrite(str(task_output_dir / screenshot_name), frame)

            # Check for coordinate adjustments
            adj_key = f"{task_id}_{step['index']}"
            code = step['code']
            if adj_key in coord_adjustments:
                adj = coord_adjustments[adj_key]
                x, y = adj['x'], adj['y']
                # Update code with adjusted coordinates
                if 'doubleClick' in code:
                    code = f"pyautogui.doubleClick({x}, {y})"
                elif 'rightClick' in code:
                    code = f"pyautogui.rightClick({x}, {y})"
                elif 'clicks=3' in code:
                    code = f"pyautogui.click({x}, {y}, clicks=3)"
                elif 'click' in code.lower() and 'pyautogui.click' in code:
                    code = f"pyautogui.click({x}, {y})"
                elif 'moveTo' in code and 'dragTo' in code:
                    match = re.search(r'dragTo\((\d+),\s*(\d+)\)', code)
                    if match:
                        code = f"pyautogui.moveTo({x}, {y}); pyautogui.dragTo({match.group(1)}, {match.group(2)})"

            traj.append({
                'index': step['index'],
                'action_type': step['action'],
                'code': code,
                'screenshot': f"step_{step['index']}.png"
            })

        cap.release()

        # Get knowledge points and related apps (from annotation or task)
        osworld_overlap = ann.get('osworld_overlap') or task_info.get('osworld_overlap', [])
        custom_nodes = ann.get('custom_nodes') or task_info.get('custom_nodes', [])
        related_apps = ann.get('related_apps') or task_info.get('related_apps', [])

        # Build output with new format
        cua_data = {
            'task_id': int(task_id) if task_id.isdigit() else task_id,
            'instruction': task_info.get('instruction', ''),
            'step_by_step_instructions': ann.get('step_by_step_instructions', ''),
            'knowledge_points': {
                'osworld_overlap': osworld_overlap,
                'custom_nodes': custom_nodes
            },
            'related_apps': related_apps,
            'traj': traj
        }

        with open(task_output_dir / f'task_{task_id}.json', 'w') as f:
            json.dump(cua_data, f, indent=2, ensure_ascii=False)

        all_data.append(cua_data)

    # Save combined file
    with open(OUTPUT_DIR / 'all_tasks.json', 'w') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    # Create ZIP
    zip_path = OUTPUT_DIR / 'export.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                if file == 'export.zip':
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, OUTPUT_DIR)
                zf.write(file_path, arcname)

    return jsonify({
        'success': True,
        'message': f'Exported {len(passed_tasks)} passed cases',
        'count': len(passed_tasks),
        'output_dir': str(OUTPUT_DIR.resolve()),
        'zip_path': str(zip_path.resolve()),
        'json_path': str((OUTPUT_DIR / 'all_tasks.json').resolve())
    })

@app.route('/api/download')
def api_download():
    zip_path = OUTPUT_DIR / 'export.zip'
    if zip_path.exists():
        return send_from_directory(OUTPUT_DIR, 'export.zip', as_attachment=True)
    return 'Export not found', 404

# ============================================================================
# Dashboard HTML Template
# ============================================================================

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>CUA Dashboard - OSS Recordings</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 16px 24px;
            border-bottom: 2px solid #00d9ff;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }
        .header h1 { font-size: 1.4em; color: #00d9ff; }
        .header-controls {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }
        .header-controls input {
            padding: 8px 12px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #0f0f1a;
            color: #e0e0e0;
            font-size: 0.9em;
            width: 250px;
        }
        .header-controls button {
            padding: 8px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
        }
        .btn-load { background: #00d9ff; color: #000; }
        .btn-load:hover { background: #33e5ff; }
        .btn-load:disabled { background: #555; color: #888; cursor: not-allowed; }
        .auto-poll-label {
            display: flex;
            align-items: center;
            gap: 6px;
            color: #888;
            font-size: 0.85em;
        }
        .nav-link {
            color: #00d9ff;
            text-decoration: none;
            font-size: 0.9em;
        }
        .nav-link:hover { text-decoration: underline; }
        .status-bar {
            padding: 10px 24px;
            background: #16213e;
            border-bottom: 1px solid #333;
            display: flex;
            gap: 20px;
            align-items: center;
            font-size: 0.85em;
        }
        .status-bar .stat { color: #888; }
        .status-bar .stat b { color: #00d9ff; }
        .container { padding: 20px 24px; }
        .loading {
            text-align: center;
            padding: 40px;
            color: #888;
            font-size: 1.1em;
        }
        .annotator-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }
        .annotator-card {
            background: #1a1a2e;
            border-radius: 8px;
            border: 1px solid #333;
            overflow: hidden;
            transition: border-color 0.2s;
        }
        .annotator-card:hover { border-color: #00d9ff; }
        .annotator-header {
            padding: 14px 16px;
            background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .annotator-header h3 {
            color: #00d9ff;
            font-size: 1em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .annotator-stats {
            display: flex;
            gap: 12px;
            font-size: 0.8em;
        }
        .stat-badge {
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .stat-badge.total { background: #333; color: #ccc; }
        .stat-badge.reviewed { background: #2e7d32; color: #fff; }
        .stat-badge.rejected { background: #c62828; color: #fff; }
        .stat-badge.unreviewed { background: #555; color: #fff; }
        .task-list {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        .task-list.expanded { max-height: 2000px; }
        .task-entry {
            padding: 10px 16px;
            border-top: 1px solid #252542;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85em;
        }
        .task-entry:hover { background: #252542; }
        .task-entry .task-info {
            display: flex;
            flex-direction: column;
            gap: 2px;
            flex: 1;
            min-width: 0;
        }
        .task-entry .task-id { color: #00d9ff; font-weight: 600; }
        .task-entry .task-query {
            color: #888;
            font-size: 0.85em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .task-entry .task-date { color: #666; font-size: 0.8em; }
        .task-entry .task-actions { display: flex; gap: 8px; align-items: center; }
        .review-badge {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .review-badge.reviewed { background: #2e7d32; color: #fff; }
        .review-badge.rejected { background: #c62828; color: #fff; }
        .review-badge.unreviewed { background: #444; color: #999; }
        .btn-view {
            padding: 4px 12px;
            border: 1px solid #00d9ff;
            border-radius: 4px;
            background: transparent;
            color: #00d9ff;
            cursor: pointer;
            font-size: 0.85em;
            text-decoration: none;
        }
        .btn-view:hover { background: #00d9ff; color: #000; }
        .expand-icon {
            transition: transform 0.3s;
            font-size: 0.8em;
        }
        .expand-icon.expanded { transform: rotate(90deg); }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        .empty-state h2 { color: #444; margin-bottom: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>OSS Recording Dashboard</h1>
        <div class="header-controls">
            <a href="/" class="nav-link">&#8592; Local Review</a>
            <input type="text" id="ossFolder" value="recordings_new" placeholder="OSS upload folder" />
            <button class="btn-load" id="loadBtn" onclick="loadDashboard()">Load</button>
            <label class="auto-poll-label">
                <input type="checkbox" id="autoPoll" /> Auto-refresh (60s)
            </label>
        </div>
    </div>
    <div class="status-bar" id="statusBar" style="display:none;">
        <span class="stat">Total recordings: <b id="totalCount">0</b></span>
        <span class="stat">Annotators: <b id="annotatorCount">0</b></span>
        <span class="stat">Reviewed: <b id="reviewedCount">0</b></span>
        <span class="stat">Rejected: <b id="rejectedCount">0</b></span>
        <span class="stat">Unreviewed: <b id="unreviewedCount">0</b></span>
        <span class="stat" style="margin-left:auto;">Last updated: <b id="lastUpdated">-</b></span>
    </div>
    <div class="container" id="content">
        <div class="empty-state">
            <h2>No data loaded</h2>
            <p>Enter an OSS upload folder name above and click Load.</p>
        </div>
    </div>

    <script>
        let pollInterval = null;
        let dashboardData = null;

        async function loadDashboard() {
            const folder = document.getElementById('ossFolder').value.trim();
            if (!folder) return;

            const loadBtn = document.getElementById('loadBtn');
            loadBtn.disabled = true;
            loadBtn.textContent = 'Loading...';

            try {
                const resp = await fetch('/api/oss/dashboard_data?folder=' + encodeURIComponent(folder));
                const data = await resp.json();

                if (data.error) {
                    document.getElementById('content').innerHTML =
                        '<div class="empty-state"><h2>Error</h2><p>' + data.error + '</p></div>';
                    return;
                }

                dashboardData = data;
                renderDashboard(data);

                document.getElementById('statusBar').style.display = 'flex';
                document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();

            } catch (err) {
                document.getElementById('content').innerHTML =
                    '<div class="empty-state"><h2>Error</h2><p>' + err.message + '</p></div>';
            } finally {
                loadBtn.disabled = false;
                loadBtn.textContent = 'Load';
            }
        }

        function renderDashboard(data) {
            const annotators = data.annotators || {};
            const keys = Object.keys(annotators).sort();

            let totalAll = 0, reviewedAll = 0, rejectedAll = 0, unreviewedAll = 0;
            keys.forEach(k => {
                const a = annotators[k];
                totalAll += a.total;
                reviewedAll += a.reviewed;
                rejectedAll += a.rejected;
                unreviewedAll += a.unreviewed;
            });

            document.getElementById('totalCount').textContent = totalAll;
            document.getElementById('annotatorCount').textContent = keys.length;
            document.getElementById('reviewedCount').textContent = reviewedAll;
            document.getElementById('rejectedCount').textContent = rejectedAll;
            document.getElementById('unreviewedCount').textContent = unreviewedAll;

            if (keys.length === 0) {
                document.getElementById('content').innerHTML =
                    '<div class="empty-state"><h2>No recordings found</h2><p>Check the OSS folder name.</p></div>';
                return;
            }

            let html = '<div class="annotator-grid">';
            keys.forEach(username => {
                const a = annotators[username];
                html += renderAnnotatorCard(username, a);
            });
            html += '</div>';

            document.getElementById('content').innerHTML = html;
        }

        function renderAnnotatorCard(username, data) {
            const folder = document.getElementById('ossFolder').value.trim();
            let tasksHtml = '';
            (data.recordings || []).forEach(rec => {
                const statusClass = rec.review_status || 'unreviewed';
                const statusLabel = statusClass.charAt(0).toUpperCase() + statusClass.slice(1);
                const query = rec.query || rec.task_id || '-';
                const date = rec.upload_timestamp ? rec.upload_timestamp.split('T')[0] : '-';
                tasksHtml += `
                    <div class="task-entry">
                        <div class="task-info">
                            <span class="task-id">${rec.task_id || rec.folder_name}</span>
                            <span class="task-query" title="${query}">${query}</span>
                            <span class="task-date">${date}</span>
                        </div>
                        <div class="task-actions">
                            <span class="review-badge ${statusClass}">${statusLabel}</span>
                            <a class="btn-view" href="/oss_review/${encodeURIComponent(rec.folder_name)}?folder=${encodeURIComponent(folder)}">View</a>
                        </div>
                    </div>`;
            });

            return `
                <div class="annotator-card">
                    <div class="annotator-header" onclick="toggleTaskList('${username}')">
                        <h3>
                            <span class="expand-icon" id="icon-${username}">&#9654;</span>
                            ${username}
                        </h3>
                        <div class="annotator-stats">
                            <span class="stat-badge total">${data.total}</span>
                            <span class="stat-badge reviewed">${data.reviewed}</span>
                            <span class="stat-badge rejected">${data.rejected}</span>
                            <span class="stat-badge unreviewed">${data.unreviewed}</span>
                        </div>
                    </div>
                    <div class="task-list" id="tasks-${username}">
                        ${tasksHtml}
                    </div>
                </div>`;
        }

        function toggleTaskList(username) {
            const el = document.getElementById('tasks-' + username);
            const icon = document.getElementById('icon-' + username);
            el.classList.toggle('expanded');
            icon.classList.toggle('expanded');
        }

        // Auto-poll
        document.getElementById('autoPoll').addEventListener('change', function() {
            if (this.checked) {
                pollInterval = setInterval(loadDashboard, 60000);
            } else {
                if (pollInterval) clearInterval(pollInterval);
                pollInterval = null;
            }
        });

        // Load on Enter key
        document.getElementById('ossFolder').addEventListener('keydown', function(e) {
            if (e.key === 'Enter') loadDashboard();
        });
    </script>
</body>
</html>
'''

# ============================================================================
# OSS Review HTML Template
# ============================================================================

OSS_REVIEW_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>OSS Review - {{ folder_name }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #00d9ff;
            flex-wrap: wrap;
            gap: 8px;
        }
        .header h1 { font-size: 1.2em; color: #00d9ff; }
        .header-info { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
        .header-info span { color: #888; font-size: 0.85em; }
        .header-info b { color: #e0e0e0; }
        .btn-back {
            padding: 6px 16px;
            border: 1px solid #00d9ff;
            border-radius: 4px;
            background: transparent;
            color: #00d9ff;
            cursor: pointer;
            text-decoration: none;
            font-size: 0.9em;
        }
        .btn-back:hover { background: #00d9ff; color: #000; }
        .main-content {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        .step-sidebar {
            width: 220px;
            background: #16213e;
            overflow-y: auto;
            padding: 10px;
            border-right: 1px solid #333;
        }
        .step-item {
            padding: 8px 10px;
            margin: 3px 0;
            background: #1a1a2e;
            border-radius: 6px;
            cursor: pointer;
            border-left: 4px solid transparent;
            transition: all 0.2s;
            font-size: 0.85em;
        }
        .step-item:hover { background: #252542; }
        .step-item.active { border-left-color: #00d9ff; background: #252542; }
        .step-item .step-num { color: #00d9ff; font-weight: bold; }
        .step-item .step-action { color: #888; font-size: 0.85em; }
        .content-area {
            flex: 1;
            padding: 15px 20px;
            overflow-y: auto;
        }
        .annotator-info-bar {
            background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 12px;
            border-left: 4px solid #ffc107;
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            font-size: 0.9em;
        }
        .annotator-info-bar .info-item span { color: #888; }
        .annotator-info-bar .info-item b { color: #ffc107; }
        .screenshot-container {
            position: relative;
            display: inline-block;
            margin-bottom: 12px;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            max-width: 100%;
        }
        .screenshot-container img {
            max-width: 100%;
            max-height: 65vh;
            display: block;
        }
        .coord-marker {
            position: absolute;
            width: 24px;
            height: 24px;
            border: 3px solid #ff3333;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            box-shadow: 0 0 10px rgba(255, 51, 51, 0.5);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 10px rgba(255, 51, 51, 0.5); }
            50% { box-shadow: 0 0 20px rgba(255, 51, 51, 0.8); }
        }
        .step-details {
            background: #1a1a2e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 12px;
            border: 1px solid #333;
        }
        .step-details h4 { color: #00d9ff; margin-bottom: 8px; font-size: 0.95em; }
        .step-details .code {
            background: #0d1117;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #4caf50;
            margin-bottom: 8px;
        }
        .step-details .description { color: #ccc; line-height: 1.5; }
        .step-details .justification {
            color: #888;
            font-style: italic;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid #333;
        }
        .review-panel {
            background: #1a1a2e;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #333;
            margin-bottom: 12px;
        }
        .review-panel h4 { color: #ffc107; margin-bottom: 12px; }
        .review-buttons {
            display: flex;
            gap: 10px;
        }
        .review-btn {
            padding: 10px 30px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
            font-size: 1em;
        }
        .review-btn.reviewed { background: #2e7d32; color: #fff; }
        .review-btn.reviewed:hover { background: #4caf50; }
        .review-btn.reviewed.active { background: #4caf50; box-shadow: 0 0 10px #4caf50; }
        .review-btn.rejected { background: #c62828; color: #fff; }
        .review-btn.rejected:hover { background: #f44336; }
        .review-btn.rejected.active { background: #f44336; box-shadow: 0 0 10px #f44336; }
        .review-btn.unreviewed { background: #555; color: #fff; }
        .review-btn.unreviewed:hover { background: #777; }
        .review-btn.unreviewed.active { background: #777; box-shadow: 0 0 10px #777; }
        .nav-buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 12px;
        }
        .nav-btn {
            padding: 8px 20px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #1a1a2e;
            color: #e0e0e0;
            cursor: pointer;
            transition: all 0.2s;
        }
        .nav-btn:hover { border-color: #00d9ff; color: #00d9ff; }
        .nav-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .step-counter {
            color: #888;
            font-size: 0.9em;
            display: flex;
            align-items: center;
        }
        .loading-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(15, 15, 26, 0.9);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            font-size: 1.2em;
            color: #00d9ff;
        }
    </style>
</head>
<body>
    <div class="loading-overlay" id="loadingOverlay">Loading recording data...</div>

    <div class="header">
        <h1>OSS Review</h1>
        <div class="header-info">
            <span>Folder: <b id="folderName"></b></span>
            <span>Task: <b id="taskName"></b></span>
            <span>Status: <b id="reviewStatus">-</b></span>
        </div>
        <a class="btn-back" id="backLink" href="/dashboard">&#8592; Dashboard</a>
    </div>

    <div class="main-content">
        <div class="step-sidebar" id="stepSidebar"></div>
        <div class="content-area" id="contentArea">
            <div class="annotator-info-bar" id="annotatorInfo"></div>
            <div class="nav-buttons">
                <button class="nav-btn" id="prevBtn" onclick="prevStep()">&#9664; Prev</button>
                <span class="step-counter" id="stepCounter">Step 0 / 0</span>
                <button class="nav-btn" id="nextBtn" onclick="nextStep()">Next &#9654;</button>
            </div>
            <div class="screenshot-container" id="screenshotContainer">
                <img id="screenshot" src="" alt="Screenshot" />
                <div class="coord-marker" id="coordMarker" style="display:none;"></div>
            </div>
            <div class="step-details" id="stepDetails"></div>
            <div class="review-panel">
                <h4>Review Decision</h4>
                <div class="review-buttons">
                    <button class="review-btn reviewed" id="btnReviewed" onclick="setReview('reviewed')">Reviewed (Pass)</button>
                    <button class="review-btn rejected" id="btnRejected" onclick="setReview('rejected')">Rejected</button>
                    <button class="review-btn unreviewed" id="btnUnreviewed" onclick="setReview('unreviewed')">Clear</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const folderName = '{{ folder_name }}';
        const ossFolder = new URLSearchParams(window.location.search).get('folder') || 'recordings_new';
        let taskData = null;
        let currentStep = 0;
        let reviewStatus = 'unreviewed';

        document.getElementById('backLink').href = '/dashboard';
        document.getElementById('folderName').textContent = folderName;

        async function loadTask() {
            try {
                const resp = await fetch('/api/oss/task/' + encodeURIComponent(folderName) + '?folder=' + encodeURIComponent(ossFolder));
                taskData = await resp.json();

                if (taskData.error) {
                    document.getElementById('contentArea').innerHTML =
                        '<div style="padding:40px;color:#f44336;">' + taskData.error + '</div>';
                    document.getElementById('loadingOverlay').style.display = 'none';
                    return;
                }

                // Set task name
                document.getElementById('taskName').textContent = taskData.task_name || folderName;

                // Render annotator info
                const info = taskData.annotator_info || {};
                let infoHtml = '';
                if (info.username) infoHtml += '<div class="info-item"><span>Annotator:</span> <b>' + info.username + '</b></div>';
                if (info.task_id) infoHtml += '<div class="info-item"><span>Task ID:</span> <b>' + info.task_id + '</b></div>';
                if (info.query) infoHtml += '<div class="info-item"><span>Query:</span> <b>' + info.query + '</b></div>';
                if (info.upload_timestamp) infoHtml += '<div class="info-item"><span>Uploaded:</span> <b>' + info.upload_timestamp + '</b></div>';
                document.getElementById('annotatorInfo').innerHTML = infoHtml || '<div class="info-item"><span>No annotator info available</span></div>';

                // Set review status
                reviewStatus = taskData.review_status || 'unreviewed';
                updateReviewUI();

                // Render step sidebar
                renderStepSidebar();
                renderStep(0);

                document.getElementById('loadingOverlay').style.display = 'none';
            } catch (err) {
                document.getElementById('contentArea').innerHTML =
                    '<div style="padding:40px;color:#f44336;">Failed to load: ' + err.message + '</div>';
                document.getElementById('loadingOverlay').style.display = 'none';
            }
        }

        function renderStepSidebar() {
            const steps = taskData.steps || [];
            let html = '';
            steps.forEach((step, i) => {
                html += '<div class="step-item' + (i === 0 ? ' active' : '') + '" onclick="selectStep(' + i + ')" id="step-' + i + '">' +
                    '<span class="step-num">Step ' + i + '</span> ' +
                    '<span class="step-action">' + (step.action || '') + '</span>' +
                    '</div>';
            });
            document.getElementById('stepSidebar').innerHTML = html;
        }

        function selectStep(idx) {
            currentStep = idx;
            renderStep(idx);
            // Update sidebar active state
            document.querySelectorAll('.step-item').forEach((el, i) => {
                el.classList.toggle('active', i === idx);
            });
        }

        function renderStep(idx) {
            const steps = taskData.steps || [];
            if (idx < 0 || idx >= steps.length) return;

            const step = steps[idx];
            currentStep = idx;

            // Update counter
            document.getElementById('stepCounter').textContent = 'Step ' + idx + ' / ' + (steps.length - 1);
            document.getElementById('prevBtn').disabled = idx <= 0;
            document.getElementById('nextBtn').disabled = idx >= steps.length - 1;

            // Load screenshot
            const img = document.getElementById('screenshot');
            img.src = '/oss_frame/' + encodeURIComponent(folderName) + '/' + step.video_time + '?folder=' + encodeURIComponent(ossFolder);

            // Update coordinate marker
            const marker = document.getElementById('coordMarker');
            if (step.has_coordinate && step.coordinate) {
                img.onload = function() {
                    const scaleX = img.clientWidth / taskData.video_width;
                    const scaleY = img.clientHeight / taskData.video_height;
                    marker.style.left = (step.coordinate.x * scaleX) + 'px';
                    marker.style.top = (step.coordinate.y * scaleY) + 'px';
                    marker.style.display = 'block';
                };
            } else {
                marker.style.display = 'none';
            }

            // Update step details
            let detailsHtml = '<h4>Step ' + idx + ': ' + (step.action || '') + '</h4>';
            detailsHtml += '<div class="code">' + (step.code || '') + '</div>';
            if (step.description) {
                detailsHtml += '<div class="description">' + step.description + '</div>';
            }
            if (step.justification) {
                detailsHtml += '<div class="justification">Justification: ' + step.justification + '</div>';
            }
            document.getElementById('stepDetails').innerHTML = detailsHtml;
        }

        function prevStep() { if (currentStep > 0) selectStep(currentStep - 1); }
        function nextStep() { if (taskData && currentStep < taskData.steps.length - 1) selectStep(currentStep + 1); }

        function updateReviewUI() {
            document.getElementById('reviewStatus').textContent = reviewStatus;
            document.getElementById('reviewStatus').style.color =
                reviewStatus === 'reviewed' ? '#4caf50' :
                reviewStatus === 'rejected' ? '#f44336' : '#888';
            ['reviewed', 'rejected', 'unreviewed'].forEach(s => {
                document.getElementById('btn' + s.charAt(0).toUpperCase() + s.slice(1)).classList.toggle('active', reviewStatus === s);
            });
        }

        async function setReview(status) {
            reviewStatus = status;
            updateReviewUI();
            try {
                await fetch('/api/oss/review', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        folder_name: folderName,
                        oss_folder: ossFolder,
                        status: status
                    })
                });
            } catch (err) {
                console.error('Failed to save review:', err);
            }
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft') prevStep();
            if (e.key === 'ArrowRight') nextStep();
        });

        loadTask();
    </script>
</body>
</html>
'''

# ============================================================================
# OSS Routes
# ============================================================================

@app.route('/dashboard')
def dashboard():
    """Dashboard page for OSS recordings overview."""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/oss/list')
def api_oss_list():
    """List recordings from an OSS folder."""
    folder = request.args.get('folder', 'recordings_new')
    try:
        import oss_client
        recordings = oss_client.list_recordings(folder)
        return jsonify({'recordings': recordings, 'folder': folder})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/oss/dashboard_data')
def api_oss_dashboard_data():
    """Aggregated per-annotator statistics for the dashboard."""
    folder = request.args.get('folder', 'recordings_new')
    try:
        import oss_client

        recordings = oss_client.list_recordings(folder)
        review_statuses = load_review_status()

        annotators = {}

        for rec_name in recordings:
            # Try to get metadata from OSS
            prefix = folder.rstrip('/') + '/' + rec_name
            metadata = oss_client.get_recording_metadata(prefix)

            if metadata is None:
                # Infer metadata from folder name
                metadata = oss_client.parse_folder_name_metadata(rec_name)

            username = metadata.get('username', 'Unknown')
            task_id = metadata.get('task_id', '')
            query = metadata.get('query', '')
            upload_ts = metadata.get('upload_timestamp', '')

            # Get review status for this recording
            review_key = f"{folder}/{rec_name}"
            rec_review_status = review_statuses.get(review_key, 'unreviewed')

            if username not in annotators:
                annotators[username] = {
                    'total': 0,
                    'reviewed': 0,
                    'rejected': 0,
                    'unreviewed': 0,
                    'recordings': []
                }

            annotators[username]['total'] += 1
            annotators[username][rec_review_status] += 1
            annotators[username]['recordings'].append({
                'folder_name': rec_name,
                'task_id': task_id,
                'query': query,
                'upload_timestamp': upload_ts,
                'review_status': rec_review_status,
            })

        return jsonify({'annotators': annotators, 'folder': folder})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/oss/task/<path:folder_name>')
def api_oss_task(folder_name):
    """Load recording data for review from OSS."""
    oss_folder = request.args.get('folder', 'recordings_new')
    try:
        import oss_client

        prefix = oss_folder.rstrip('/') + '/' + folder_name

        # Create local cache directory
        local_dir = OSS_CACHE_DIR / folder_name
        local_dir.mkdir(parents=True, exist_ok=True)

        # Download metadata files
        oss_client.download_recording_metadata_files(prefix, str(local_dir))

        # Download video (lazy - skip if already cached)
        oss_client.download_video(prefix, str(local_dir))

        # Load task data from cached files
        data = load_oss_task_data(local_dir)
        if not data:
            return jsonify({'error': 'Could not load recording data. Missing required files.'}), 404

        # Get review status
        review_statuses = load_review_status()
        review_key = f"{oss_folder}/{folder_name}"
        data['review_status'] = review_statuses.get(review_key, 'unreviewed')
        data['folder_name'] = folder_name

        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/oss_frame/<path:folder_name>/<float:video_time>')
def oss_serve_frame(folder_name, video_time):
    """Extract and serve a frame from a cached OSS recording video."""
    import cv2

    local_dir = OSS_CACHE_DIR / folder_name
    video_file = None
    for f in local_dir.glob('*.mp4'):
        if 'video_clips' not in str(f):
            video_file = f
            break

    if not video_file:
        # Try to download the video first
        oss_folder = request.args.get('folder', 'recordings_new')
        try:
            import oss_client
            prefix = oss_folder.rstrip('/') + '/' + folder_name
            video_path = oss_client.download_video(prefix, str(local_dir))
            if video_path:
                video_file = Path(video_path)
        except Exception:
            pass

    if not video_file or not video_file.exists():
        return 'Video not found', 404

    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_num = int(video_time * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = max(0, min(frame_num, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return 'Frame extraction failed', 500

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/api/oss/review', methods=['POST'])
def api_oss_review():
    """Save review status for a recording."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_new')
    status = data.get('status', 'unreviewed')

    if status not in ('reviewed', 'rejected', 'unreviewed'):
        return jsonify({'error': 'Invalid status'}), 400

    review_statuses = load_review_status()
    review_key = f"{oss_folder}/{folder_name}"
    review_statuses[review_key] = status
    save_review_status(review_statuses)

    return jsonify({'success': True})

@app.route('/oss_review/<path:folder_name>')
def oss_review_page(folder_name):
    """Review page for a single OSS recording."""
    return render_template_string(OSS_REVIEW_TEMPLATE, folder_name=folder_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CUA Annotation Tool')
    parser.add_argument('--data', default='./data', help='Data directory')
    parser.add_argument('--csv', default='./task_assignments.csv', help='CSV file with task assignments')
    parser.add_argument('--output', default='./output', help='Output directory')
    parser.add_argument('--oss-cache', default='./oss_cache', help='OSS cache directory')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--host', default='0.0.0.0', help='Host')
    args = parser.parse_args()

    DATA_DIR = Path(args.data)
    CSV_FILE = Path(args.csv)
    OUTPUT_DIR = Path(args.output)
    ANNOTATIONS_FILE = OUTPUT_DIR / 'annotations.json'
    OSS_CACHE_DIR = Path(args.oss_cache)

    print(f"\n{'='*50}")
    print("CUA Annotation Tool")
    print(f"{'='*50}")
    print(f"Data directory: {DATA_DIR}")
    print(f"CSV file: {CSV_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"OSS cache: {OSS_CACHE_DIR}")
    print(f"\nStarting server at: http://{args.host}:{args.port}")
    print(f"Dashboard: http://{args.host}:{args.port}/dashboard")
    print("Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=False)
