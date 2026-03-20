#!/usr/bin/env python3
"""
CUA Annotation Tool - Human verification system for CUA-SFT data.
"""

import json
import os
import re
import csv
import io
import shutil
import zipfile
import base64
import threading
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, render_template_string, send_from_directory, jsonify, request, Response, stream_with_context, session, redirect, url_for
from functools import wraps

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'cua-annotation-tool-secret-key-2026')
app.permanent_session_lifetime = 86400  # 24 hours

# Authentication config
REVIEWER_ACCOUNTS = {
    'admin': '1qazXSW@3edc',
}


def reviewer_login_required(f):
    """Decorator to require reviewer login for dashboard routes."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('reviewer_logged_in'):
            from urllib.parse import quote
            return redirect('/login?next=' + quote(request.full_path, safe=''))
        return f(*args, **kwargs)
    return decorated


def annotator_login_required(f):
    """Decorator to require annotator login."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('annotator_username'):
            return redirect('/annotator_login')
        return f(*args, **kwargs)
    return decorated


def any_login_required(f):
    """Decorator to require either reviewer or annotator login."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('reviewer_logged_in') or session.get('annotator_username'):
            return f(*args, **kwargs)
        if request.path.startswith('/api/') or request.path.startswith('/oss_frame/'):
            return jsonify({'error': 'Authentication required'}), 401
        return redirect('/login')
    return decorated


def reviewer_api_required(f):
    """Decorator for API routes only reviewers can call."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('reviewer_logged_in'):
            return jsonify({'error': 'Reviewer authentication required'}), 401
        return f(*args, **kwargs)
    return decorated


# Configuration - can be overridden via environment variables
DATA_DIR = Path(os.environ.get('CUA_DATA_DIR', './data'))
CSV_FILE = Path(os.environ.get('CUA_CSV_FILE', './task_assignments.csv'))
OUTPUT_DIR = Path(os.environ.get('CUA_OUTPUT_DIR', './output'))
ANNOTATIONS_FILE = Path(os.environ.get('CUA_ANNOTATIONS_FILE', './annotations.json'))
OSS_CACHE_DIR = Path(os.environ.get('CUA_OSS_CACHE', './oss_cache'))
REVIEW_STATUS_FILE = Path(os.environ.get('CUA_REVIEW_STATUS', './review_status.json'))
OSS_ANNOTATIONS_FILE = Path(os.environ.get('CUA_OSS_ANNOTATIONS', './oss_annotations.json'))
OSS_COORD_ADJUSTMENTS_FILE = Path(os.environ.get('CUA_OSS_COORD_ADJ', './oss_coord_adjustments.json'))

# Server-side dashboard cache: { folder_name: { annotators: {...}, folder: str, _timestamp: float } }
_dashboard_cache = {}

# AI Check configuration
AI_CHECK_API_KEY = os.environ.get('AI_CHECK_API_KEY', '')
AI_CHECK_BASE_URL = os.environ.get('AI_CHECK_BASE_URL', 'https://openai.sufy.com/v1')
AI_CHECK_MODEL = os.environ.get('AI_CHECK_MODEL', 'gemini-3.1-flash-lite-preview')
AI_CHECK_BATCH_SIZE = 1  # Sequential inference for accuracy

# In-memory tracker for running AI checks: { ann_key: {status, progress, total, steps, ...} }
_ai_check_tasks = {}
_ai_check_lock = threading.Lock()

AI_CHECK_SYSTEM_PROMPT = """You are a strict quality assurance AI for CUA (Computer Use Agent) task annotations.
You review individual steps of GUI operations recorded by human annotators performing tasks on an Ubuntu desktop.

For each step you evaluate TWO things:
1. OPERATION CORRECTNESS — Is this step correct, necessary, and well-executed for the given task?
2. JUSTIFICATION QUALITY — Does the written justification meet the strict standards below?

=====================================
JUSTIFICATION QUALITY STANDARDS
=====================================
Each justification MUST answer two core questions:
  (a) "WHY is this operation performed?" — the specific reason/intent
  (b) "What is its NECESSITY for completing the task?" — how it advances the overall goal

Grading rubric:
- "good": Clearly states both WHY and NECESSITY. Specific to this exact step. Written in English.
- "acceptable": States WHY but NECESSITY is only implied, or is slightly vague but still understandable.
- "poor": Only describes WHAT is done (not WHY), or uses vague/generic language, or doesn't match the step.
- "missing": Empty or whitespace-only justification.

STRICT RULES for justification:
1. A justification that merely re-describes the action (e.g., "Click on the menu" for a click action) is POOR — it says WHAT, not WHY.
2. Phrases like "do the task", "continue", "next step", "proceed", "complete the operation" are POOR.
3. Chinese non-executable phrases like "然后完成", "然后处理一下" are POOR.
4. Must match the actual action. If the step is a click but justification describes typing, it is POOR.
5. If the justification describes the step's purpose clearly (e.g., "Open the terminal to execute the memory check command") that is GOOD.
6. ALWAYS provide rewritten_justification when quality is "poor" or "missing". The rewrite should follow the format: "[Specific reason for this action] to [how it advances the task goal]"

EXAMPLES:
- Action: click on "Show Applications" icon. Justification: "Click the menu and find LibreOffice Calc" → POOR (re-describes action, doesn't explain why LibreOffice Calc is needed for the task)
  Better: "Open the application launcher to find and start LibreOffice Calc, which is needed to create the spreadsheet for recording memory usage data."
- Action: type "free -h". Justification: "Type free -h in the command line to check the memory status." → GOOD (explains what + why)
- Action: click. Justification: "Click Save As to save the document." → POOR (just restates the action)
  Better: "Click 'Save As' to save the Writer document as 'system_memory_summary.odt', which is the required output format specified in the task."

=====================================
OPERATION CORRECTNESS STANDARDS
=====================================
- "correct": Step is necessary and properly executed for the task.
- "wrong": Step clicks wrong target, types wrong text, or performs an action that doesn't make sense for the task.
- "redundant": Step does something already accomplished, or is an unnecessary repeat (e.g., clicking same button twice, scrolling when already at the target, back-and-forth navigation).
- "suspicious": Step seems questionable — possibly unnecessary exploratory action, or the intent is unclear from the screenshot.

Look at the SCREENSHOT carefully to verify:
- Is the click target correct? Does the coordinate match a meaningful UI element?
- After a scroll, is the scroll direction and position reasonable?
- For typing, is the text relevant to the task?
- Does this step logically follow from the previous operations summary?
- Is the same action being repeated unnecessarily?

SPECIFIC OPERATION RULES:
- Scroll: Mouse should be in a scrollable area. Scroll amount should be moderate.
- Drag: Must have clear start+end points with no mid-release.
- Click: Target must be a meaningful UI element for the task.
- Type: Text must be relevant to the task goal.
- PROHIBITED: Shaky clicking (repeated clicks at same position), meaningless back-and-forth, redundant operations, clicks on wrong targets.

VISUAL ANNOTATIONS ON SCREENSHOTS:
- The screenshot includes RED visual annotations showing the operation:
  - Click: Red circle at click position. Double-click has double circle + "x2". Right-click has square + "R". Triple-click has triple circle + "x3".
  - Drag: Red line from start to end with filled arrowhead at end, circle at start.
  - Scroll: Red arrow showing scroll direction + text label (e.g., "Scroll Down x3").
- Use these visual annotations to verify the operation target and position.

MISSING JUSTIFICATION HANDLING:
- If justification is marked as "(NOT PROVIDED - justification is missing)", this is NORMAL — annotators fill in justifications during web review, not during recording. Set justification_quality to "missing".
- You MUST write a complete, high-quality justification in English in rewritten_justification (format: "[Specific reason] to [how it advances the task]").
- Focus on explaining what the operation does and why it is needed for the task.
- Missing justification is NOT an error — it just means the annotator hasn't filled it in yet.

=====================================
OUTPUT FORMAT
=====================================
IMPORTANT: The fields correctness_reason, reasoning, and justification_issues MUST be written in Chinese (简体中文). The rewritten_justification should be in English. The operation_summary_update should be in English.

Output ONLY a valid JSON object — no markdown fences, no text before/after:
{"correctness":"correct|wrong|redundant|suspicious","correctness_reason":"用1-2句中文解释为什么这一步是正确/错误/冗余/可疑的","reasoning":"用2-4句中文分析：截图中看到什么，操作做了什么，是否符合任务目标，以及为什么给出这个判定","justification_quality":"good|acceptable|poor|missing","justification_issues":["用中文描述具体问题"],"rewritten_justification":"Improved justification in English following the [reason] to [necessity] format, or null if quality is good/acceptable","flags":[],"operation_summary_update":"Cumulative 2-4 sentence summary in English of ALL operations from step 1 through this step, for context in checking the next step"}

Possible flags (use when applicable): redundant_click, meaningless_scroll, shaky_click, back_and_forth, wrong_target, unnecessary_step, missing_justification, vague_justification"""

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

        # Normalize ambiguous action types based on description content
        if action == 'mouse_press':
            if '⌨️ Press:' in description:
                action = 'press'
            elif '⌨️ Type:' in description:
                action = 'type'
            else:
                action = 'click'  # actual mouse press → treat as click

        code = build_pyautogui_code(action, ce, description)

        # Only show coordinate marker for actions that have meaningful coordinates
        has_coordinate = action in ('click', 'drag', 'scroll') and (x != 0 or y != 0)

        # Parse drag end coordinates
        drag_to = None
        if action == 'drag':
            drag_match = re.search(r'Drag from \((\d+),\s*(\d+)\) to \((\d+),\s*(\d+)\)', description)
            if drag_match:
                drag_to = {'x': int(drag_match.group(3)), 'y': int(drag_match.group(4))}

        # Parse scroll direction
        scroll_info = None
        if action == 'scroll':
            down = re.search(r'⬇️×(\d+)', description)
            up = re.search(r'⬆️×(\d+)', description)
            left = re.search(r'⬅️×(\d+)', description)
            right = re.search(r'➡️×(\d+)', description)
            scroll_info = {'dx': 0, 'dy': 0}
            if down: scroll_info['dy'] = int(down.group(1))
            if up: scroll_info['dy'] = -int(up.group(1))
            if right: scroll_info['dx'] = int(right.group(1))
            if left: scroll_info['dx'] = -int(left.group(1))

        # Detect click subtype from description/code
        click_type = 'click'
        if action == 'click':
            if 'doubleClick' in code or 'double' in description.lower():
                click_type = 'doubleClick'
            elif 'rightClick' in code or 'right' in description.lower():
                click_type = 'rightClick'
            elif 'clicks=3' in code or 'triple' in description.lower():
                click_type = 'tripleClick'

        steps.append({
            'index': i,
            'action': action,
            'description': description,
            'justification': ce.get('justification', ''),
            'code': code,
            'coordinate': {'x': x, 'y': y},
            'has_coordinate': has_coordinate,
            'drag_to': drag_to,
            'scroll_info': scroll_info,
            'click_type': click_type,
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

def load_oss_annotations():
    """Load OSS recording annotations."""
    if OSS_ANNOTATIONS_FILE.exists():
        with open(OSS_ANNOTATIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_oss_annotations(annotations):
    """Save OSS recording annotations."""
    OSS_ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OSS_ANNOTATIONS_FILE, 'w') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

def load_oss_coord_adjustments():
    """Load OSS coordinate adjustments."""
    if OSS_COORD_ADJUSTMENTS_FILE.exists():
        with open(OSS_COORD_ADJUSTMENTS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_oss_coord_adjustments(adjustments):
    """Save OSS coordinate adjustments."""
    OSS_COORD_ADJUSTMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OSS_COORD_ADJUSTMENTS_FILE, 'w') as f:
        json.dump(adjustments, f, indent=2, ensure_ascii=False)

def _build_case_overlay(oss_folder, folder_name):
    """Build the complete overlay dict for a case by merging annotations + coord adjustments.
    This is what gets uploaded to OSS as the single source of truth."""
    ann_key = f"{oss_folder}/{folder_name}"
    oss_annotations = load_oss_annotations()
    ann = oss_annotations.get(ann_key, {})

    # Merge coord adjustments into the overlay
    coord_adjustments = load_oss_coord_adjustments()
    case_coords = {}
    prefix = ann_key + '_'
    for k, v in coord_adjustments.items():
        if k.startswith(prefix):
            step_idx = k[len(prefix):]
            case_coords[step_idx] = v
    if case_coords:
        ann['coord_adjustments'] = case_coords

    # Add timestamp
    ann['_last_updated'] = datetime.utcnow().isoformat() + 'Z'

    return ann


def _sync_overlay_to_oss(oss_folder, folder_name):
    """Build overlay and upload to OSS at {oss_folder}_annotations/{folder_name}/overlay.json.
    Called after each local save to keep OSS in sync. Runs best-effort (non-blocking on failure)."""
    try:
        import oss_client
        overlay = _build_case_overlay(oss_folder, folder_name)
        oss_client.upload_annotation_overlay(oss_folder, folder_name, overlay)
    except Exception:
        pass  # Best-effort; local save already succeeded


def _load_overlay_from_oss(oss_folder, folder_name):
    """Download overlay from OSS and merge into local files.
    Called when loading a case to ensure we have the latest data from OSS."""
    try:
        import oss_client
        overlay = oss_client.download_annotation_overlay(oss_folder, folder_name)
        if overlay is None:
            return  # No overlay on OSS, use local data

        ann_key = f"{oss_folder}/{folder_name}"

        # Extract coord_adjustments from overlay and store separately
        case_coords = overlay.pop('coord_adjustments', {})
        overlay.pop('_last_updated', None)

        # Update local annotations
        oss_annotations = load_oss_annotations()
        local_ann = oss_annotations.get(ann_key, {})
        local_ts = local_ann.get('_last_updated', '')
        oss_ts = overlay.get('_last_updated', '')

        # OSS overlay wins (it's the shared source of truth)
        oss_annotations[ann_key] = overlay
        save_oss_annotations(oss_annotations)

        # Update local coord adjustments
        if case_coords:
            coord_adjustments = load_oss_coord_adjustments()
            for step_idx, adj in case_coords.items():
                coord_adjustments[f"{ann_key}_{step_idx}"] = adj
            save_oss_coord_adjustments(coord_adjustments)

    except Exception:
        pass  # Best-effort; fall back to local data


# ============================================================================
# AI Check Helper Functions
# ============================================================================

def _extract_frame_base64(video_path, video_time, quality=50, max_width=800):
    """Extract a frame from video and return as base64 JPEG string."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_num = int(video_time * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = max(0, min(frame_num, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def _extract_frame_with_annotation(video_path, video_time, step, video_width, video_height, quality=50, max_width=800):
    """Extract a frame and draw the operation annotation on it (like AgentNet-Tool).
    This gives the AI model visual context of what operation is being performed."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return _extract_frame_base64(video_path, video_time, quality, max_width)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_num = int(video_time * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = max(0, min(frame_num, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    fh, fw = frame.shape[:2]
    # Scale coordinates from video_width/video_height to actual frame size
    sx = fw / (video_width or fw)
    sy = fh / (video_height or fh)

    action = step.get('action', '')
    coord = step.get('coordinate', {})
    x = int(coord.get('x', 0) * sx)
    y = int(coord.get('y', 0) * sy)
    color = (0, 0, 255)  # Red in BGR

    if action == 'click':
        click_type = step.get('click_type', 'click')
        # Draw circle at click position
        cv2.circle(frame, (x, y), 15, color, 2)
        if click_type == 'doubleClick':
            # Double circle for double click
            cv2.circle(frame, (x, y), 22, color, 2)
            cv2.putText(frame, 'x2', (x + 18, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif click_type == 'rightClick':
            # Square around circle for right click
            cv2.rectangle(frame, (x - 18, y - 18), (x + 18, y + 18), color, 2)
            cv2.putText(frame, 'R', (x + 18, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif click_type == 'tripleClick':
            cv2.circle(frame, (x, y), 22, color, 2)
            cv2.circle(frame, (x, y), 29, color, 2)
            cv2.putText(frame, 'x3', (x + 18, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    elif action == 'drag':
        drag_to = step.get('drag_to', {})
        if drag_to:
            ex = int(drag_to.get('x', 0) * sx)
            ey = int(drag_to.get('y', 0) * sy)
            # Draw line from start to end
            cv2.line(frame, (x, y), (ex, ey), color, 2)
            # Arrowhead at end
            angle = np.arctan2(ey - y, ex - x)
            a_len = 20
            tip = (ex, ey)
            left_wing = (int(ex - a_len * np.cos(angle - 0.4)), int(ey - a_len * np.sin(angle - 0.4)))
            right_wing = (int(ex - a_len * np.cos(angle + 0.4)), int(ey - a_len * np.sin(angle + 0.4)))
            cv2.fillPoly(frame, [np.array([tip, left_wing, right_wing], dtype=np.int32)], color)
            # Start circle
            cv2.circle(frame, (x, y), 8, color, 2)

    elif action == 'scroll':
        scroll_info = step.get('scroll_info', {})
        dy = scroll_info.get('dy', 0) if scroll_info else 0
        dx = scroll_info.get('dx', 0) if scroll_info else 0
        arrow_len = 60
        # Draw scroll arrow at position
        if dy != 0:
            end_y = y + (arrow_len if dy > 0 else -arrow_len)
            cv2.arrowedLine(frame, (x, y), (x, end_y), color, 2, tipLength=0.3)
            label = f'Scroll {"Down" if dy > 0 else "Up"} x{abs(dy)}'
        elif dx != 0:
            end_x = x + (arrow_len if dx > 0 else -arrow_len)
            cv2.arrowedLine(frame, (x, y), (end_x, y), color, 2, tipLength=0.3)
            label = f'Scroll {"Right" if dx > 0 else "Left"} x{abs(dx)}'
        else:
            cv2.arrowedLine(frame, (x, y), (x, y + arrow_len), color, 2, tipLength=0.3)
            label = 'Scroll'
        cv2.putText(frame, label, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Resize after drawing annotations (better quality annotations)
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def _call_gemini(messages):
    """Call the Gemini API via OpenAI-compatible endpoint. Returns response text."""
    import requests as http_req
    headers = {
        "Authorization": f"Bearer {AI_CHECK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": AI_CHECK_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    resp = http_req.post(
        f"{AI_CHECK_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=180
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _parse_ai_response(text):
    """Parse JSON from AI response, tolerating markdown fences."""
    text = text.strip()
    # Remove markdown code fences
    if text.startswith('```'):
        lines = text.split('\n')
        lines = [l for l in lines if not l.strip().startswith('```')]
        text = '\n'.join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract first JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _check_single_step(query, instructions, running_summary, step_idx, step, img_b64, total_steps):
    """Check a single step using Gemini API with annotated screenshot. Returns (result_dict, updated_summary) or None."""
    action = step.get('action', '')
    code = step.get('code', '')
    description = step.get('description', '')
    justification = step.get('justification', '')
    coord = step.get('coordinate', {})
    has_justification = bool(justification and justification.strip())

    user_text = f"""Task Query: {query}
Task Instructions: {instructions or 'Not provided'}

== Previous Operations Summary ==
{running_summary}

== Current Step {step_idx + 1} of {total_steps} ==
- Action: {action}
- Code: {code}
- Description: {description}
- Justification: "{justification or '(NOT PROVIDED - justification is missing)'}"
- Coordinate: ({coord.get('x', 0)}, {coord.get('y', 0)})

[Screenshot attached below — the operation is visually annotated on the screenshot with red markers: circles for clicks, arrows for scroll direction, lines with arrowheads for drag paths]

"""
    if has_justification:
        user_text += "Evaluate this step's operation correctness and justification quality. Remember: correctness_reason and reasoning must be in Chinese (简体中文). Output JSON only."
    else:
        user_text += """This step has NO justification yet (this is normal — annotators fill it in during web review).
1. Evaluate the operation correctness as usual.
2. Set justification_quality to "missing".
3. WRITE a complete, high-quality justification in English in rewritten_justification. Format: "[Specific reason] to [how it advances the task goal]".
4. Explain in correctness_reason (in Chinese 简体中文) what this operation does and whether it makes sense.
Output JSON only."""

    content = [{"type": "text", "text": user_text}]
    if img_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    messages = [
        {"role": "system", "content": AI_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]

    for attempt in range(3):
        try:
            response_text = _call_gemini(messages)
            result = _parse_ai_response(response_text)
            if result and isinstance(result, dict) and 'correctness' in result:
                summary = result.get('operation_summary_update', running_summary)
                return result, summary
            logger.warning(f"AI check step {step_idx}: parse attempt {attempt+1} failed, response: {response_text[:200]}")
        except Exception as e:
            logger.warning(f"AI check step {step_idx}: API attempt {attempt+1} error: {e}")
    return None


def _run_ai_check_thread(ann_key, oss_folder, folder_name):
    """Background thread that runs AI quality check sequentially — one step per API call.
    Each call includes an annotated screenshot with the operation drawn on it."""
    try:
        local_dir = OSS_CACHE_DIR / folder_name
        task_data = load_oss_task_data(local_dir)
        if not task_data:
            with _ai_check_lock:
                _ai_check_tasks[ann_key] = {'status': 'failed', 'error': 'Could not load task data'}
            return

        # Apply overlay to get current state of steps
        oss_annotations = load_oss_annotations()
        ann = oss_annotations.get(ann_key, {})
        justification_edits = ann.get('justification_edits', {})
        code_edits = ann.get('code_edits', {})
        video_time_edits = ann.get('video_time_edits', {})
        deleted_steps = set(ann.get('deleted_steps', []))
        coord_adjustments = load_oss_coord_adjustments()

        steps = task_data.get('steps', [])
        for step in steps:
            step['original_index'] = step['index']
            si = str(step['index'])
            if si in justification_edits:
                step['justification'] = justification_edits[si]
            if si in video_time_edits:
                step['video_time'] = video_time_edits[si]
            # Apply coord adjustments and regenerate code
            adj_key_s = f"{ann_key}_{step['index']}"
            if adj_key_s in coord_adjustments:
                adj = coord_adjustments[adj_key_s]
                step['coordinate'] = {'x': adj['x'], 'y': adj['y']}
                x, y = adj['x'], adj['y']
                code = step.get('code', '')
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
            # Code edits override everything (after coord regeneration)
            if si in code_edits:
                step['code'] = code_edits[si]
        steps = [s for s in steps if s['original_index'] not in deleted_steps]

        # Get query and instructions
        info = task_data.get('annotator_info', {})
        query = ann.get('query', '') or info.get('query', '')
        instructions = ann.get('step_by_step_instructions', '') or info.get('step_by_step_instruction', '')

        video_width = task_data.get('video_width', 1920)
        video_height = task_data.get('video_height', 1080)
        total_steps = len(steps)

        with _ai_check_lock:
            _ai_check_tasks[ann_key] = {
                'status': 'running',
                'progress': 0,
                'total': total_steps,
                'steps': {},
            }

        # Find video file
        video_file = None
        for f in local_dir.glob('*.mp4'):
            if 'video_clips' not in str(f):
                video_file = f
                break

        running_summary = "This is the first operation. No previous operations yet."

        for i, step in enumerate(steps):
            orig_idx = step.get('original_index', i)

            # Extract frame with operation annotation drawn on it
            img_b64 = None
            if video_file:
                img_b64 = _extract_frame_with_annotation(
                    video_file, step.get('video_time', 0), step,
                    video_width, video_height
                )

            result = _check_single_step(
                query=query,
                instructions=instructions,
                running_summary=running_summary,
                step_idx=i,
                step=step,
                img_b64=img_b64,
                total_steps=total_steps,
            )

            if result:
                step_result, new_summary = result
                running_summary = new_summary
                with _ai_check_lock:
                    _ai_check_tasks[ann_key]['steps'][str(orig_idx)] = step_result
            else:
                with _ai_check_lock:
                    _ai_check_tasks[ann_key]['steps'][str(orig_idx)] = {
                        'correctness': 'unknown',
                        'correctness_reason': 'AI check failed after 3 retries',
                        'justification_quality': 'unknown',
                        'justification_issues': [],
                        'rewritten_justification': None,
                        'flags': [],
                    }

            with _ai_check_lock:
                _ai_check_tasks[ann_key]['progress'] = i + 1

        # Save completed results to annotations
        with _ai_check_lock:
            checked_steps = dict(_ai_check_tasks[ann_key].get('steps', {}))
        final_results = {
            'status': 'completed',
            'checked_at': datetime.utcnow().isoformat() + 'Z',
            'steps': checked_steps,
            'overall_summary': running_summary,
            'total_checked': total_steps,
        }

        oss_annotations = load_oss_annotations()
        if ann_key not in oss_annotations:
            oss_annotations[ann_key] = {}
        oss_annotations[ann_key]['ai_check_results'] = final_results
        save_oss_annotations(oss_annotations)

        _sync_overlay_to_oss(oss_folder, folder_name)

        with _ai_check_lock:
            _ai_check_tasks[ann_key] = final_results

        logger.info(f"AI check completed for {ann_key}: {total_steps} steps sequentially")

    except Exception as e:
        logger.error(f"AI check failed for {ann_key}: {e}")
        with _ai_check_lock:
            _ai_check_tasks[ann_key] = {'status': 'failed', 'error': str(e)}


def _preextract_video_frames(local_dir, steps):
    """Pre-extract all step frames from video in one pass, caching to _frames/.
    Opens the video once, seeks to each timestamp, and saves as JPEG.
    This makes subsequent /oss_frame/ requests instant (serve from disk)."""
    local_dir = Path(local_dir)
    frame_dir = local_dir / '_frames'

    # Check how many frames are missing
    missing = []
    for step in steps:
        vt = step.get('video_time', 0)
        cache_file = frame_dir / f'{float(vt):.1f}.jpg'
        if not cache_file.exists():
            missing.append(vt)

    if not missing:
        return  # All frames already cached

    # Find video file
    video_file = None
    for f in local_dir.glob('*.mp4'):
        if 'video_clips' not in str(f):
            video_file = f
            break
    if not video_file:
        return

    try:
        import cv2
        frame_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for vt in missing:
            frame_num = int(float(vt) * fps)
            frame_num = max(0, min(frame_num, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                cache_file = frame_dir / f'{float(vt):.1f}.jpg'
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                cache_file.write_bytes(buf.tobytes())

        cap.release()
    except Exception:
        pass


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

    # Load events - two formats:
    # Format 1 (old): reduced_events_complete.jsonl + vis.jsonl + full video
    # Format 2 (new): reduced_events_vis.jsonl only + screenshots/ + video_clips/
    complete_file = local_dir / 'reduced_events_complete.jsonl'
    vis_file = local_dir / 'reduced_events_vis.jsonl'

    vis_events = []
    if vis_file.exists():
        with open(vis_file) as f:
            vis_events = [json.loads(line) for line in f if line.strip()]

    complete_events = []
    if complete_file.exists():
        with open(complete_file) as f:
            complete_events = [json.loads(line) for line in f if line.strip()]

    has_complete = len(complete_events) > 0
    if not has_complete and not vis_events:
        return None

    primary_events = complete_events if has_complete else vis_events

    # Get dimensions from metadata if no video
    if not video_path:
        video_width = metadata.get('screen_width', 1920) or 1920
        video_height = metadata.get('screen_height', 1080) or 1080

    steps = []
    for i, ce in enumerate(primary_events):
        ve = vis_events[i] if i < len(vis_events) else {}
        coord = ce.get('coordinate', {})
        x, y = coord.get('x', 0), coord.get('y', 0)

        if has_complete:
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
        else:
            # New format: use event ID for screenshot lookup
            video_time = float(ce.get('id', i))

        action = ce.get('action', '')
        description = ve.get('description', ce.get('description', ''))

        # Normalize ambiguous action types based on description content
        if action == 'mouse_press':
            if '⌨️ Press:' in description:
                action = 'press'
            elif '⌨️ Type:' in description:
                action = 'type'
            else:
                action = 'click'

        code = build_pyautogui_code(action, ce, description)
        has_coordinate = action in ('click', 'drag', 'scroll') and (x != 0 or y != 0)

        # Parse drag end coordinates
        drag_to = None
        if action == 'drag':
            match = re.search(r'Drag from \((\d+),\s*(\d+)\) to \((\d+),\s*(\d+)\)', description)
            if match:
                drag_to = {'x': int(match.group(3)), 'y': int(match.group(4))}

        # Parse scroll direction
        scroll_info = None
        if action == 'scroll':
            s_down = re.search(r'⬇️×(\d+)', description)
            s_up = re.search(r'⬆️×(\d+)', description)
            s_left = re.search(r'⬅️×(\d+)', description)
            s_right = re.search(r'➡️×(\d+)', description)
            scroll_info = {'dx': 0, 'dy': 0}
            if s_down: scroll_info['dy'] = int(s_down.group(1))
            if s_up: scroll_info['dy'] = -int(s_up.group(1))
            if s_right: scroll_info['dx'] = int(s_right.group(1))
            if s_left: scroll_info['dx'] = -int(s_left.group(1))

        # Detect click subtype
        click_type = 'click'
        if action == 'click':
            if 'doubleClick' in code or 'double' in description.lower():
                click_type = 'doubleClick'
            elif 'rightClick' in code or 'right' in description.lower():
                click_type = 'rightClick'
            elif 'clicks=3' in code or 'triple' in description.lower():
                click_type = 'tripleClick'

        steps.append({
            'index': i,
            'action': action,
            'description': description,
            'justification': ce.get('justification', ''),
            'code': code,
            'coordinate': {'x': x, 'y': y},
            'has_coordinate': has_coordinate,
            'drag_to': drag_to,
            'scroll_info': scroll_info,
            'click_type': click_type,
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

    # Load knowledge points
    knowledge_points = []
    kp_file = local_dir / 'knowledge_points.json'
    if kp_file.exists():
        with open(kp_file) as f:
            kp_data = json.load(f)
            if isinstance(kp_data, list):
                knowledge_points = kp_data

    return {
        'video_file': video_file,
        'video_start_ts': video_start_ts,
        'video_width': video_width,
        'video_height': video_height,
        'steps': steps,
        'annotator_info': annotator_info,
        'task_name': task_name,
        'knowledge_points': knowledge_points,
        'screenshot_mode': 'screenshots' if not has_complete else 'video',
    }


@app.route('/')
def index():
    return render_template('local_review.html')

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


# ============================================================================
# OSS Review HTML Template
# ============================================================================


# ============================================================================
# Authentication Routes
# ============================================================================




@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username in REVIEWER_ACCOUNTS and REVIEWER_ACCOUNTS[username] == password:
            session.permanent = True
            session['reviewer_logged_in'] = True
            session['reviewer_username'] = username
            next_url = request.args.get('next', '/dashboard')
            return redirect(next_url)
        error = 'Invalid username or password'
    return render_template('login.html', error=error)


@app.route('/annotator_login', methods=['GET', 'POST'])
def annotator_login():
    error = None
    folder = request.form.get('folder', '') or request.args.get('folder', '')
    username = request.form.get('username', '') or request.args.get('user', '')
    if request.method == 'POST':
        password = request.form.get('password', '')
        if not folder or not username:
            error = 'All fields required'
        elif password.lower() != username.lower():
            error = 'Invalid password'
        else:
            session.permanent = True
            session['annotator_username'] = username
            session['annotator_folder'] = folder
            return redirect('/annotator_dashboard?folder=' + folder + '&user=' + username)
    return render_template('annotator_login.html', error=error, folder=folder, username=username)


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# ============================================================================
# OSS Routes
# ============================================================================

@app.route('/dashboard')
@reviewer_login_required
def dashboard():
    """Dashboard page for OSS recordings overview."""
    return render_template('dashboard.html')

@app.route('/api/oss/list')
@reviewer_login_required
def api_oss_list():
    """List recordings from an OSS folder."""
    folder = request.args.get('folder', 'recordings_0303')
    try:
        import oss_client
        recordings = oss_client.list_recordings(folder)
        return jsonify({'recordings': recordings, 'folder': folder})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _fetch_dashboard_data(folder):
    """Fetch dashboard data from OSS for a folder. Uses cache for metadata."""
    import oss_client
    import time

    recordings = oss_client.list_recordings(folder)

    # Check if we have cached metadata for this folder
    cached = _dashboard_cache.get(folder)
    cached_recs = {}
    if cached:
        # Build lookup of cached recording metadata by folder_name
        for ann_data in cached.get('annotators', {}).values():
            for rec in ann_data.get('recordings', []):
                cached_recs[rec['folder_name']] = rec

    review_statuses = load_review_status()
    annotators = {}

    for rec_name in recordings:
        review_key = f"{folder}/{rec_name}"
        rec_review_status = review_statuses.get(review_key, 'unreviewed')

        # Use cached metadata if available, otherwise fetch from OSS
        if rec_name in cached_recs:
            meta = cached_recs[rec_name]
            username = meta.get('_username', meta.get('username', 'Unknown'))
            task_id = meta.get('_task_id', meta.get('task_id', ''))
            query = meta.get('_query', meta.get('query', ''))
            upload_ts = meta.get('_upload_ts', meta.get('upload_timestamp', ''))
        else:
            prefix = folder.rstrip('/') + '/' + rec_name
            metadata = oss_client.get_recording_metadata(prefix)
            if metadata is None:
                metadata = oss_client.parse_folder_name_metadata(rec_name)
            username = metadata.get('username', 'Unknown')
            task_id = metadata.get('task_id', '')
            query = metadata.get('query', '')
            upload_ts = metadata.get('upload_timestamp', '')

        # Also fetch task_name from task_name.json (the visible identifier)
        task_name = ''
        if rec_name in cached_recs:
            task_name = cached_recs[rec_name].get('_task_name', '')
        if not task_name:
            try:
                tn_key = folder.rstrip('/') + '/' + rec_name + '/task_name.json'
                bucket = oss_client._get_bucket()
                tn_result = bucket.get_object(tn_key)
                tn_data = json.loads(tn_result.read().decode('utf-8'))
                task_name = tn_data.get('task_name', '')
            except Exception:
                task_name = ''

        if username not in annotators:
            annotators[username] = {
                'total': 0, 'reviewed': 0, 'rejected': 0, 'unreviewed': 0,
                'recordings': []
            }

        annotators[username]['total'] += 1
        annotators[username][rec_review_status] += 1
        annotators[username]['recordings'].append({
            'folder_name': rec_name,
            'task_id': task_id,
            'task_name': task_name,
            'query': query,
            'upload_timestamp': upload_ts,
            'review_status': rec_review_status,
            '_username': username,
            '_task_id': task_id,
            '_task_name': task_name,
            '_query': query,
            '_upload_ts': upload_ts,
        })

    # Enrich with error/correction stats from annotations
    oss_annotations_all = load_oss_annotations()
    for username, ann_data in annotators.items():
        error_total = 0
        corrected_total = 0
        for rec in ann_data.get('recordings', []):
            rec_ann_key = f"{folder}/{rec['folder_name']}"
            rec_ann = oss_annotations_all.get(rec_ann_key, {})
            step_errors = rec_ann.get('step_errors', {})
            just_edits = rec_ann.get('justification_edits', {})
            code_edits_r = rec_ann.get('code_edits', {})
            error_total += len(step_errors)
            for si in step_errors:
                if si in just_edits or si in code_edits_r:
                    corrected_total += 1
        ann_data['error_steps'] = error_total
        ann_data['corrected_steps'] = corrected_total

    result = {'annotators': annotators, 'folder': folder, '_timestamp': time.time()}
    _dashboard_cache[folder] = result
    return result


@app.route('/api/oss/dashboard_data')
@reviewer_login_required
def api_oss_dashboard_data():
    """Aggregated per-annotator statistics for the dashboard. Uses server-side cache."""
    folder = request.args.get('folder', 'recordings_0303')
    refresh = request.args.get('refresh', '0') == '1'
    try:
        import time
        cached = _dashboard_cache.get(folder)
        # Use cache if <300s old and not forced refresh — but always refresh review statuses
        if cached and not refresh and (time.time() - cached.get('_timestamp', 0)) < 300:
            # Update review statuses from disk (cheap operation)
            review_statuses = load_review_status()
            for ann_data in cached.get('annotators', {}).values():
                ann_data['reviewed'] = 0
                ann_data['rejected'] = 0
                ann_data['unreviewed'] = 0
                for rec in ann_data.get('recordings', []):
                    review_key = f"{folder}/{rec['folder_name']}"
                    rec['review_status'] = review_statuses.get(review_key, 'unreviewed')
                    ann_data[rec['review_status']] += 1
                ann_data['total'] = len(ann_data['recordings'])
            return jsonify({'annotators': cached['annotators'], 'folder': folder, 'cached': True})

        result = _fetch_dashboard_data(folder)
        return jsonify({'annotators': result['annotators'], 'folder': folder, 'cached': False})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/oss/folder_recordings')
@any_login_required
def api_oss_folder_recordings():
    """Return flat list of all recordings in a folder with review status. For the review-page sidebar."""
    folder = request.args.get('folder', 'recordings_0303')
    try:
        cached = _dashboard_cache.get(folder)
        if not cached:
            cached = _fetch_dashboard_data(folder)

        review_statuses = load_review_status()
        recordings = []
        for ann_data in cached.get('annotators', {}).values():
            for rec in ann_data.get('recordings', []):
                review_key = f"{folder}/{rec['folder_name']}"
                recordings.append({
                    'folder_name': rec['folder_name'],
                    'task_id': rec.get('task_id', ''),
                    'task_name': rec.get('task_name', ''),
                    'query': rec.get('query', ''),
                    'username': rec.get('_username', ''),
                    'review_status': review_statuses.get(review_key, 'unreviewed'),
                })
        return jsonify({'recordings': recordings, 'folder': folder})
    except Exception as e:
        return jsonify({'error': str(e), 'recordings': []})


@app.route('/api/oss/task/<path:folder_name>')
@any_login_required
def api_oss_task(folder_name):
    """Load recording data for review from OSS.

    Architecture:
      - oss_cache/ files are a READ-ONLY mirror of {oss_folder}/{folder_name}/ on OSS
      - User modifications are stored as overlay in {oss_folder}_annotations/{folder_name}/overlay.json on OSS
      - Local files (oss_annotations.json, oss_coord_adjustments.json) are a write-through cache
      - On load: download overlay from OSS -> update local cache -> apply on top of original data
      - On save: write local -> upload overlay to OSS
      - Original data on OSS is NEVER modified.
    """
    oss_folder = request.args.get('folder', 'recordings_0303')
    try:
        import oss_client

        prefix = oss_folder.rstrip('/') + '/' + folder_name

        # Step 1: Download overlay from OSS -> merge into local files
        _load_overlay_from_oss(oss_folder, folder_name)

        # Step 2: Create local cache directory (read-only mirror of OSS)
        local_dir = OSS_CACHE_DIR / folder_name
        local_dir.mkdir(parents=True, exist_ok=True)

        # Step 3: Download metadata files (skips if already cached)
        oss_client.download_recording_metadata_files(prefix, str(local_dir))

        # Download video only if this is a video-format recording
        # (has reduced_events_complete.jsonl). Screenshot-format recordings
        # have no full video — skip the slow OSS scan.
        if (local_dir / 'reduced_events_complete.jsonl').exists():
            oss_client.download_video(prefix, str(local_dir))

        # Load ORIGINAL task data from cached files
        data = load_oss_task_data(local_dir)
        if not data:
            return jsonify({'error': 'Could not load recording data. Missing required files.'}), 404

        # Pre-extract ALL frames in one pass (avoids re-opening video per frame)
        if data.get('screenshot_mode') == 'video':
            _preextract_video_frames(local_dir, data.get('steps', []))

        # Get review status
        review_statuses = load_review_status()
        review_key = f"{oss_folder}/{folder_name}"
        data['review_status'] = review_statuses.get(review_key, 'unreviewed')
        data['folder_name'] = folder_name

        # === Load overlay data ===
        oss_annotations = load_oss_annotations()
        ann = oss_annotations.get(review_key, {})
        data['annotation'] = ann

        # === Apply overlay: justification edits, video_time edits, original_index ===
        # Each step keeps its original_index so overlays reference the correct step
        justification_edits = ann.get('justification_edits', {})
        code_edits = ann.get('code_edits', {})
        video_time_edits = ann.get('video_time_edits', {})
        for step in data.get('steps', []):
            step['original_index'] = step['index']  # preserve original index
            si = str(step['index'])
            if si in justification_edits:
                step['justification'] = justification_edits[si]
            if si in video_time_edits:
                step['video_time'] = video_time_edits[si]

        # === Apply overlay: query edit ===
        if 'query' in ann and ann['query']:
            if 'annotator_info' in data:
                data['annotator_info']['query'] = ann['query']

        # === Apply overlay: coordinate adjustments (regenerates code) ===
        coord_adjustments = load_oss_coord_adjustments()
        data['coord_adjustments'] = {}
        for step in data.get('steps', []):
            adj_key = f"{review_key}_{step['original_index']}"
            if adj_key in coord_adjustments:
                adj = coord_adjustments[adj_key]
                data['coord_adjustments'][str(step['original_index'])] = adj
                # Apply adjustment to step data
                step['coordinate'] = {'x': adj['x'], 'y': adj['y']}
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
                    drag_adj = adj.get('drag_to', {})
                    if drag_adj:
                        step['code'] = f"pyautogui.moveTo({x}, {y}); pyautogui.dragTo({drag_adj['x']}, {drag_adj['y']})"
                        step['drag_to'] = {'x': drag_adj['x'], 'y': drag_adj['y']}
                    else:
                        match = re.search(r'dragTo\((\d+),\s*(\d+)\)', code)
                        if match:
                            step['code'] = f"pyautogui.moveTo({x}, {y}); pyautogui.dragTo({match.group(1)}, {match.group(2)})"
                # Also store drag_to adjustment for frontend display
                if 'drag_to' in adj:
                    data['coord_adjustments'][str(step['original_index'])]['drag_to'] = adj['drag_to']
                    data['coord_adjustments'][str(step['original_index'])]['original_drag_to'] = adj.get('original_drag_to', {})

        # === Apply overlay: code_edits AFTER coord adjustments (manual edits take priority) ===
        for step in data.get('steps', []):
            si = str(step['original_index'])
            if si in code_edits:
                step['code'] = code_edits[si]

        # === Apply overlay: filter deleted steps ===
        deleted_steps = set(ann.get('deleted_steps', []))
        if deleted_steps:
            data['steps'] = [s for s in data['steps'] if s['original_index'] not in deleted_steps]
            # Re-index for display (but keep original_index for overlay references)
            for i, step in enumerate(data['steps']):
                step['index'] = i

        data['deleted_step_count'] = len(deleted_steps)

        # === Include step errors (reviewer marks) ===
        data['step_errors'] = ann.get('step_errors', {})
        data['annotator_status'] = ann.get('annotator_status', 'not_started')

        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/oss_frame/<path:folder_name>/<float:video_time>')
@any_login_required
def oss_serve_frame(folder_name, video_time):
    """Serve a frame — from pre-extracted cache, screenshot file, or video."""
    local_dir = OSS_CACHE_DIR / folder_name
    oss_folder = request.args.get('folder', 'recordings_0303')

    # Check pre-extracted frame cache first (fastest path)
    frame_cache = local_dir / '_frames' / f'{video_time:.1f}.jpg'
    if frame_cache.exists():
        return Response(frame_cache.read_bytes(), mimetype='image/jpeg',
                        headers={'Cache-Control': 'public, max-age=3600'})

    # Try screenshot file (new format: video_time encodes event ID)
    step_idx = int(video_time)
    screenshots_dir = local_dir / 'screenshots'
    for ext in ('.jpg', '.png'):
        ss_file = screenshots_dir / f'step_{step_idx}{ext}'
        if ss_file.exists():
            return Response(ss_file.read_bytes(), mimetype='image/jpeg')

    # Try downloading screenshot from OSS
    try:
        import oss_client
        bucket = oss_client._get_bucket()
        prefix = oss_folder.rstrip('/') + '/' + folder_name
        for ext in ('.jpg', '.png'):
            try:
                result = bucket.get_object(f"{prefix}/screenshots/step_{step_idx}{ext}")
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                local_path = screenshots_dir / f'step_{step_idx}{ext}'
                with open(local_path, 'wb') as f:
                    for chunk in result:
                        f.write(chunk)
                return Response(local_path.read_bytes(), mimetype='image/jpeg')
            except Exception:
                continue
    except Exception:
        pass

    # Fall back to video frame extraction (old format)
    import cv2
    video_file = None
    for f in local_dir.glob('*.mp4'):
        if 'video_clips' not in str(f):
            video_file = f
            break

    if not video_file:
        try:
            import oss_client
            prefix = oss_folder.rstrip('/') + '/' + folder_name
            video_path = oss_client.download_video(prefix, str(local_dir))
            if video_path:
                video_file = Path(video_path)
        except Exception:
            pass

    if not video_file or not video_file.exists():
        return 'Video/screenshot not found', 404

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
    jpg_bytes = buffer.tobytes()

    # Cache extracted frame for next time
    try:
        fc_dir = local_dir / '_frames'
        fc_dir.mkdir(parents=True, exist_ok=True)
        (fc_dir / f'{video_time:.1f}.jpg').write_bytes(jpg_bytes)
    except Exception:
        pass

    return Response(jpg_bytes, mimetype='image/jpeg',
                    headers={'Cache-Control': 'public, max-age=3600'})

@app.route('/api/oss/annotate', methods=['POST'])
@any_login_required
def api_oss_annotate():
    """Save full annotation for an OSS recording."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    mark = data.get('mark')  # pass/fail/null
    scores = data.get('scores', {})
    pass_reason = data.get('pass_reason', '')
    osworld_overlap = data.get('osworld_overlap', [])
    custom_nodes = data.get('custom_nodes', [])
    related_apps = data.get('related_apps', [])
    step_by_step_instructions = data.get('step_by_step_instructions', '')

    ann_key = f"{oss_folder}/{folder_name}"

    oss_annotations = load_oss_annotations()
    # Preserve existing fields if present
    existing = oss_annotations.get(ann_key, {})
    existing.update({
        'mark': mark,
        'scores': scores,
        'pass_reason': pass_reason,
        'osworld_overlap': osworld_overlap,
        'custom_nodes': custom_nodes,
        'related_apps': related_apps,
        'step_by_step_instructions': step_by_step_instructions,
    })
    # Also save optional fields if provided
    if 'query' in data:
        existing['query'] = data['query']
    if 'justification_edits' in data:
        existing['justification_edits'] = data['justification_edits']
    if 'deleted_steps' in data:
        existing['deleted_steps'] = data['deleted_steps']
    oss_annotations[ann_key] = existing
    save_oss_annotations(oss_annotations)

    # Sync to review_status.json
    review_statuses = load_review_status()
    if mark == 'pass':
        review_statuses[ann_key] = 'reviewed'
    elif mark == 'fail':
        review_statuses[ann_key] = 'rejected'
    else:
        review_statuses[ann_key] = 'unreviewed'
    save_review_status(review_statuses)

    _sync_overlay_to_oss(oss_folder, folder_name)
    return jsonify({'success': True})

@app.route('/api/oss/update_coordinate', methods=['POST'])
@any_login_required
def api_oss_update_coordinate():
    """Save coordinate adjustment for an OSS recording step."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    step_index = data.get('step_index')
    new_x = data.get('x')
    new_y = data.get('y')
    original_x = data.get('original_x')
    original_y = data.get('original_y')

    ann_key = f"{oss_folder}/{folder_name}"
    adj_key = f"{ann_key}_{step_index}"

    coord_adjustments = load_oss_coord_adjustments()

    # Preserve original if already adjusted
    if adj_key in coord_adjustments and 'original' in coord_adjustments[adj_key]:
        original = coord_adjustments[adj_key]['original']
    else:
        original = {'x': original_x, 'y': original_y}

    adj_data = {
        'folder_name': folder_name,
        'oss_folder': oss_folder,
        'step_index': step_index,
        'x': new_x,
        'y': new_y,
        'original': original,
    }
    # Handle drag-end coordinates
    if data.get('drag_to_x') is not None:
        adj_data['drag_to'] = {'x': data['drag_to_x'], 'y': data['drag_to_y']}
        if adj_key in coord_adjustments and 'original_drag_to' in coord_adjustments[adj_key]:
            adj_data['original_drag_to'] = coord_adjustments[adj_key]['original_drag_to']
        else:
            adj_data['original_drag_to'] = {'x': data.get('original_drag_to_x'), 'y': data.get('original_drag_to_y')}

    coord_adjustments[adj_key] = adj_data
    save_oss_coord_adjustments(coord_adjustments)

    _sync_overlay_to_oss(oss_folder, folder_name)
    return jsonify({'success': True})

@app.route('/api/oss/review', methods=['POST'])
@reviewer_api_required
def api_oss_review():
    """Save review status for a recording."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    status = data.get('status', 'unreviewed')

    if status not in ('reviewed', 'rejected', 'unreviewed'):
        return jsonify({'error': 'Invalid status'}), 400

    review_statuses = load_review_status()
    review_key = f"{oss_folder}/{folder_name}"
    review_statuses[review_key] = status
    save_review_status(review_statuses)

    # Sync to oss_annotations.json mark field
    oss_annotations = load_oss_annotations()
    if review_key in oss_annotations:
        if status == 'reviewed':
            oss_annotations[review_key]['mark'] = 'pass'
        elif status == 'rejected':
            oss_annotations[review_key]['mark'] = 'fail'
        else:
            oss_annotations[review_key]['mark'] = None
        save_oss_annotations(oss_annotations)

    _sync_overlay_to_oss(oss_folder, folder_name)
    return jsonify({'success': True})

@app.route('/api/oss/update_justification', methods=['POST'])
@any_login_required
def api_oss_update_justification():
    """Save edited justification for a step of an OSS recording.
    Stores in overlay only - never modifies cached OSS files."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    step_index = data.get('step_index')
    justification = data.get('justification', '')

    ann_key = f"{oss_folder}/{folder_name}"
    oss_annotations = load_oss_annotations()
    existing = oss_annotations.get(ann_key, {})
    edits = existing.get('justification_edits', {})
    edits[str(step_index)] = justification
    existing['justification_edits'] = edits
    oss_annotations[ann_key] = existing
    save_oss_annotations(oss_annotations)

    _sync_overlay_to_oss(oss_folder, folder_name)
    return jsonify({'success': True})


@app.route('/api/oss/update_code', methods=['POST'])
@any_login_required
def api_oss_update_code():
    """Save edited code for a step of an OSS recording.
    Stores in overlay only - never modifies cached OSS files."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    step_index = data.get('step_index')
    code = data.get('code', '')

    ann_key = f"{oss_folder}/{folder_name}"
    oss_annotations = load_oss_annotations()
    existing = oss_annotations.get(ann_key, {})
    edits = existing.get('code_edits', {})
    edits[str(step_index)] = code
    existing['code_edits'] = edits
    oss_annotations[ann_key] = existing
    save_oss_annotations(oss_annotations)

    _sync_overlay_to_oss(oss_folder, folder_name)
    return jsonify({'success': True})


@app.route('/api/oss/update_video_time', methods=['POST'])
@any_login_required
def api_oss_update_video_time():
    """Save adjusted video timestamp for a step's screenshot."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    step_index = data.get('step_index')
    video_time = data.get('video_time', 0)

    ann_key = f"{oss_folder}/{folder_name}"
    oss_annotations = load_oss_annotations()
    existing = oss_annotations.get(ann_key, {})
    edits = existing.get('video_time_edits', {})
    edits[str(step_index)] = video_time
    existing['video_time_edits'] = edits
    oss_annotations[ann_key] = existing
    save_oss_annotations(oss_annotations)

    _sync_overlay_to_oss(oss_folder, folder_name)
    return jsonify({'success': True})


@app.route('/api/oss/update_query', methods=['POST'])
@any_login_required
def api_oss_update_query():
    """Save edited query for an OSS recording.
    Stores in overlay only - never modifies cached OSS files."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    query = data.get('query', '')

    ann_key = f"{oss_folder}/{folder_name}"
    oss_annotations = load_oss_annotations()
    existing = oss_annotations.get(ann_key, {})
    existing['query'] = query
    oss_annotations[ann_key] = existing
    save_oss_annotations(oss_annotations)

    _sync_overlay_to_oss(oss_folder, folder_name)
    return jsonify({'success': True})


@app.route('/api/oss/delete_step', methods=['POST'])
@any_login_required
def api_oss_delete_step():
    """Delete a step from an OSS recording.
    Stores the ORIGINAL step index in overlay - never modifies cached OSS files.
    The original_index refers to the step's position in the unmodified OSS data."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    original_index = data.get('original_index')  # index in original unmodified data

    ann_key = f"{oss_folder}/{folder_name}"

    # Track deleted steps by their ORIGINAL index in overlay
    oss_annotations = load_oss_annotations()
    existing = oss_annotations.get(ann_key, {})
    deleted = existing.get('deleted_steps', [])
    if original_index not in deleted:
        deleted.append(original_index)
    existing['deleted_steps'] = sorted(deleted)
    oss_annotations[ann_key] = existing
    save_oss_annotations(oss_annotations)

    _sync_overlay_to_oss(oss_folder, folder_name)
    return jsonify({'success': True})


@app.route('/api/oss/undelete_step', methods=['POST'])
@any_login_required
def api_oss_undelete_step():
    """Restore a previously deleted step."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    original_index = data.get('original_index')

    ann_key = f"{oss_folder}/{folder_name}"
    oss_annotations = load_oss_annotations()
    existing = oss_annotations.get(ann_key, {})
    deleted = existing.get('deleted_steps', [])
    if original_index in deleted:
        deleted.remove(original_index)
    existing['deleted_steps'] = sorted(deleted)
    oss_annotations[ann_key] = existing
    save_oss_annotations(oss_annotations)
    _sync_overlay_to_oss(oss_folder, folder_name)
    return jsonify({'success': True, 'deleted_steps': deleted})


@app.route('/api/oss/mark_step_error', methods=['POST'])
@reviewer_api_required
def api_oss_mark_step_error():
    """Toggle a step error mark for reviewer feedback to annotator."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    step_index = data.get('step_index')
    is_error = data.get('is_error', True)
    note = data.get('note', '')

    ann_key = f"{oss_folder}/{folder_name}"
    oss_annotations = load_oss_annotations()
    existing = oss_annotations.get(ann_key, {})
    step_errors = existing.get('step_errors', {})
    si = str(step_index)
    if is_error:
        step_errors[si] = {'note': note, 'marked_at': datetime.now().isoformat()}
    else:
        step_errors.pop(si, None)
    existing['step_errors'] = step_errors
    oss_annotations[ann_key] = existing
    save_oss_annotations(oss_annotations)

    _sync_overlay_to_oss(oss_folder, folder_name)
    return jsonify({'success': True, 'error_count': len(step_errors)})


@app.route('/api/oss/set_annotator_status', methods=['POST'])
@any_login_required
def api_oss_set_annotator_status():
    """Set annotation status for a recording (not_started/in_progress/completed)."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    status = data.get('status', 'not_started')
    if status not in ('not_started', 'in_progress', 'completed'):
        return jsonify({'error': 'Invalid status'}), 400
    ann_key = f"{oss_folder}/{folder_name}"
    oss_annotations = load_oss_annotations()
    existing = oss_annotations.get(ann_key, {})
    existing['annotator_status'] = status
    oss_annotations[ann_key] = existing
    save_oss_annotations(oss_annotations)
    _sync_overlay_to_oss(oss_folder, folder_name)
    return jsonify({'success': True, 'status': status})


@app.route('/api/oss/reset_case', methods=['POST'])
@reviewer_api_required
def api_oss_reset_case():
    """Reset a case to its original OSS data by clearing all overlay modifications.
    Also deletes cached metadata files so they get re-downloaded fresh from OSS."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    ann_key = f"{oss_folder}/{folder_name}"

    # Clear annotation overlay for this case
    oss_annotations = load_oss_annotations()
    if ann_key in oss_annotations:
        del oss_annotations[ann_key]
        save_oss_annotations(oss_annotations)

    # Clear coordinate adjustments for this case
    coord_adjustments = load_oss_coord_adjustments()
    keys_to_delete = [k for k in coord_adjustments if k.startswith(ann_key + '_')]
    for k in keys_to_delete:
        del coord_adjustments[k]
    if keys_to_delete:
        save_oss_coord_adjustments(coord_adjustments)

    # Clear review status
    review_statuses = load_review_status()
    if ann_key in review_statuses:
        del review_statuses[ann_key]
        save_review_status(review_statuses)

    # Delete cached metadata files (NOT video) so they get re-downloaded fresh from OSS
    # This ensures that if the cache was ever corrupted by old code, it self-heals
    local_dir = OSS_CACHE_DIR / folder_name
    for fname in ['reduced_events_complete.jsonl', 'reduced_events_vis.jsonl',
                   'metadata.json', 'annotator_info.json', 'task_name.json', 'knowledge_points.json']:
        fpath = local_dir / fname
        if fpath.exists():
            fpath.unlink()

    # Delete overlay from OSS
    try:
        import oss_client
        oss_client.delete_annotation_overlay(oss_folder, folder_name)
    except Exception:
        pass

    return jsonify({'success': True})


@app.route('/api/oss/export_case/<path:folder_name>')
@any_login_required
def api_oss_export_case(folder_name):
    """Export a single graded case as a downloadable zip with JSONL and images."""
    try:
        import cv2

        oss_folder = request.args.get('folder', 'recordings_0303')
        ann_key = f"{oss_folder}/{folder_name}"

        local_dir = OSS_CACHE_DIR / folder_name
        if not local_dir.exists():
            return jsonify({'error': 'Recording not cached locally'}), 404

        # Load data
        task_data = load_oss_task_data(local_dir)
        if not task_data:
            return jsonify({'error': 'Could not load recording data'}), 404

        oss_annotations = load_oss_annotations()
        ann = oss_annotations.get(ann_key, {})
        coord_adjustments = load_oss_coord_adjustments()

        # Apply overlay: coord adjustments, justification edits, code edits, video_time edits, deleted steps
        steps = task_data.get('steps', [])
        justification_edits = ann.get('justification_edits', {})
        code_edits = ann.get('code_edits', {})
        video_time_edits = ann.get('video_time_edits', {})
        deleted_steps = set(ann.get('deleted_steps', []))
        for step in steps:
            si = str(step['index'])
            adj_key_s = f"{ann_key}_{step['index']}"
            if adj_key_s in coord_adjustments:
                adj = coord_adjustments[adj_key_s]
                step['coordinate'] = {'x': adj['x'], 'y': adj['y']}
                # Regenerate code with adjusted coordinates
                x, y = adj['x'], adj['y']
                code = step.get('code', '')
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
            if si in justification_edits:
                step['justification'] = justification_edits[si]
            # Code edits override everything (including coord-regenerated code)
            if si in code_edits:
                step['code'] = code_edits[si]
            if si in video_time_edits:
                step['video_time'] = video_time_edits[si]

        # Filter out deleted steps
        if deleted_steps:
            steps = [s for s in steps if s['index'] not in deleted_steps]

        # Find video
        video_file = None
        for f in local_dir.glob('*.mp4'):
            if 'video_clips' not in str(f):
                video_file = f
                break

        # Build zip in memory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Build export JSONL
            export_lines = []
            traj = []
            for step in steps:
                code = step.get('code', '')
                coord = step.get('coordinate', {})
                x, y = coord.get('x', 0), coord.get('y', 0)
                action = step.get('action', '')

                traj.append({
                    'index': step['index'],
                    'action_type': action,
                    'code': code,
                    'screenshot': f"step_{step['index']}.png",
                    'coordinate': coord,
                    'justification': step.get('justification', ''),
                    'description': step.get('description', ''),
                })
                export_lines.append(json.dumps({
                    'action': action,
                    'coordinate': coord,
                    'justification': step.get('justification', ''),
                    'description': step.get('description', ''),
                    'code': code,
                }, ensure_ascii=False))

            # Write jsonl
            zf.writestr(f"{folder_name}/events.jsonl", '\n'.join(export_lines) + '\n')

            # Write export JSON
            info = task_data.get('annotator_info', {})
            query = ann.get('query', info.get('query', ''))
            export_json = {
                'folder_name': folder_name,
                'query': query,
                'instruction': query,
                'annotator': info.get('username', ''),
                'task_id': info.get('task_id', ''),
                'mark': ann.get('mark'),
                'scores': ann.get('scores', {}),
                'pass_reason': ann.get('pass_reason', ''),
                'step_by_step_instructions': ann.get('step_by_step_instructions', ''),
                'knowledge_points': {
                    'osworld_overlap': ann.get('osworld_overlap', []),
                    'custom_nodes': ann.get('custom_nodes', []),
                },
                'related_apps': ann.get('related_apps', []),
                'traj': traj,
            }
            zf.writestr(f"{folder_name}/export.json", json.dumps(export_json, indent=2, ensure_ascii=False))

            # Extract and save screenshots
            if video_file:
                cap = cv2.VideoCapture(str(video_file))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                for step in steps:
                    frame_num = int(step.get('video_time', 0) * fps)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_num = max(0, min(frame_num, total_frames - 1))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if ret:
                        _, buffer = cv2.imencode('.png', frame)
                        zf.writestr(f"{folder_name}/step_{step['index']}.png", buffer.tobytes())
                cap.release()

        buf.seek(0)
        return Response(
            buf.getvalue(),
            mimetype='application/zip',
            headers={'Content-Disposition': f'attachment; filename={folder_name}.zip'}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


@app.route('/api/oss/export_all')
@reviewer_login_required
def api_oss_export_all():
    """Export all graded cases in a folder as a single zip. Streams progress."""
    import cv2

    oss_folder = request.args.get('folder', 'recordings_0303')
    oss_annotations = load_oss_annotations()
    coord_adjustments = load_oss_coord_adjustments()

    # Find all graded (pass or fail) recordings in this folder
    graded = {}
    for ann_key, ann in oss_annotations.items():
        if ann_key.startswith(oss_folder + '/') and ann.get('mark') in ('pass', 'fail'):
            rec_name = ann_key[len(oss_folder) + 1:]
            graded[rec_name] = ann

    if not graded:
        return jsonify({'error': 'No graded recordings to export'}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        all_export = []

        for rec_name, ann in graded.items():
            local_dir = OSS_CACHE_DIR / rec_name
            if not local_dir.exists():
                continue

            task_data = load_oss_task_data(local_dir)
            if not task_data:
                continue

            ann_key = f"{oss_folder}/{rec_name}"
            steps = task_data.get('steps', [])
            justification_edits = ann.get('justification_edits', {})
            code_edits = ann.get('code_edits', {})
            video_time_edits = ann.get('video_time_edits', {})
            deleted_steps = set(ann.get('deleted_steps', []))

            for step in steps:
                si = str(step['index'])
                adj_key_s = f"{ann_key}_{step['index']}"
                if adj_key_s in coord_adjustments:
                    adj = coord_adjustments[adj_key_s]
                    step['coordinate'] = {'x': adj['x'], 'y': adj['y']}
                    x, y = adj['x'], adj['y']
                    code = step.get('code', '')
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
                if si in justification_edits:
                    step['justification'] = justification_edits[si]
                if si in code_edits:
                    step['code'] = code_edits[si]
                if si in video_time_edits:
                    step['video_time'] = video_time_edits[si]

            # Filter out deleted steps
            if deleted_steps:
                steps = [s for s in steps if s['index'] not in deleted_steps]

            # Build export data
            traj = []
            export_lines = []
            for step in steps:
                coord = step.get('coordinate', {})
                traj.append({
                    'index': step['index'],
                    'action_type': step.get('action', ''),
                    'code': step.get('code', ''),
                    'screenshot': f"step_{step['index']}.png",
                    'coordinate': coord,
                    'justification': step.get('justification', ''),
                    'description': step.get('description', ''),
                })
                export_lines.append(json.dumps({
                    'action': step.get('action', ''),
                    'coordinate': coord,
                    'justification': step.get('justification', ''),
                    'description': step.get('description', ''),
                    'code': step.get('code', ''),
                }, ensure_ascii=False))

            zf.writestr(f"{rec_name}/events.jsonl", '\n'.join(export_lines) + '\n')

            info = task_data.get('annotator_info', {})
            query = ann.get('query', info.get('query', ''))
            export_json = {
                'folder_name': rec_name,
                'query': query,
                'instruction': query,
                'annotator': info.get('username', ''),
                'task_id': info.get('task_id', ''),
                'mark': ann.get('mark'),
                'scores': ann.get('scores', {}),
                'pass_reason': ann.get('pass_reason', ''),
                'step_by_step_instructions': ann.get('step_by_step_instructions', ''),
                'knowledge_points': {
                    'osworld_overlap': ann.get('osworld_overlap', []),
                    'custom_nodes': ann.get('custom_nodes', []),
                },
                'related_apps': ann.get('related_apps', []),
                'traj': traj,
            }
            zf.writestr(f"{rec_name}/export.json", json.dumps(export_json, indent=2, ensure_ascii=False))
            all_export.append(export_json)

            # Extract screenshots
            video_file = None
            for f in local_dir.glob('*.mp4'):
                if 'video_clips' not in str(f):
                    video_file = f
                    break

            if video_file:
                cap = cv2.VideoCapture(str(video_file))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                for step in steps:
                    frame_num = int(step.get('video_time', 0) * fps)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_num = max(0, min(frame_num, total_frames - 1))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if ret:
                        _, buffer = cv2.imencode('.png', frame)
                        zf.writestr(f"{rec_name}/step_{step['index']}.png", buffer.tobytes())
                cap.release()

        # Write combined JSON
        zf.writestr('all_export.json', json.dumps(all_export, indent=2, ensure_ascii=False))

    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype='application/zip',
        headers={'Content-Disposition': f'attachment; filename=export_{oss_folder}.zip'}
    )


@app.route('/api/oss/export_progress')
@reviewer_login_required
def api_oss_export_progress():
    """Return count of graded recordings for progress estimation."""
    oss_folder = request.args.get('folder', 'recordings_0303')
    oss_annotations = load_oss_annotations()
    total = 0
    cached = 0
    for ann_key, ann in oss_annotations.items():
        if ann_key.startswith(oss_folder + '/') and ann.get('mark') in ('pass', 'fail'):
            total += 1
            rec_name = ann_key[len(oss_folder) + 1:]
            if (OSS_CACHE_DIR / rec_name).exists():
                cached += 1
    return jsonify({'total': total, 'cached': cached})


@app.route('/api/oss/ai_check', methods=['POST'])
@any_login_required
def api_oss_ai_check():
    """Start AI quality check for a recording. Runs in background thread."""
    data = request.json
    folder_name = data.get('folder_name', '')
    oss_folder = data.get('oss_folder', 'recordings_0303')
    force = data.get('force', False)
    ann_key = f"{oss_folder}/{folder_name}"

    # Check if already running
    with _ai_check_lock:
        task = _ai_check_tasks.get(ann_key)
        if task and task.get('status') == 'running':
            return jsonify({'status': 'already_running', 'progress': task.get('progress', 0), 'total': task.get('total', 0)})

    # Check if already completed (skip re-run unless forced)
    if not force:
        oss_annotations = load_oss_annotations()
        ann = oss_annotations.get(ann_key, {})
        ai_results = ann.get('ai_check_results', {})
        if ai_results.get('status') == 'completed':
            return jsonify({'status': 'already_completed', 'results': ai_results})

    # Ensure data is cached locally first
    local_dir = OSS_CACHE_DIR / folder_name
    if not local_dir.exists() or not (local_dir / 'reduced_events_complete.jsonl').exists():
        try:
            import oss_client
            prefix = oss_folder.rstrip('/') + '/' + folder_name
            local_dir.mkdir(parents=True, exist_ok=True)
            oss_client.download_recording_metadata_files(prefix, str(local_dir))
            oss_client.download_video(prefix, str(local_dir))
        except Exception as e:
            return jsonify({'status': 'failed', 'error': f'Could not download recording: {e}'}), 500

    # Count total steps for progress display
    total_steps = 0
    try:
        task_data = load_oss_task_data(local_dir)
        if task_data:
            oss_annotations = load_oss_annotations()
            ann = oss_annotations.get(ann_key, {})
            deleted_steps = set(ann.get('deleted_steps', []))
            total_steps = sum(1 for s in task_data.get('steps', []) if s.get('index', 0) not in deleted_steps)
    except Exception:
        pass

    # Start background thread
    thread = threading.Thread(target=_run_ai_check_thread, args=(ann_key, oss_folder, folder_name), daemon=True)
    thread.start()

    return jsonify({'status': 'started', 'total': total_steps})


@app.route('/api/oss/ai_check_status')
@any_login_required
def api_oss_ai_check_status():
    """Get current status of AI check for a recording."""
    folder_name = request.args.get('folder_name', '')
    oss_folder = request.args.get('folder', 'recordings_0303')
    ann_key = f"{oss_folder}/{folder_name}"

    # Check running task first
    with _ai_check_lock:
        task = _ai_check_tasks.get(ann_key)
        if task and task.get('status') == 'running':
            return jsonify(task)

    # Check completed in-memory cache
    with _ai_check_lock:
        task = _ai_check_tasks.get(ann_key)
        if task and task.get('status') in ('completed', 'failed'):
            return jsonify(task)

    # Check saved results in annotations
    oss_annotations = load_oss_annotations()
    ann = oss_annotations.get(ann_key, {})
    ai_results = ann.get('ai_check_results')
    if ai_results:
        return jsonify(ai_results)

    return jsonify({'status': 'not_started'})


@app.route('/oss_review/<path:folder_name>')
@reviewer_login_required
def oss_review_page(folder_name):
    """Review page for a single OSS recording."""
    return render_template('oss_review.html', folder_name=folder_name, page_mode='reviewer')


@app.route('/annotator/<path:folder_name>')
def annotator_page(folder_name):
    """Annotator page - allows corrections with AI check. No review decision access."""
    # Require either reviewer or annotator login
    if not session.get('reviewer_logged_in') and not session.get('annotator_username'):
        return redirect('/annotator_login')
    return render_template('oss_review.html', folder_name=folder_name, page_mode='annotator')


@app.route('/annotator_dashboard')
def annotator_dashboard():
    """Dashboard for annotators to see all their tasks and reviewer feedback."""
    # Require annotator or reviewer session
    ann_user = session.get('annotator_username', '')
    ann_folder = session.get('annotator_folder', '')
    if not ann_user and not session.get('reviewer_logged_in'):
        return redirect('/annotator_login')
    # Auto-redirect with params if coming from session but URL is bare
    if ann_user and not request.args.get('user') and not request.args.get('folder'):
        return redirect(f'/annotator_dashboard?folder={ann_folder}&user={ann_user}')
    return render_template('annotator_dashboard.html')


@app.route('/api/oss/annotator_tasks')
@any_login_required
def api_oss_annotator_tasks():
    """List all tasks for a specific annotator with error/correction status."""
    folder = request.args.get('folder', '')
    username = request.args.get('user', '')

    if not folder or not username:
        return jsonify({'error': 'Missing folder or user'})

    try:
        import time as _t
        refresh = request.args.get('refresh', '0') == '1'
        cached = _dashboard_cache.get(folder)
        # Re-fetch if cache is stale (>120s) or forced refresh or no cache
        if not cached or refresh or (_t.time() - cached.get('_timestamp', 0)) > 120:
            cached = _fetch_dashboard_data(folder)

        annotators = cached.get('annotators', {})
        user_data = None
        for uname, data in annotators.items():
            if uname.lower() == username.lower():
                user_data = data
                break

        if not user_data:
            return jsonify({'error': f'No recordings found for user: {username}', 'recordings': []})

        oss_annotations = load_oss_annotations()
        recordings = []
        for rec in user_data.get('recordings', []):
            ann_key = f"{folder}/{rec['folder_name']}"
            ann = oss_annotations.get(ann_key, {})
            rec_data = dict(rec)
            rec_data['error_count'] = len(ann.get('step_errors', {}))
            rec_data['has_ai_results'] = ann.get('ai_check_results', {}).get('status') == 'completed'
            rec_data['has_corrections'] = bool(ann.get('justification_edits') or ann.get('code_edits'))
            recordings.append(rec_data)

        return jsonify({'recordings': recordings, 'username': username})
    except Exception as e:
        return jsonify({'error': str(e)})


# ============================================================================
# Direct Access Page - Limited access for annotators to fix specific cases
# ============================================================================




@app.route('/edit')
def direct_access_page():
    """Redirect to annotator login."""
    return redirect('/annotator_login')


@app.route('/api/oss/verify_access')
def api_oss_verify_access():
    """Verify that a user has access to a specific recording case.
    Accepts task_id (the visible task identifier) and looks up the folder_name.
    Checks that the case exists and the username matches the annotator."""
    folder = request.args.get('folder', '')
    task_id = request.args.get('task_id', '')
    username = request.args.get('user', '')

    if not folder or not task_id or not username:
        return jsonify({'error': 'Missing required fields: folder, task_id, user'})

    try:
        import oss_client

        # Look up folder_name from task_id
        # Load dashboard data (fetches from OSS if not cached) to get proper task_id mapping
        cached = _dashboard_cache.get(folder)
        if not cached:
            cached = _fetch_dashboard_data(folder)

        matched_folder = None
        matched_username = None
        if cached:
            for ann_data in cached.get('annotators', {}).values():
                for rec in ann_data.get('recordings', []):
                    if rec.get('task_id', '') == task_id or rec.get('task_name', '') == task_id or rec.get('folder_name', '') == task_id:
                        matched_folder = rec['folder_name']
                        matched_username = rec.get('_username', '')
                        break
                if matched_folder:
                    break

        if not matched_folder:
            return jsonify({'error': 'Task ID not found: ' + task_id + '. Please check the task ID and OSS folder name.'})

        # Verify username using cached data or OSS metadata
        actual_username = matched_username or 'Unknown'
        if actual_username == 'Unknown' or not actual_username:
            prefix = folder.rstrip('/') + '/' + matched_folder
            metadata = oss_client.get_recording_metadata(prefix)
            if metadata:
                actual_username = metadata.get('username', 'Unknown')

        # Verify username match (case-insensitive)
        if actual_username.lower() != username.lower() and actual_username != 'Unknown':
            return jsonify({
                'error': 'Access denied. This recording belongs to "' + actual_username + '", not "' + username + '".',
                'access_granted': False
            })

        return jsonify({'access_granted': True, 'annotator': actual_username, 'folder_name': matched_folder})

    except Exception as e:
        return jsonify({'error': 'Failed to verify access: ' + str(e)})


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
    print(f"Direct access: http://{args.host}:{args.port}/edit")
    print("Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
