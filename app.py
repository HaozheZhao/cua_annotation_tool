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
from flask import Flask, render_template_string, send_from_directory, jsonify, request, Response, stream_with_context, session, redirect, url_for
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
        .coord-marker.drag-end {
            border-color: #00c8ff;
            box-shadow: 0 0 12px rgba(0, 200, 255, 0.5);
        }
        .coord-marker.drag-end::before {
            background: #00c8ff;
        }
        .coord-marker.drag-end::after {
            background: #00c8ff;
        }
        .drag-line {
            pointer-events: none;
            z-index: 10;
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
                    ${step.has_coordinate && step.action === 'drag' && step.drag_to ? `
                        <div class="coord-marker" id="coord-marker" style="left:${markerX}px;top:${markerY}px"
                             data-label="from (${step.coordinate.x},${step.coordinate.y})"></div>
                        <div class="coord-marker drag-end" id="coord-marker-end" style="left:${step.drag_to.x * scale}px;top:${step.drag_to.y * scale}px"
                             data-label="to (${step.drag_to.x},${step.drag_to.y})"></div>
                        <svg class="drag-line" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;">
                            <line x1="${markerX}" y1="${markerY}" x2="${step.drag_to.x * scale}" y2="${step.drag_to.y * scale}"
                                  stroke="#ff0040" stroke-width="2" stroke-dasharray="6,3" />
                            <circle cx="${step.drag_to.x * scale}" cy="${step.drag_to.y * scale}" r="4" fill="#00c8ff" />
                        </svg>
                    ` : step.has_coordinate ? `<div class="coord-marker" id="coord-marker" style="left:${markerX}px;top:${markerY}px"
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
            background: #0a0a14;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 50%, #16213e 100%);
            padding: 20px 28px;
            border-bottom: 3px solid transparent;
            border-image: linear-gradient(90deg, #00d9ff, #7c4dff, #00d9ff) 1;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }
        .header h1 {
            font-size: 1.6em;
            background: linear-gradient(135deg, #00d9ff 0%, #7c4dff 50%, #00d9ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        .header-controls {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }
        .header-controls input {
            padding: 10px 14px;
            border: 1px solid #333;
            border-radius: 8px;
            background: rgba(15, 15, 26, 0.8);
            color: #e0e0e0;
            font-size: 0.9em;
            width: 260px;
            transition: border-color 0.2s;
        }
        .header-controls input:focus { border-color: #00d9ff; outline: none; }
        .header-controls button {
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
            font-size: 0.9em;
        }
        .btn-load {
            background: linear-gradient(135deg, #00d9ff, #0099cc);
            color: #000;
            position: relative;
            overflow: hidden;
        }
        .btn-load:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3); }
        .btn-load:disabled { background: #333; color: #666; cursor: not-allowed; transform: none; box-shadow: none; }
        .btn-load.loading::after {
            content: '';
            position: absolute;
            top: 50%; left: 50%;
            width: 18px; height: 18px;
            margin: -9px 0 0 -9px;
            border: 2px solid transparent;
            border-top-color: #000;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
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
            padding: 6px 14px;
            border: 1px solid #00d9ff33;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .nav-link:hover { background: rgba(0,217,255,0.1); border-color: #00d9ff; }

        /* Stats row */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 16px;
            padding: 20px 28px;
            background: #0d0d18;
        }
        .stat-card {
            background: linear-gradient(145deg, #151525 0%, #1a1a2e 100%);
            border-radius: 12px;
            padding: 18px 20px;
            border: 1px solid #252542;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            border-radius: 12px 12px 0 0;
        }
        .stat-card:nth-child(1)::before { background: linear-gradient(90deg, #00d9ff, #0099cc); }
        .stat-card:nth-child(2)::before { background: linear-gradient(90deg, #7c4dff, #536dfe); }
        .stat-card:nth-child(3)::before { background: linear-gradient(90deg, #4caf50, #66bb6a); }
        .stat-card:nth-child(4)::before { background: linear-gradient(90deg, #f44336, #ff5252); }
        .stat-card:nth-child(5)::before { background: linear-gradient(90deg, #888, #aaa); }
        .stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,0.3); border-color: #444; }
        .stat-card .stat-icon {
            font-size: 1.6em;
            margin-bottom: 8px;
        }
        .stat-card .stat-label {
            font-size: 0.78em;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .stat-card .stat-value {
            font-size: 1.8em;
            font-weight: 800;
        }
        .stat-card:nth-child(1) .stat-value { color: #00d9ff; }
        .stat-card:nth-child(2) .stat-value { color: #7c4dff; }
        .stat-card:nth-child(3) .stat-value { color: #4caf50; }
        .stat-card:nth-child(4) .stat-value { color: #f44336; }
        .stat-card:nth-child(5) .stat-value { color: #999; }
        .stat-card .stat-sub {
            font-size: 0.75em;
            color: #666;
            margin-top: 4px;
        }

        .container { padding: 20px 28px; }

        /* Annotator cards */
        .annotator-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 18px;
            margin-top: 8px;
        }
        .annotator-card {
            background: linear-gradient(145deg, #151525 0%, #1a1a2e 100%);
            border-radius: 12px;
            border: 1px solid #252542;
            overflow: hidden;
            transition: all 0.3s;
            animation: fadeIn 0.4s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .annotator-card:hover { border-color: #00d9ff44; transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.25); }
        .annotator-header {
            padding: 16px 18px;
            background: linear-gradient(135deg, rgba(30, 58, 95, 0.5) 0%, rgba(22, 33, 62, 0.5) 100%);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .annotator-header h3 {
            color: #e0e0e0;
            font-size: 1em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1em;
            color: white;
            flex-shrink: 0;
        }
        .annotator-meta {
            display: flex;
            flex-direction: column;
        }
        .annotator-meta .username { color: #e0e0e0; font-weight: 700; }
        .annotator-meta .task-count { color: #888; font-size: 0.8em; font-weight: normal; }
        .annotator-stats {
            display: flex;
            gap: 8px;
            font-size: 0.78em;
        }
        .stat-badge {
            padding: 3px 10px;
            border-radius: 20px;
            font-weight: bold;
        }
        .stat-badge.reviewed { background: rgba(76, 175, 80, 0.2); color: #4caf50; border: 1px solid rgba(76, 175, 80, 0.3); }
        .stat-badge.rejected { background: rgba(244, 67, 54, 0.2); color: #f44336; border: 1px solid rgba(244, 67, 54, 0.3); }
        .stat-badge.unreviewed { background: rgba(150, 150, 150, 0.15); color: #999; border: 1px solid rgba(150, 150, 150, 0.2); }

        /* Progress bar in annotator card */
        .annotator-progress {
            padding: 0 18px 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .progress-track {
            flex: 1;
            height: 6px;
            background: #252542;
            border-radius: 3px;
            overflow: hidden;
            display: flex;
        }
        .progress-fill-reviewed {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #66bb6a);
            transition: width 0.5s ease;
        }
        .progress-fill-rejected {
            height: 100%;
            background: linear-gradient(90deg, #f44336, #ff5252);
            transition: width 0.5s ease;
        }
        .progress-pct {
            font-size: 0.75em;
            color: #888;
            min-width: 36px;
            text-align: right;
        }

        .expand-icon {
            transition: transform 0.3s;
            font-size: 0.75em;
            color: #666;
        }
        .expand-icon.expanded { transform: rotate(90deg); }

        .task-list {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease;
        }
        .task-list.expanded { max-height: 3000px; }
        .task-entry {
            padding: 10px 18px;
            border-top: 1px solid #1e1e35;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85em;
            transition: background 0.2s;
            animation: fadeIn 0.3s ease;
        }
        .task-entry:hover { background: rgba(0, 217, 255, 0.03); }
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
        .task-entry .task-date { color: #555; font-size: 0.8em; }
        .task-entry .task-actions { display: flex; gap: 8px; align-items: center; }
        .review-badge {
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.78em;
            font-weight: bold;
        }
        .review-badge.reviewed { background: rgba(76, 175, 80, 0.2); color: #4caf50; }
        .review-badge.rejected { background: rgba(244, 67, 54, 0.2); color: #f44336; }
        .review-badge.unreviewed { background: rgba(150, 150, 150, 0.1); color: #666; }
        .btn-view {
            padding: 5px 14px;
            border: 1px solid #00d9ff44;
            border-radius: 6px;
            background: transparent;
            color: #00d9ff;
            cursor: pointer;
            font-size: 0.85em;
            text-decoration: none;
            transition: all 0.2s;
        }
        .btn-view:hover { background: #00d9ff; color: #000; border-color: #00d9ff; }
        .empty-state {
            text-align: center;
            padding: 80px 20px;
            color: #555;
        }
        .empty-state .empty-icon { font-size: 4em; margin-bottom: 20px; opacity: 0.3; }
        .empty-state h2 { color: #444; margin-bottom: 12px; font-size: 1.4em; }
        .empty-state p { color: #555; font-size: 1em; }
        .btn-export-all {
            padding: 10px 24px;
            border: none; border-radius: 8px;
            background: linear-gradient(135deg, #4caf50, #2e7d32);
            color: white; cursor: pointer;
            font-weight: bold; font-size: 0.9em;
            transition: all 0.3s;
        }
        .btn-export-all:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(76,175,80,0.3); }
        .btn-export-all:disabled { background: #333; color: #666; cursor: not-allowed; transform: none; box-shadow: none; }
    </style>
</head>
<body>
    <div class="header">
        <h1>OSS Recording Dashboard</h1>
        <div class="header-controls">
            <a href="/" class="nav-link">&#8592; Local Review</a>
            <a href="/logout" class="nav-link" style="border-color:#f44336;color:#f44336;">Logout</a>
            <input type="text" id="ossFolder" value="recordings_0303" placeholder="OSS upload folder..." />
            <button class="btn-load" id="loadBtn" onclick="loadDashboard()">Load</button>
            <label class="auto-poll-label">
                <input type="checkbox" id="autoPoll" /> Auto-refresh (60s)
            </label>
            <button class="btn-export-all" id="exportAllBtn" onclick="exportAll()" style="display:none;">Export All Graded</button>
        </div>
    </div>
    <!-- Export progress overlay -->
    <div id="exportOverlay" style="display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(15,15,26,0.92);z-index:2000;justify-content:center;align-items:center;">
        <div style="background:#1a1a2e;border:1px solid #333;border-radius:12px;padding:30px 40px;text-align:center;min-width:360px;">
            <div style="color:#00d9ff;font-size:1.2em;margin-bottom:16px;" id="exportTitle">Exporting...</div>
            <div style="background:#252542;border-radius:6px;height:12px;overflow:hidden;margin-bottom:12px;">
                <div id="exportProgressBar" style="height:100%;background:linear-gradient(90deg,#00d9ff,#7c4dff);width:0%;transition:width 0.3s;"></div>
            </div>
            <div style="color:#888;font-size:0.85em;" id="exportStatus">Preparing export...</div>
        </div>
    </div>
    <div class="stats-row" id="statsRow" style="display:none;">
        <div class="stat-card">
            <div class="stat-icon">&#128202;</div>
            <div class="stat-label">Total</div>
            <div class="stat-value" id="totalCount">0</div>
            <div class="stat-sub">recordings loaded</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">&#128101;</div>
            <div class="stat-label">Annotators</div>
            <div class="stat-value" id="annotatorCount">0</div>
            <div class="stat-sub">contributors</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">&#9989;</div>
            <div class="stat-label">Reviewed</div>
            <div class="stat-value" id="reviewedCount">0</div>
            <div class="stat-sub" id="reviewedPct">0% complete</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">&#10060;</div>
            <div class="stat-label">Rejected</div>
            <div class="stat-value" id="rejectedCount">0</div>
            <div class="stat-sub" id="rejectedPct">0% of total</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">&#9203;</div>
            <div class="stat-label">Unreviewed</div>
            <div class="stat-value" id="unreviewedCount">0</div>
            <div class="stat-sub" id="unreviewedPct">0% remaining</div>
        </div>
    </div>
    <div class="container" id="content">
        <div class="empty-state">
            <div class="empty-icon">&#128194;</div>
            <h2>No data loaded</h2>
            <p>Enter an OSS upload folder name above and click Load to get started.</p>
        </div>
    </div>

    <script>
        let pollInterval = null;
        let dashboardData = null;

        function hashColor(str) {
            let hash = 0;
            for (let i = 0; i < str.length; i++) {
                hash = str.charCodeAt(i) + ((hash << 5) - hash);
            }
            const hue = Math.abs(hash) % 360;
            return 'hsl(' + hue + ', 55%, 45%)';
        }

        async function loadDashboard() {
            const folder = document.getElementById('ossFolder').value.trim();
            if (!folder) return;

            const loadBtn = document.getElementById('loadBtn');
            loadBtn.disabled = true;
            loadBtn.textContent = '';
            loadBtn.classList.add('loading');

            try {
                const resp = await fetch('/api/oss/dashboard_data?folder=' + encodeURIComponent(folder));
                const data = await resp.json();

                if (data.error) {
                    document.getElementById('content').innerHTML =
                        '<div class="empty-state"><div class="empty-icon">&#9888;</div><h2>Error</h2><p>' + data.error + '</p></div>';
                    return;
                }

                dashboardData = data;
                renderDashboard(data);

                document.getElementById('statsRow').style.display = 'grid';
                document.getElementById('exportAllBtn').style.display = 'inline-block';

            } catch (err) {
                document.getElementById('content').innerHTML =
                    '<div class="empty-state"><div class="empty-icon">&#9888;</div><h2>Error</h2><p>' + err.message + '</p></div>';
            } finally {
                loadBtn.disabled = false;
                loadBtn.textContent = 'Load';
                loadBtn.classList.remove('loading');
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

            const pct = totalAll > 0 ? ((reviewedAll / totalAll) * 100).toFixed(1) : 0;
            const rejPct = totalAll > 0 ? ((rejectedAll / totalAll) * 100).toFixed(1) : 0;
            const unrPct = totalAll > 0 ? ((unreviewedAll / totalAll) * 100).toFixed(1) : 0;
            document.getElementById('reviewedPct').textContent = pct + '% complete';
            document.getElementById('rejectedPct').textContent = rejPct + '% of total';
            document.getElementById('unreviewedPct').textContent = unrPct + '% remaining';

            if (keys.length === 0) {
                document.getElementById('content').innerHTML =
                    '<div class="empty-state"><div class="empty-icon">&#128269;</div><h2>No recordings found</h2><p>Check the OSS folder name and try again.</p></div>';
                return;
            }

            let html = '<div class="annotator-grid">';
            keys.forEach((username, idx) => {
                const a = annotators[username];
                html += renderAnnotatorCard(username, a, idx);
            });
            html += '</div>';

            document.getElementById('content').innerHTML = html;
        }

        function renderAnnotatorCard(username, data, idx) {
            const folder = document.getElementById('ossFolder').value.trim();
            const color = hashColor(username);
            const letter = username.charAt(0).toUpperCase();
            const completePct = data.total > 0 ? ((data.reviewed + data.rejected) / data.total * 100).toFixed(0) : 0;
            const reviewedPct = data.total > 0 ? (data.reviewed / data.total * 100) : 0;
            const rejectedPct = data.total > 0 ? (data.rejected / data.total * 100) : 0;

            let tasksHtml = '';
            (data.recordings || []).forEach(rec => {
                const statusClass = rec.review_status || 'unreviewed';
                const statusLabel = statusClass.charAt(0).toUpperCase() + statusClass.slice(1);
                const query = rec.query || rec.task_id || '-';
                const date = rec.upload_timestamp ? rec.upload_timestamp.split('T')[0] : '-';
                tasksHtml +=
                    '<div class="task-entry">' +
                    '<div class="task-info">' +
                    '<span class="task-id">' + (rec.task_name || rec.task_id || rec.folder_name) + '</span>' +
                    '<span class="task-query" title="' + query + '">' + query + '</span>' +
                    '<span class="task-date">' + date + '</span>' +
                    '</div>' +
                    '<div class="task-actions">' +
                    '<span class="review-badge ' + statusClass + '">' + statusLabel + '</span>' +
                    '<a class="btn-view" href="/oss_review/' + encodeURIComponent(rec.folder_name) + '?folder=' + encodeURIComponent(folder) + '">View</a>' +
                    '</div></div>';
            });

            return '<div class="annotator-card" style="animation-delay:' + (idx * 0.05) + 's">' +
                '<div class="annotator-header" onclick="toggleTaskList(\\'' + username + '\\')">' +
                '<h3>' +
                '<span class="expand-icon" id="icon-' + username + '">&#9654;</span>' +
                '<span class="avatar" style="background:' + color + '">' + letter + '</span>' +
                '<span class="annotator-meta"><span class="username">' + username + '</span>' +
                '<span class="task-count">' + data.total + ' recordings</span></span>' +
                '</h3>' +
                '<div class="annotator-stats">' +
                '<span class="stat-badge reviewed">' + data.reviewed + '</span>' +
                '<span class="stat-badge rejected">' + data.rejected + '</span>' +
                '<span class="stat-badge unreviewed">' + data.unreviewed + '</span>' +
                (data.error_steps > 0 ? '<span class="stat-badge" style="background:rgba(244,67,54,0.15);color:#f44336;border:1px solid rgba(244,67,54,0.3);">' + data.corrected_steps + '/' + data.error_steps + ' fixed</span>' : '') +
                '</div></div>' +
                '<div class="annotator-progress">' +
                '<div class="progress-track">' +
                '<div class="progress-fill-reviewed" style="width:' + reviewedPct + '%"></div>' +
                '<div class="progress-fill-rejected" style="width:' + rejectedPct + '%"></div>' +
                '</div>' +
                '<span class="progress-pct">' + completePct + '%</span>' +
                '</div>' +
                '<div class="task-list" id="tasks-' + username + '">' + tasksHtml + '</div>' +
                '</div>';
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

        async function exportAll() {
            const folder = document.getElementById('ossFolder').value.trim();
            if (!folder) return;

            const overlay = document.getElementById('exportOverlay');
            overlay.style.display = 'flex';
            document.getElementById('exportProgressBar').style.width = '10%';
            document.getElementById('exportStatus').textContent = 'Checking graded recordings...';

            try {
                // First check how many to export
                const checkResp = await fetch('/api/oss/export_progress?folder=' + encodeURIComponent(folder));
                const checkData = await checkResp.json();

                if (checkData.total === 0) {
                    document.getElementById('exportTitle').textContent = 'Nothing to export';
                    document.getElementById('exportStatus').textContent = 'No graded recordings found. Grade some recordings first.';
                    document.getElementById('exportProgressBar').style.width = '100%';
                    document.getElementById('exportProgressBar').style.background = '#f44336';
                    setTimeout(() => { overlay.style.display = 'none'; }, 2000);
                    return;
                }

                document.getElementById('exportStatus').textContent =
                    'Exporting ' + checkData.total + ' graded recordings (' + checkData.cached + ' cached)...';
                document.getElementById('exportProgressBar').style.width = '30%';

                // Trigger the export download
                document.getElementById('exportProgressBar').style.width = '50%';
                document.getElementById('exportStatus').textContent = 'Building ZIP file with screenshots...';

                // Use XMLHttpRequest for progress tracking
                const xhr = new XMLHttpRequest();
                xhr.open('GET', '/api/oss/export_all?folder=' + encodeURIComponent(folder), true);
                xhr.responseType = 'blob';

                xhr.onprogress = function(e) {
                    if (e.lengthComputable) {
                        const pct = Math.round(50 + (e.loaded / e.total) * 50);
                        document.getElementById('exportProgressBar').style.width = pct + '%';
                    } else {
                        document.getElementById('exportProgressBar').style.width = '75%';
                    }
                };

                xhr.onload = function() {
                    if (xhr.status === 200) {
                        document.getElementById('exportProgressBar').style.width = '100%';
                        document.getElementById('exportStatus').textContent = 'Download starting...';

                        // Trigger download
                        const blob = xhr.response;
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'export_' + folder + '.zip';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);

                        document.getElementById('exportTitle').textContent = 'Export Complete!';
                        document.getElementById('exportStatus').textContent = checkData.total + ' recordings exported successfully.';
                        setTimeout(() => { overlay.style.display = 'none'; }, 2000);
                    } else {
                        document.getElementById('exportTitle').textContent = 'Export Failed';
                        document.getElementById('exportStatus').textContent = 'Server returned error ' + xhr.status;
                        document.getElementById('exportProgressBar').style.background = '#f44336';
                        setTimeout(() => { overlay.style.display = 'none'; }, 3000);
                    }
                };

                xhr.onerror = function() {
                    document.getElementById('exportTitle').textContent = 'Export Failed';
                    document.getElementById('exportStatus').textContent = 'Network error occurred';
                    document.getElementById('exportProgressBar').style.background = '#f44336';
                    setTimeout(() => { overlay.style.display = 'none'; }, 3000);
                };

                xhr.send();

            } catch (err) {
                document.getElementById('exportTitle').textContent = 'Export Failed';
                document.getElementById('exportStatus').textContent = err.message;
                document.getElementById('exportProgressBar').style.background = '#f44336';
                setTimeout(() => { overlay.style.display = 'none'; }, 3000);
            }
        }
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
        html, body {
            height: 100%;
            overflow: hidden;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #00d9ff;
            flex-wrap: wrap;
            gap: 6px;
        }
        .header h1 { font-size: 1.1em; color: #00d9ff; }
        .header-info { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
        .header-info span { color: #888; font-size: 0.8em; }
        .header-info b { color: #e0e0e0; }
        .btn-back {
            padding: 5px 14px;
            border: 1px solid #00d9ff;
            border-radius: 4px;
            background: transparent;
            color: #00d9ff;
            cursor: pointer;
            text-decoration: none;
            font-size: 0.85em;
            transition: all 0.2s;
        }
        .btn-back:hover { background: #00d9ff; color: #000; }
        .mode-toggle {
            display: flex;
            background: #16213e;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid #333;
        }
        .mode-toggle button {
            padding: 5px 14px;
            border: none;
            background: transparent;
            color: #888;
            cursor: pointer;
            font-weight: bold;
            font-size: 0.8em;
            transition: all 0.2s;
        }
        .mode-toggle button.active { background: #00d9ff; color: #000; }
        .mode-toggle button:hover:not(.active) { color: #ccc; }

        /* Main layout: recording sidebar | step sidebar | content */
        .main-layout {
            flex: 1;
            display: flex;
            overflow: hidden;
        }

        /* Recording sidebar (left) */
        .rec-sidebar {
            width: 260px;
            background: #0d1020;
            border-right: 1px solid #252542;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .rec-sidebar-header {
            padding: 10px 12px;
            background: #16213e;
            border-bottom: 1px solid #252542;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .rec-sidebar-header h3 { color: #888; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; }
        .rec-sidebar-header .rec-count { color: #00d9ff; font-size: 0.8em; font-weight: bold; }
        .rec-filter {
            padding: 6px 12px;
            border-bottom: 1px solid #252542;
        }
        .rec-filter input {
            width: 100%;
            padding: 5px 8px;
            border: 1px solid #333;
            border-radius: 4px;
            background: #0f0f1a;
            color: #e0e0e0;
            font-size: 0.8em;
        }
        .rec-filter input:focus { border-color: #00d9ff; outline: none; }
        .rec-list {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 4px 6px;
            scrollbar-width: thin;
            scrollbar-color: #333 #0d1020;
        }
        .rec-list::-webkit-scrollbar { width: 6px; }
        .rec-list::-webkit-scrollbar-track { background: #0d1020; }
        .rec-list::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
        .rec-list::-webkit-scrollbar-thumb:hover { background: #555; }

        /* Person group */
        .person-group { margin-bottom: 4px; }
        .person-header {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 7px 8px;
            background: #151530;
            border-radius: 5px;
            cursor: pointer;
            user-select: none;
            transition: background 0.15s;
        }
        .person-header:hover { background: #1e1e40; }
        .person-toggle {
            font-size: 0.65em;
            color: #666;
            transition: transform 0.2s;
            flex-shrink: 0;
            width: 12px;
            text-align: center;
        }
        .person-toggle.expanded { transform: rotate(90deg); }
        .person-avatar {
            width: 22px; height: 22px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: 800; font-size: 0.7em; color: white;
            flex-shrink: 0;
        }
        .person-name {
            color: #ccc;
            font-size: 0.8em;
            font-weight: 600;
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .person-stats {
            display: flex;
            gap: 4px;
            flex-shrink: 0;
        }
        .person-stat {
            font-size: 0.65em;
            padding: 1px 5px;
            border-radius: 8px;
            font-weight: bold;
        }
        .person-stat.rev { background: rgba(76,175,80,0.2); color: #4caf50; }
        .person-stat.rej { background: rgba(244,67,54,0.2); color: #f44336; }
        .person-stat.total { background: rgba(0,217,255,0.1); color: #00d9ff; }
        .person-tasks {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.25s ease;
            padding-left: 10px;
        }
        .person-tasks.expanded { max-height: 5000px; }

        .rec-item {
            padding: 6px 8px;
            margin: 2px 0;
            background: #131322;
            border-radius: 5px;
            cursor: pointer;
            border-left: 3px solid transparent;
            transition: all 0.15s;
            font-size: 0.75em;
        }
        .rec-item:hover { background: #1e1e35; }
        .rec-item.active { border-left-color: #00d9ff; background: #1e1e35; }
        .rec-item.reviewed { border-left-color: #4caf50; }
        .rec-item.rejected { border-left-color: #f44336; }
        .rec-item .rec-title {
            color: #ccc;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 1px;
        }
        .rec-item .rec-meta {
            color: #666;
            font-size: 0.85em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .rec-item .rec-status {
            display: inline-block;
            padding: 1px 5px;
            border-radius: 8px;
            font-size: 0.8em;
            font-weight: bold;
            margin-top: 1px;
        }
        .rec-item .rec-status.reviewed { background: rgba(76,175,80,0.2); color: #4caf50; }
        .rec-item .rec-status.rejected { background: rgba(244,67,54,0.2); color: #f44336; }
        .rec-item .rec-status.unreviewed { background: rgba(150,150,150,0.1); color: #666; }

        /* Step sidebar */
        .step-sidebar {
            width: 180px;
            min-width: 180px;
            background: #16213e;
            overflow-y: auto;
            padding: 8px;
            border-right: 1px solid #333;
            flex-shrink: 0;
        }
        .step-sidebar-header {
            padding: 6px 8px;
            color: #888;
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .step-item {
            padding: 6px 8px;
            margin: 2px 0;
            background: #1a1a2e;
            border-radius: 4px;
            cursor: pointer;
            border-left: 3px solid transparent;
            transition: all 0.15s;
            font-size: 0.8em;
        }
        .step-item:hover { background: #252542; }
        .step-item.active { border-left-color: #00d9ff; background: #252542; }
        .step-item .step-num { color: #00d9ff; font-weight: bold; }
        .step-item .step-action { color: #888; font-size: 0.9em; }

        /* Content area */
        .content-area {
            flex: 1;
            padding: 12px 16px;
            overflow-y: auto;
        }
        .annotator-info-bar {
            background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
            padding: 10px 14px;
            border-radius: 6px;
            margin-bottom: 10px;
            border-left: 3px solid #ffc107;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            font-size: 0.85em;
        }
        .annotator-info-bar .info-item span { color: #888; }
        .annotator-info-bar .info-item b { color: #ffc107; word-break: break-word; }

        /* Screenshot */
        .screenshot-container {
            position: relative;
            display: inline-block;
            margin-bottom: 10px;
            background: #000;
            border-radius: 6px;
            overflow: hidden;
            max-width: 100%;
        }
        .screenshot-container img {
            max-width: 100%;
            max-height: 50vh;
            display: block;
        }
        .coord-marker {
            position: absolute;
            width: 28px;
            height: 28px;
            border: 3px solid #ff0040;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            animation: pulse 1.5s infinite;
            box-shadow: 0 0 12px rgba(255, 0, 64, 0.5);
        }
        .coord-marker::before {
            content: '';
            position: absolute;
            top: 50%; left: 50%;
            width: 6px; height: 6px;
            background: #ff0040;
            border-radius: 50%;
            transform: translate(-50%, -50%);
        }
        .coord-marker::after {
            content: attr(data-label);
            position: absolute;
            top: -24px; left: 50%;
            transform: translateX(-50%);
            background: #ff0040;
            color: white;
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 9px;
            font-weight: bold;
            white-space: nowrap;
        }
        .coord-marker.drag-end {
            border-color: #00c8ff;
            box-shadow: 0 0 12px rgba(0, 200, 255, 0.5);
        }
        .coord-marker.drag-end::before {
            background: #00c8ff;
        }
        .coord-marker.drag-end::after {
            background: #00c8ff;
        }
        .drag-line {
            position: absolute;
            pointer-events: none;
            z-index: 10;
        }
        .drag-line line {
            stroke: #ff0040;
            stroke-width: 2;
            stroke-dasharray: 6, 3;
        }
        .drag-line polygon {
            fill: #00c8ff;
        }
        @keyframes pulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
            50% { transform: translate(-50%, -50%) scale(1.15); opacity: 0.8; }
        }

        /* Step details */
        .step-details {
            background: #1a1a2e;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
            border: 1px solid #333;
        }
        .step-details h4 { color: #00d9ff; margin-bottom: 6px; font-size: 0.9em; }
        .step-details .code {
            background: #0d1117;
            padding: 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            color: #7ee787;
            margin-bottom: 6px;
        }
        .code-editable {
            position: relative;
        }
        .code-edit-input {
            width: 100%;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            color: #7ee787;
            font-family: 'Courier New', monospace;
            font-size: 1em;
            padding: 4px 6px;
            box-sizing: border-box;
        }
        .code-edit-input:focus {
            outline: none;
            border-color: #58a6ff;
            box-shadow: 0 0 0 2px rgba(88,166,255,0.3);
        }
        .step-details .description { color: #ccc; line-height: 1.4; font-size: 0.9em; }
        .step-details .justification {
            color: #ffc107;
            margin-top: 6px;
            padding-top: 6px;
            border-top: 1px solid #333;
            border-left: 3px solid #ffc107;
            padding-left: 8px;
            font-size: 0.85em;
        }
        .adjusted-badge {
            display: inline-block;
            background: #ff9800;
            color: #000;
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 0.7em;
            font-weight: bold;
            margin-left: 6px;
        }
        .original-coord { color: #888; font-size: 0.8em; text-decoration: line-through; margin-left: 6px; }
        .current-coord { color: #4caf50; font-weight: bold; }

        /* Review panel */
        .review-panel {
            background: #1a1a2e;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #333;
            margin-bottom: 10px;
        }
        .review-panel h4 { color: #ffc107; margin-bottom: 10px; font-size: 0.9em; }
        .review-buttons { display: flex; gap: 8px; }
        .review-btn {
            padding: 8px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
            font-size: 0.9em;
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

        /* Annotation panels */
        .annotation-panel {
            background: #1a1a2e;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
            border: 1px solid #333;
        }
        .annotation-panel h4 { color: #ffc107; margin-bottom: 10px; font-size: 0.9em; }
        .mark-buttons { display: flex; gap: 8px; margin-bottom: 10px; }
        .mark-btn {
            padding: 8px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
            flex: 1;
            font-size: 0.9em;
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

        .case-evaluation {
            background: linear-gradient(135deg, #1a2e1a 0%, #1a1a2e 100%);
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
            border: 1px solid #2e7d32;
        }
        .case-evaluation h4 { color: #4caf50; margin-bottom: 10px; font-size: 0.9em; }
        .case-evaluation.fail-mode {
            background: linear-gradient(135deg, #2e1a1a 0%, #1a1a2e 100%);
            border-color: #c62828;
        }
        .case-evaluation.fail-mode h4 { color: #f44336; }
        .rating-section {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 12px;
        }
        .rating-item { background: #0d1117; padding: 10px; border-radius: 5px; }
        .rating-item label { display: block; color: #888; font-size: 0.8em; margin-bottom: 6px; }
        .rating-item .label-primary { color: #ffc107; font-size: 0.7em; margin-left: 4px; }
        .rating-stars { display: flex; gap: 3px; }
        .rating-star {
            width: 28px; height: 28px;
            border: none;
            background: #333;
            color: #666;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }
        .rating-star:hover { background: #444; color: #aaa; }
        .rating-star.active { background: #ffc107; color: #000; }
        .pass-reason-input {
            width: 100%;
            padding: 8px;
            border: 1px solid #444;
            border-radius: 5px;
            background: #0d1117;
            color: #e0e0e0;
            resize: vertical;
            min-height: 60px;
            font-family: inherit;
            font-size: 0.85em;
        }
        .pass-reason-input::placeholder { color: #555; }

        .knowledge-panel {
            background: linear-gradient(135deg, #1a1a3e 0%, #16213e 100%);
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
            border: 1px solid #4a4a8a;
        }
        .knowledge-panel h4 { color: #9c88ff; margin-bottom: 10px; font-size: 0.9em; }
        .knowledge-section {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 10px;
        }
        .knowledge-item { background: #0d1117; padding: 10px; border-radius: 5px; }
        .knowledge-item label {
            display: block; color: #9c88ff;
            font-size: 0.8em; margin-bottom: 6px; font-weight: bold;
        }
        .knowledge-item .label-hint { color: #666; font-size: 0.75em; font-weight: normal; }
        .knowledge-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; min-height: 24px; }
        .knowledge-tag {
            display: inline-flex; align-items: center;
            background: #4a4a8a; color: white;
            padding: 3px 8px; border-radius: 12px; font-size: 0.75em;
        }
        .knowledge-tag .remove-tag { margin-left: 5px; cursor: pointer; opacity: 0.7; }
        .knowledge-tag .remove-tag:hover { opacity: 1; }
        .knowledge-input-row { display: flex; gap: 4px; }
        .knowledge-input {
            flex: 1; padding: 4px 8px;
            border: 1px solid #444; border-radius: 4px;
            background: #1a1a2e; color: #e0e0e0; font-size: 0.8em;
        }
        .knowledge-add-btn {
            padding: 4px 10px; background: #4a4a8a;
            border: none; border-radius: 4px;
            color: white; cursor: pointer; font-size: 0.8em;
        }
        .knowledge-add-btn:hover { background: #5a5a9a; }
        .step-instructions-input {
            width: 100%; padding: 8px;
            border: 1px solid #444; border-radius: 5px;
            background: #0d1117; color: #e0e0e0;
            resize: vertical; min-height: 50px;
            font-family: inherit; font-size: 0.85em; margin-top: 6px;
        }

        /* Coordinate fine-tuning */
        .coord-adjust-panel {
            display: none;
            background: #252542;
            padding: 8px 12px;
            border-radius: 5px;
            margin-bottom: 8px;
            border: 1px solid #ffc107;
        }
        .coord-adjust-panel.show { display: block; }
        .coord-adjust-panel h5 { color: #ffc107; margin-bottom: 8px; font-size: 0.85em; }
        .coord-adjust-controls { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
        .coord-field { display: flex; align-items: center; gap: 4px; }
        .coord-field label { color: #888; font-size: 0.8em; }
        .coord-input {
            width: 65px; padding: 4px 6px;
            border: 1px solid #444; border-radius: 4px;
            background: #0d1117; color: #e0e0e0;
            font-family: monospace; font-size: 0.85em; text-align: center;
        }
        .coord-input:focus { border-color: #00d9ff; outline: none; }
        .finetune-btn {
            padding: 4px 10px; background: #ffc107;
            border: none; border-radius: 4px;
            color: #000; font-weight: bold;
            cursor: pointer; font-size: 0.8em;
        }
        .finetune-btn:hover { background: #ffca28; }
        .finetune-btn.save { background: #4caf50; color: white; }
        .finetune-btn.save:hover { background: #66bb6a; }
        .finetune-btn.cancel { background: #666; color: white; }
        .finetune-btn.cancel:hover { background: #888; }
        .image-info {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 6px; padding: 6px 10px;
            background: #1a1a2e; border-radius: 5px; border: 1px solid #333;
        }
        .resolution-info { color: #00d9ff; font-size: 0.8em; font-family: monospace; }

        .nav-buttons { display: flex; gap: 8px; margin-bottom: 10px; align-items: center; }
        .nav-btn {
            padding: 6px 16px; border: 1px solid #444;
            border-radius: 5px; background: #1a1a2e;
            color: #e0e0e0; cursor: pointer; transition: all 0.2s;
        }
        .nav-btn:hover { border-color: #00d9ff; color: #00d9ff; }
        .nav-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .step-counter { color: #888; font-size: 0.85em; display: flex; align-items: center; }

        /* View mode toggle */
        .view-toggle {
            display: flex; gap: 4px; margin-left: 12px;
            background: #16213e; border-radius: 4px; overflow: hidden; border: 1px solid #333;
        }
        .view-toggle button {
            padding: 4px 10px; border: none; background: transparent;
            color: #888; cursor: pointer; font-size: 0.75em; font-weight: bold; transition: all 0.2s;
        }
        .view-toggle button.active { background: #00d9ff; color: #000; }
        .view-toggle button:hover:not(.active) { color: #ccc; }
        .page-size-select {
            padding: 4px 6px; border: 1px solid #444; border-radius: 4px;
            background: #0d1117; color: #e0e0e0; font-size: 0.8em; margin-left: 8px;
        }

        /* Stacked single view (all steps listed) */
        .steps-stacked {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .stacked-step-card {
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 10px;
            overflow: hidden;
            padding: 16px;
            transition: border-color 0.2s;
        }
        .stacked-step-card:hover { border-color: #555; }
        .stacked-step-card.active { border-color: #00d9ff; box-shadow: 0 0 10px rgba(0,217,255,0.15); }
        .stacked-step-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 10px;
        }
        .stacked-step-header h4 { color: #00d9ff; margin: 0; font-size: 1em; }
        .stacked-img-container {
            position: relative;
            display: inline-block;
            background: #000;
            border-radius: 6px;
            overflow: hidden;
            max-width: 100%;
            margin-bottom: 10px;
        }
        .stacked-img-container img {
            max-width: 100%;
            max-height: 55vh;
            display: block;
        }

        /* Grid view */
        .steps-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }
        .grid-card {
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 10px;
            overflow: hidden;
            transition: all 0.2s;
            cursor: pointer;
        }
        .grid-card:hover { border-color: #00d9ff; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        .grid-card.selected { border-color: #ffc107; box-shadow: 0 0 10px rgba(255,193,7,0.3); }
        .grid-card-img {
            position: relative;
            background: #000;
            width: 100%;
            aspect-ratio: 16/9;
            overflow: hidden;
        }
        .grid-card-img img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .grid-card-img .grid-marker {
            position: absolute;
            width: 20px; height: 20px;
            border: 2px solid #ff0040;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            box-shadow: 0 0 8px rgba(255,0,64,0.5);
        }
        .grid-card-img .grid-marker::before {
            content: '';
            position: absolute; top: 50%; left: 50%;
            width: 4px; height: 4px;
            background: #ff0040; border-radius: 50%;
            transform: translate(-50%, -50%);
        }
        .grid-card-img .grid-marker.drag-end { border-color: #00c8ff; box-shadow: 0 0 8px rgba(0,200,255,0.5); }
        .grid-card-img .grid-marker.drag-end::before { background: #00c8ff; }
        .grid-card-body { padding: 10px 14px; }
        .grid-card-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 6px;
        }
        .grid-card-step {
            color: #00d9ff; font-weight: bold; font-size: 0.95em;
        }
        .grid-card-action {
            background: #252542; padding: 3px 10px; border-radius: 10px;
            font-size: 0.82em; color: #ccc;
        }
        .grid-card-code {
            background: #0d1117; padding: 5px 8px; border-radius: 4px;
            font-family: monospace; font-size: 0.82em; color: #7ee787;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            margin-bottom: 6px;
        }
        .grid-card-justification {
            color: #ffc107; font-size: 0.82em;
            border-left: 2px solid #ffc107; padding-left: 8px;
            max-height: 48px; overflow: hidden; text-overflow: ellipsis;
        }
        .grid-card-badges { display: flex; gap: 4px; margin-top: 4px; }
        .grid-card-badges .badge {
            font-size: 0.65em; padding: 1px 5px; border-radius: 3px; font-weight: bold;
        }
        .grid-card-badges .badge-adjusted { background: #ff9800; color: #000; }
        .grid-card-badges .badge-deleted { background: #f44336; color: #fff; }
        .grid-page-nav {
            display: flex; align-items: center; justify-content: center;
            gap: 8px; margin-bottom: 12px;
        }
        .grid-page-btn {
            padding: 6px 16px; border: 1px solid #444;
            border-radius: 5px; background: #1a1a2e;
            color: #e0e0e0; cursor: pointer; transition: all 0.2s; font-size: 0.85em;
        }
        .grid-page-btn:hover { border-color: #00d9ff; color: #00d9ff; }
        .grid-page-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .grid-page-info { color: #888; font-size: 0.85em; }

        /* Detail overlay for grid view */
        .detail-overlay {
            display: none;
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(15,15,26,0.92);
            z-index: 500;
            overflow-y: auto;
            padding: 20px;
        }
        .detail-overlay.show { display: block; }
        .detail-overlay-content {
            max-width: 1100px; margin: 0 auto;
            background: #1a1a2e; border-radius: 10px;
            border: 1px solid #333; overflow: hidden;
        }
        .detail-overlay-header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 16px; background: #16213e; border-bottom: 1px solid #333;
        }
        .detail-overlay-header h3 { color: #00d9ff; font-size: 1em; }
        .detail-overlay-close {
            padding: 4px 12px; background: #444; border: none; border-radius: 4px;
            color: #e0e0e0; cursor: pointer; font-size: 0.85em;
        }
        .detail-overlay-close:hover { background: #666; }
        .detail-overlay-body { padding: 16px; }

        .loading-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(15, 15, 26, 0.92);
            display: flex; justify-content: center; align-items: center;
            z-index: 1000; color: #00d9ff;
        }
        .loading-card {
            text-align: center; max-width: 420px; width: 90%;
        }
        .loading-card .load-title {
            font-size: 1.1em; margin-bottom: 14px; font-weight: 600;
        }
        .loading-card .load-bar-track {
            background: #252542; border-radius: 6px; height: 8px; overflow: hidden; margin-bottom: 8px;
        }
        .loading-card .load-bar-fill {
            height: 100%; background: linear-gradient(90deg, #00d9ff, #7c4dff); border-radius: 6px;
            transition: width 0.3s; width: 0%;
        }
        .loading-card .load-status {
            color: #888; font-size: 0.82em;
        }
        @keyframes loadPulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }

        .panel-review { display: block; }
        .panel-annotation { display: none; }
        body.annotation-mode .panel-review { display: none; }
        body.annotation-mode .panel-annotation { display: block; }

        /* Human-provided data highlight */
        .human-data-section {
            background: linear-gradient(135deg, #1a2e2a 0%, #16213e 100%);
            padding: 10px 12px;
            border-radius: 6px;
            margin-bottom: 10px;
            border: 1px solid #00897b;
        }
        .human-data-section h4 { color: #4db6ac; margin-bottom: 8px; font-size: 0.9em; }
        .human-data-section .data-field { margin-bottom: 6px; }
        .human-data-section .data-label { color: #888; font-size: 0.78em; display: block; margin-bottom: 2px; }
        .human-data-section .data-value {
            color: #e0e0e0; font-size: 0.85em;
            background: #0d1117; padding: 6px 8px; border-radius: 4px;
            white-space: pre-wrap; line-height: 1.4;
        }
        .human-data-section .kp-tag {
            display: inline-block; background: #00897b; color: white;
            padding: 2px 8px; border-radius: 10px;
            font-size: 0.78em; margin: 2px;
        }

        /* Editable justification */
        .justification-edit-area {
            margin-top: 6px;
            padding-top: 6px;
            border-top: 1px solid #333;
        }
        .justification-input {
            width: 100%; padding: 6px 8px;
            border: 1px solid #444; border-radius: 4px;
            background: #0d1117; color: #ffc107;
            resize: vertical; min-height: 40px;
            font-family: inherit; font-size: 0.85em;
            border-left: 3px solid #ffc107;
        }
        .justification-input:focus { border-color: #00d9ff; outline: none; }

        /* Editable query */
        .query-edit-input {
            padding: 6px 10px;
            border: 1px solid #444; border-radius: 4px;
            background: #0d1117; color: #ffc107;
            font-family: inherit; font-size: 0.9em;
            resize: vertical; min-height: 60px; width: 100%;
        }
        .query-edit-input:focus { border-color: #00d9ff; outline: none; }
        .info-item-query { flex: 2; min-width: 400px; }

        /* Delete step button */
        .delete-step-btn {
            padding: 2px 10px;
            background: transparent;
            border: 1px solid #f4433688;
            border-radius: 4px;
            color: #f44336;
            cursor: pointer;
            font-size: 0.7em;
            font-weight: bold;
            margin-left: 8px;
            transition: all 0.2s;
        }
        .delete-step-btn:hover { background: #f44336; color: #fff; }

        /* Export button */
        .export-case-btn {
            padding: 5px 14px;
            background: linear-gradient(135deg, #4caf50, #2e7d32);
            border: none; border-radius: 5px;
            color: white; cursor: pointer;
            font-weight: bold; font-size: 0.8em;
            transition: all 0.2s;
        }
        .export-case-btn:hover { transform: translateY(-1px); box-shadow: 0 3px 10px rgba(76,175,80,0.3); }

        /* Step error marks */
        .btn-mark-error {
            padding: 3px 10px;
            border: 1px solid #f44336;
            border-radius: 4px;
            background: transparent;
            color: #f44336;
            cursor: pointer;
            font-size: 0.75em;
            font-weight: bold;
            transition: all 0.2s;
            margin-left: 8px;
        }
        .btn-mark-error:hover { background: #f44336; color: #fff; }
        .btn-mark-error.marked {
            background: #f44336;
            color: #fff;
        }
        .btn-mark-error.marked:hover {
            background: transparent;
            color: #f44336;
        }
        .step-error-banner {
            background: linear-gradient(135deg, #f4433620, #ff980020);
            border: 1px solid #f44336;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .step-error-banner .error-icon {
            color: #f44336;
            font-size: 1.2em;
            flex-shrink: 0;
        }
        .step-error-banner .error-text {
            color: #ffab91;
            font-size: 0.85em;
        }
        .step-error-note-input {
            width: 100%;
            background: #1a1a2e;
            border: 1px solid #f4433660;
            border-radius: 4px;
            color: #ffab91;
            font-size: 0.8em;
            padding: 4px 8px;
            margin-top: 4px;
        }
        .step-error-note-input:focus {
            outline: none;
            border-color: #f44336;
        }
        .review-progress-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 10px;
            background: #16213e;
            border-radius: 4px;
            font-size: 0.8em;
        }
        .review-progress-bar .progress-text {
            color: #888;
            white-space: nowrap;
        }
        .review-progress-bar .progress-text b {
            color: #f44336;
        }
        .review-progress-bar .progress-track {
            flex: 1;
            height: 4px;
            background: #333;
            border-radius: 2px;
            min-width: 60px;
        }
        .review-progress-bar .progress-fill {
            height: 100%;
            background: #4caf50;
            border-radius: 2px;
            transition: width 0.3s;
        }
        .stacked-step-card.has-error {
            border-left: 3px solid #f44336 !important;
        }
        .step-item.has-error {
            border-left: 3px solid #f44336;
        }

        /* Frame adjustment */
        .frame-adjust-controls {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 3px 0;
            margin-top: 2px;
        }
        .frame-btn {
            padding: 2px 8px;
            border: 1px solid #555;
            border-radius: 4px;
            background: #1a1a2e;
            color: #00d9ff;
            cursor: pointer;
            font-size: 0.75em;
            transition: all 0.2s;
        }
        .frame-btn:hover { background: #00d9ff; color: #000; }
        .frame-time {
            color: #888;
            font-size: 0.72em;
            min-width: 70px;
            text-align: center;
            font-family: monospace;
        }

        /* Version grouping in sidebar */
        .version-toggle {
            padding: 2px 12px 2px 28px;
            color: #666;
            font-size: 0.7em;
            cursor: pointer;
            user-select: none;
        }
        .version-toggle:hover { color: #00d9ff; }
        .rec-item.older-version {
            opacity: 0.55;
            font-size: 0.85em;
            border-left: 2px solid #333;
            margin-left: 8px;
        }

        /* AI Check */
        .btn-ai-check {
            padding: 5px 14px;
            border: 1px solid #9c27b0;
            border-radius: 4px;
            background: transparent;
            color: #ce93d8;
            cursor: pointer;
            font-size: 0.8em;
            font-weight: bold;
            transition: all 0.2s;
        }
        .btn-ai-check:hover { background: #9c27b0; color: #fff; }
        .btn-ai-check.running {
            background: #9c27b0; color: #fff;
            animation: aipulse 1.5s infinite;
            cursor: wait;
        }
        @keyframes aipulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .ai-progress-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.65);
            z-index: 10000;
            justify-content: center;
            align-items: center;
        }
        .ai-progress-overlay.show {
            display: flex;
        }
        .ai-progress-card {
            background: #1a1a2e;
            border: 1px solid #9c27b0;
            border-radius: 16px;
            padding: 36px 48px;
            min-width: 420px;
            text-align: center;
            box-shadow: 0 0 40px rgba(156,39,176,0.3);
        }
        .ai-progress-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #ce93d8;
            margin-bottom: 8px;
        }
        .ai-progress-subtitle {
            font-size: 0.85em;
            color: #888;
            margin-bottom: 20px;
        }
        .ai-progress-bar-track {
            height: 12px;
            background: #2a2a3e;
            border-radius: 6px;
            overflow: hidden;
            position: relative;
            margin-bottom: 14px;
        }
        .ai-progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #7b1fa2, #ce93d8, #9c27b0);
            background-size: 200% 100%;
            animation: aiBarShimmer 2s linear infinite;
            transition: width 0.5s ease;
            border-radius: 6px;
            min-width: 2%;
        }
        @keyframes aiBarShimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        .ai-progress-stats {
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            color: #aaa;
            margin-bottom: 6px;
        }
        .ai-progress-pct {
            font-size: 2.2em;
            font-weight: bold;
            color: #e1bee7;
            margin-bottom: 4px;
        }
        .ai-progress-eta {
            font-size: 0.8em;
            color: #777;
            margin-top: 2px;
        }
        .ai-progress-issues {
            margin-top: 14px;
            padding-top: 12px;
            border-top: 1px solid #333;
            font-size: 0.82em;
            color: #888;
        }
        .ai-progress-issues span.issue-count {
            color: #f44336;
            font-weight: bold;
        }
        .ai-progress-issues span.ok-count {
            color: #4caf50;
            font-weight: bold;
        }
        .ai-progress-spinner {
            display: inline-block;
            width: 18px; height: 18px;
            border: 2px solid #555;
            border-top-color: #ce93d8;
            border-radius: 50%;
            animation: aispin 0.8s linear infinite;
            vertical-align: middle;
            margin-right: 6px;
        }
        @keyframes aispin {
            to { transform: rotate(360deg); }
        }
        .ai-result {
            margin-top: 6px;
            padding: 8px 10px;
            background: #151530;
            border-radius: 5px;
            border: 1px solid #333;
        }
        .ai-badges { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; align-items: center; }
        .ai-badge {
            padding: 2px 8px; border-radius: 10px;
            font-size: 0.7em; font-weight: bold;
            display: inline-block;
        }
        .ai-badge.ai-correct { background: rgba(76,175,80,0.2); color: #4caf50; }
        .ai-badge.ai-wrong { background: rgba(244,67,54,0.25); color: #f44336; }
        .ai-badge.ai-redundant { background: rgba(255,152,0,0.2); color: #ff9800; }
        .ai-badge.ai-suspicious { background: rgba(255,193,7,0.2); color: #ffc107; }
        .ai-badge.ai-unknown { background: rgba(150,150,150,0.15); color: #888; }
        .ai-badge.ai-good { background: rgba(76,175,80,0.15); color: #81c784; }
        .ai-badge.ai-acceptable { background: rgba(0,217,255,0.12); color: #4dd0e1; }
        .ai-badge.ai-poor { background: rgba(244,67,54,0.15); color: #ef9a9a; }
        .ai-badge.ai-missing { background: rgba(244,67,54,0.2); color: #f44336; }
        .ai-flag {
            padding: 1px 6px; border-radius: 8px;
            font-size: 0.65em; font-weight: bold;
            background: rgba(156,39,176,0.2); color: #ce93d8;
        }
        .ai-reason {
            color: #bbb; font-size: 0.78em; margin-bottom: 4px;
            padding-left: 8px; border-left: 2px solid #444;
        }
        .ai-reasoning {
            color: #90caf9; font-size: 0.78em; margin-bottom: 6px;
            padding: 6px 8px; background: #0d1530; border-radius: 4px;
            border-left: 2px solid #42a5f5; line-height: 1.5;
        }
        .ai-issues {
            font-size: 0.75em; color: #ef9a9a; margin-bottom: 4px;
        }
        .ai-issue { margin-bottom: 2px; }
        .ai-rewrite {
            background: #1a2e1a; border: 1px solid #2e7d32;
            border-radius: 4px; padding: 6px 8px; margin-top: 4px;
        }
        .ai-rewrite label {
            color: #66bb6a; font-size: 0.72em; font-weight: bold;
            display: block; margin-bottom: 3px;
        }
        .ai-rewrite-text {
            color: #a5d6a7; font-size: 0.82em; line-height: 1.4;
            margin-bottom: 4px;
        }
        .ai-apply-btn {
            padding: 2px 10px; background: #2e7d32; color: #fff;
            border: none; border-radius: 3px; cursor: pointer;
            font-size: 0.72em; font-weight: bold;
        }
        .ai-apply-btn:hover { background: #4caf50; }
        /* Grid card AI indicators */
        .grid-card.ai-wrong { border-color: #f44336; }
        .grid-card.ai-redundant { border-color: #ff9800; }
        .grid-card.ai-suspicious { border-color: #ffc107; }
        .grid-card.ai-correct { border-color: #333; }
        .grid-card.has-error { border-color: #f44336; box-shadow: 0 0 8px rgba(244,67,54,0.3); }

        /* Reset button */
        .btn-reset {
            padding: 5px 14px;
            border: 1px solid #ff980088;
            border-radius: 4px;
            background: transparent;
            color: #ff9800;
            cursor: pointer;
            font-size: 0.8em;
            font-weight: bold;
            transition: all 0.2s;
        }
        .btn-reset:hover { background: #ff9800; color: #000; }
        .btn-reset.confirming { background: #f44336; color: #fff; border-color: #f44336; }
    </style>
</head>
<body>
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-card">
            <div class="load-title" id="loadTitle">Loading recording data...</div>
            <div class="load-bar-track"><div class="load-bar-fill" id="loadBar" style="animation:loadPulse 1.5s infinite;width:5%"></div></div>
            <div class="load-status" id="loadStatus">Connecting...</div>
        </div>
    </div>

    <div class="header">
        <h1>OSS Review</h1>
        <div class="header-info">
            <span>Task: <b id="taskName"></b></span>
            <span>Status: <b id="reviewStatus">-</b></span>
        </div>
        <div class="mode-toggle">
            <button id="modeReview" class="active" onclick="setMode('review')">Review</button>
            <button id="modeAnnotation" onclick="setMode('annotation')">Annotate</button>
        </div>
        <div id="reviewProgressContainer" class="review-progress-bar" style="display:none;">
            <span class="progress-text" id="reviewProgressText">Errors: <b>0</b> / Reviewed: 0</span>
            <div class="progress-track"><div class="progress-fill" id="reviewProgressFill" style="width:0%"></div></div>
        </div>
        <div id="aiCheckContainer" style="display:flex;align-items:center;gap:6px;">
            <button class="btn-ai-check" id="aiCheckBtn" onclick="startAiCheck()">AI Check</button>
            <span id="aiCheckStatus" style="font-size:0.75em;color:#888;"></span>
        </div>
        <button class="btn-reset" onclick="confirmResetCase()">Reset to Original</button>
        <a class="btn-back" href="/dashboard" id="backLink">Dashboard</a>
    </div>

    <div class="main-layout">
        <!-- Recording sidebar -->
        <div class="rec-sidebar" id="recSidebar">
            <div class="rec-sidebar-header">
                <h3>Recordings</h3>
                <span class="rec-count" id="recCount">0</span>
            </div>
            <div class="rec-filter">
                <input type="text" id="recFilterInput" placeholder="Filter..." oninput="filterRecordings()" />
            </div>
            <div class="rec-list" id="recList">
                <div style="padding:20px;color:#666;text-align:center;font-size:0.85em;">Loading...</div>
            </div>
        </div>

        <!-- Step sidebar -->
        <div class="step-sidebar" id="stepSidebar">
            <div class="step-sidebar-header">Steps</div>
        </div>

        <!-- Content -->
        <div class="content-area" id="contentArea">
            <div class="annotator-info-bar" id="annotatorInfo"></div>
            <div id="humanDataSection"></div>

            <div class="nav-buttons">
                <button class="nav-btn" id="prevBtn" onclick="prevStep()" style="display:none;">Prev</button>
                <span class="step-counter" id="stepCounter">Step 0 / 0</span>
                <button class="nav-btn" id="nextBtn" onclick="nextStep()" style="display:none;">Next</button>
                <div class="view-toggle">
                    <button id="viewSingle" class="active" onclick="setViewMode('single')">Single</button>
                    <button id="viewGrid" onclick="setViewMode('grid')">Grid</button>
                </div>
                <select class="page-size-select" id="pageSizeSelect" onchange="changePageSize()" title="Steps per page in grid view">
                    <option value="10">10/page</option>
                    <option value="20">20/page</option>
                    <option value="30" selected>30/page</option>
                    <option value="50">50/page</option>
                    <option value="100">100/page</option>
                </select>
            </div>

            <!-- Single step view (all steps stacked) -->
            <div id="singleStepView">
                <div class="steps-stacked" id="stepsStacked"></div>
            </div>

            <!-- Grid view (all steps) -->
            <div id="gridStepView" style="display:none;">
                <div class="grid-page-nav" id="gridPageNav"></div>
                <div class="steps-grid" id="stepsGrid"></div>
                <div class="grid-page-nav" id="gridPageNavBottom"></div>
            </div>

            <div class="panel-review">
                <div class="review-panel">
                    <h4>Review Decision</h4>
                    <div class="review-buttons">
                        <button class="review-btn reviewed" id="btnReviewed" onclick="setReview('reviewed')">Reviewed (Pass)</button>
                        <button class="review-btn rejected" id="btnRejected" onclick="setReview('rejected')">Rejected</button>
                        <button class="review-btn unreviewed" id="btnUnreviewed" onclick="setReview('unreviewed')">Reset to Unreviewed</button>
                    </div>
                </div>
            </div>

            <div class="panel-annotation">
                <div class="annotation-panel" id="verdictPanel"></div>
                <div id="evaluationPanel"></div>
                <div id="knowledgePanel"></div>
            </div>
        </div>
    </div>

    <!-- AI Check progress overlay -->
    <div class="ai-progress-overlay" id="aiProgressOverlay">
        <div class="ai-progress-card">
            <div class="ai-progress-title"><span class="ai-progress-spinner"></span>AI Quality Check Running</div>
            <div class="ai-progress-subtitle" id="aiProgressSubtitle">Analyzing steps with Gemini...</div>
            <div class="ai-progress-pct" id="aiProgressPct">0%</div>
            <div class="ai-progress-bar-track">
                <div class="ai-progress-bar-fill" id="aiProgressFill" style="width:0%"></div>
            </div>
            <div class="ai-progress-stats">
                <span id="aiProgressSteps">Step 0 / 0</span>
                <span id="aiProgressEta">Estimating...</span>
            </div>
            <div class="ai-progress-issues" id="aiProgressIssues"></div>
        </div>
    </div>

    <!-- Detail overlay for grid card click -->
    <div class="detail-overlay" id="detailOverlay">
        <div class="detail-overlay-content">
            <div class="detail-overlay-header">
                <h3 id="detailOverlayTitle">Step Detail</h3>
                <div style="display:flex;gap:8px;align-items:center;">
                    <button class="grid-page-btn" onclick="detailPrev()" id="detailPrevBtn">Prev</button>
                    <button class="grid-page-btn" onclick="detailNext()" id="detailNextBtn">Next</button>
                    <button class="detail-overlay-close" onclick="closeDetail()">Close (Esc)</button>
                </div>
            </div>
            <div class="detail-overlay-body" id="detailOverlayBody"></div>
        </div>
    </div>

    <script>
        let folderName = '{{ folder_name }}';
        const ossFolder = new URLSearchParams(window.location.search).get('folder') || 'recordings_0303';
        const directMode = new URLSearchParams(window.location.search).get('direct') === '1';
        const pageMode = '{{ page_mode | default("reviewer") }}';  // 'reviewer' or 'annotator'
        const isAnnotatorMode = pageMode === 'annotator';
        let taskData = null;
        let currentStep = 0;
        let reviewStatus = 'unreviewed';
        let currentMode = 'review';
        let annotation = {};
        let coordAdjustments = {};
        let stepErrors = {};  // {origIdx: {note: '...'}} - reviewer-marked errors
        let annotatorStatus = 'not_started';
        let finetuneActive = false;
        let allRecordings = [];
        let filteredRecordings = [];
        let viewMode = 'single'; // 'single' or 'grid'
        let gridPage = 0;
        let gridPageSize = 30;
        let detailStepIdx = -1;

        document.getElementById('backLink').href = '/dashboard';

        // If direct access mode, hide recording sidebar and AI check
        if (directMode) {
            const recSidebar = document.getElementById('recSidebar');
            if (recSidebar) recSidebar.style.display = 'none';
            const aiContainer = document.getElementById('aiCheckContainer');
            if (aiContainer) aiContainer.style.display = 'none';
        }

        // Annotator mode: hide AI check, mode toggle, reset button; change header
        if (isAnnotatorMode) {
            // Annotators CAN use AI check (kept visible)
            document.querySelector('.mode-toggle').style.display = 'none';
            document.querySelector('.btn-reset').style.display = 'none';
            document.querySelector('.header h1').textContent = 'Annotator Review';
            document.querySelector('.header h1').style.color = '#ff9800';
            // Hide review decision panel for annotator
            document.querySelector('.panel-review').style.display = 'none';
            // Back link goes to annotator dashboard, NOT reviewer dashboard
            const backLink = document.getElementById('backLink');
            backLink.textContent = 'My Tasks';
            backLink.href = '/annotator_dashboard?folder=' + encodeURIComponent(ossFolder);
        }

        // ========== Recording sidebar ==========

        async function loadRecordingSidebar() {
            try {
                const resp = await fetch('/api/oss/folder_recordings?folder=' + encodeURIComponent(ossFolder));
                const data = await resp.json();
                allRecordings = data.recordings || [];
                filteredRecordings = allRecordings;
                document.getElementById('recCount').textContent = allRecordings.length;
                renderRecordingSidebar();
            } catch (err) {
                document.getElementById('recList').innerHTML = '<div style="padding:10px;color:#f44336;font-size:0.8em;">Failed to load</div>';
            }
        }

        // Track which person groups are expanded
        let expandedPersons = {};
        let expandedVersionGroups = {};

        function getTaskBase(taskId) {
            const match = taskId.match(/^(.+)_(\d{3})$/);
            if (match) return { base: match[1], version: parseInt(match[2]) };
            return { base: taskId, version: -1 };
        }

        function toggleVersionGroup(baseKey) {
            expandedVersionGroups[baseKey] = !expandedVersionGroups[baseKey];
            renderRecordingSidebar();
        }

        function hashColor(str) {
            let hash = 0;
            for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash);
            return 'hsl(' + (Math.abs(hash) % 360) + ', 55%, 42%)';
        }

        function renderRecordingSidebar() {
            const list = document.getElementById('recList');
            if (filteredRecordings.length === 0) {
                list.innerHTML = '<div style="padding:20px;color:#666;text-align:center;font-size:0.8em;">No recordings</div>';
                return;
            }

            // Group by username
            const groups = {};
            filteredRecordings.forEach(rec => {
                const user = rec.username || 'Unknown';
                if (!groups[user]) groups[user] = [];
                groups[user].push(rec);
            });

            // Auto-expand the group containing current recording
            const currentUser = filteredRecordings.find(r => r.folder_name === folderName);
            if (currentUser) {
                const uname = currentUser.username || 'Unknown';
                if (expandedPersons[uname] === undefined) expandedPersons[uname] = true;
            }

            const sortedUsers = Object.keys(groups).sort();
            let html = '';
            sortedUsers.forEach(user => {
                const recs = groups[user];
                const rev = recs.filter(r => r.review_status === 'reviewed').length;
                const rej = recs.filter(r => r.review_status === 'rejected').length;
                const isExpanded = !!expandedPersons[user];
                const color = hashColor(user);
                const letter = user.charAt(0).toUpperCase();
                const safeUser = user.replace(/'/g, "\\\\'");

                html += '<div class="person-group">';
                html += '<div class="person-header" onclick="togglePerson(\\'' + safeUser + '\\')">';
                html += '<span class="person-toggle' + (isExpanded ? ' expanded' : '') + '" id="ptoggle-' + user + '">&#9654;</span>';
                html += '<span class="person-avatar" style="background:' + color + '">' + letter + '</span>';
                html += '<span class="person-name">' + user + '</span>';
                html += '<span class="person-stats">';
                if (rev > 0) html += '<span class="person-stat rev">' + rev + '</span>';
                if (rej > 0) html += '<span class="person-stat rej">' + rej + '</span>';
                html += '<span class="person-stat total">' + recs.length + '</span>';
                html += '</span></div>';

                html += '<div class="person-tasks' + (isExpanded ? ' expanded' : '') + '" id="ptasks-' + user + '">';

                // Group by task base name for versioning
                const versionGroups = {};
                recs.forEach(rec => {
                    const tid = rec.task_name || rec.task_id || rec.folder_name;
                    const parsed = getTaskBase(tid);
                    if (!versionGroups[parsed.base]) versionGroups[parsed.base] = [];
                    versionGroups[parsed.base].push({ rec: rec, version: parsed.version });
                });

                Object.keys(versionGroups).sort().forEach(base => {
                    const versions = versionGroups[base];
                    versions.sort((a, b) => b.version - a.version); // newest first
                    const newest = versions[0];
                    const hasOlder = versions.length > 1;

                    // Render newest version
                    const nRec = newest.rec;
                    const nIsActive = nRec.folder_name === folderName;
                    const nStatus = nRec.review_status || 'unreviewed';
                    const nStatusLabel = nStatus.charAt(0).toUpperCase() + nStatus.slice(1);
                    const nTitle = nRec.task_name || nRec.task_id || nRec.folder_name.substring(0, 30);
                    html += '<div class="rec-item ' + nStatus + (nIsActive ? ' active' : '') + '" ' +
                        'onclick="switchRecording(\\'' + nRec.folder_name.replace(/'/g, "\\\\'") + '\\')">' +
                        '<div class="rec-title" title="' + nRec.folder_name + '">' + nTitle + '</div>' +
                        '<span class="rec-status ' + nStatus + '">' + nStatusLabel + '</span>' +
                        '</div>';

                    // Older versions toggle
                    if (hasOlder) {
                        const safeBase = base.replace(/'/g, "\\\\'");
                        const vExpanded = !!expandedVersionGroups[base];
                        html += '<div class="version-toggle" onclick="event.stopPropagation();toggleVersionGroup(\\'' + safeBase + '\\')">';
                        html += (vExpanded ? '&#9660; ' : '&#9654; ') + (versions.length - 1) + ' older version(s)';
                        html += '</div>';
                        if (vExpanded) {
                            versions.slice(1).forEach(v => {
                                const oRec = v.rec;
                                const oIsActive = oRec.folder_name === folderName;
                                const oStatus = oRec.review_status || 'unreviewed';
                                const oStatusLabel = oStatus.charAt(0).toUpperCase() + oStatus.slice(1);
                                const oTitle = oRec.task_name || oRec.task_id || oRec.folder_name.substring(0, 30);
                                html += '<div class="rec-item older-version ' + oStatus + (oIsActive ? ' active' : '') + '" ' +
                                    'onclick="switchRecording(\\'' + oRec.folder_name.replace(/'/g, "\\\\'") + '\\')">' +
                                    '<div class="rec-title" title="' + oRec.folder_name + '">' + oTitle + '</div>' +
                                    '<span class="rec-status ' + oStatus + '">' + oStatusLabel + '</span>' +
                                    '</div>';
                            });
                        }
                    }
                });
                html += '</div></div>';
            });

            list.innerHTML = html;

            // Scroll active item into view
            setTimeout(() => {
                const active = list.querySelector('.rec-item.active');
                if (active) active.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            }, 100);
        }

        function togglePerson(user) {
            expandedPersons[user] = !expandedPersons[user];
            const tasks = document.getElementById('ptasks-' + user);
            const toggle = document.getElementById('ptoggle-' + user);
            if (tasks) tasks.classList.toggle('expanded');
            if (toggle) toggle.classList.toggle('expanded');
        }

        function filterRecordings() {
            const q = document.getElementById('recFilterInput').value.toLowerCase();
            if (!q) {
                filteredRecordings = allRecordings;
            } else {
                filteredRecordings = allRecordings.filter(r =>
                    (r.folder_name || '').toLowerCase().includes(q) ||
                    (r.task_id || '').toLowerCase().includes(q) ||
                    (r.username || '').toLowerCase().includes(q) ||
                    (r.query || '').toLowerCase().includes(q)
                );
            }
            renderRecordingSidebar();
        }

        function switchRecording(newFolderName) {
            if (newFolderName === folderName) return;
            folderName = newFolderName;
            // Update URL without reload
            const baseRoute = isAnnotatorMode ? '/annotator/' : '/oss_review/';
            const url = baseRoute + encodeURIComponent(folderName) + '?folder=' + encodeURIComponent(ossFolder);
            history.pushState(null, '', url);
            document.title = (isAnnotatorMode ? 'Annotator Review - ' : 'OSS Review - ') + folderName;
            // Reset and reload
            taskData = null;
            currentStep = 0;
            finetuneActive = false;
            annotation = {};
            coordAdjustments = {};
            stepErrors = {};
            annotatorStatus = 'not_started';
            // Reset AI check state
            if (aiCheckPolling) { clearInterval(aiCheckPolling); aiCheckPolling = null; }
            aiCheckResults = null;
            aiCheckStartTime = null;
            showAiProgress(false);
            const aiBtn = document.getElementById('aiCheckBtn');
            aiBtn.classList.remove('running');
            aiBtn.textContent = 'AI Check';
            document.getElementById('aiCheckStatus').textContent = '';
            document.getElementById('aiCheckStatus').style.color = '#888';
            document.getElementById('loadingOverlay').style.display = 'flex';
            setLoadProgress(10, 'Switching recording...');
            loadTask();
            renderRecordingSidebar();
        }

        function setLoadProgress(pct, msg) {
            var bar = document.getElementById('loadBar');
            var status = document.getElementById('loadStatus');
            if (bar) { bar.style.width = pct + '%'; bar.style.animation = pct >= 100 ? 'none' : 'loadPulse 1.5s infinite'; }
            if (status) status.textContent = msg || '';
        }

        // ========== Mode toggle ==========

        function setMode(mode) {
            currentMode = mode;
            document.body.classList.toggle('annotation-mode', mode === 'annotation');
            document.getElementById('modeReview').classList.toggle('active', mode === 'review');
            document.getElementById('modeAnnotation').classList.toggle('active', mode === 'annotation');
            if (taskData) renderAnnotationPanels();
        }

        // ========== Load task ==========

        async function loadTask() {
            try {
                setLoadProgress(15, 'Fetching recording data from server...');
                const resp = await fetch('/api/oss/task/' + encodeURIComponent(folderName) + '?folder=' + encodeURIComponent(ossFolder));
                setLoadProgress(50, 'Parsing response...');
                taskData = await resp.json();

                if (taskData.error) {
                    document.getElementById('contentArea').innerHTML =
                        '<div style="padding:40px;color:#f44336;">' + taskData.error + '</div>';
                    document.getElementById('loadingOverlay').style.display = 'none';
                    return;
                }
                setLoadProgress(65, 'Loading ' + (taskData.steps || []).length + ' steps...');

                annotation = taskData.annotation || {};
                coordAdjustments = taskData.coord_adjustments || {};
                stepErrors = taskData.step_errors || {};
                annotatorStatus = taskData.annotator_status || 'not_started';
                // Load AI check results if available
                if (annotation.ai_check_results && annotation.ai_check_results.status === 'completed') {
                    aiCheckResults = annotation.ai_check_results;
                    onAiCheckComplete();
                }

                // Pre-populate annotation fields from annotator data if annotation is empty
                const info = taskData.annotator_info || {};
                if (!annotation.step_by_step_instructions && info.step_by_step_instruction) {
                    annotation.step_by_step_instructions = info.step_by_step_instruction;
                }
                const humanKP = taskData.knowledge_points || [];
                if (humanKP.length > 0 && (!annotation.custom_nodes || annotation.custom_nodes.length === 0)) {
                    annotation.custom_nodes = [...humanKP];
                }

                document.getElementById('taskName').textContent = taskData.task_name || folderName;

                // Annotator info bar with editable query
                let infoHtml = '';
                if (info.username) infoHtml += '<div class="info-item"><span>Annotator:</span> <b>' + info.username + '</b></div>';
                if (info.task_id) infoHtml += '<div class="info-item"><span>Task:</span> <b>' + info.task_id + '</b></div>';
                infoHtml += '<div class="info-item info-item-query"><span>Query:</span> ' +
                    '<textarea class="query-edit-input" id="query-edit" onchange="saveQuery(this.value)" placeholder="Enter query...">' +
                    (info.query || '') + '</textarea></div>';
                infoHtml += '<div class="info-item"><button class="export-case-btn" onclick="exportCase()">Export Case</button></div>';
                // Annotator status
                if (isAnnotatorMode) {
                    infoHtml += '<div class="info-item" style="display:flex;gap:6px;align-items:center;"><span style="color:#888;font-size:0.8em;">Status:</span>';
                    infoHtml += '<button id="annBtnIP" style="padding:3px 10px;border:1px solid #ff9800;border-radius:4px;background:' + (annotatorStatus==='in_progress'?'#ff9800':'transparent') + ';color:' + (annotatorStatus==='in_progress'?'#000':'#ff9800') + ';cursor:pointer;font-size:0.78em;font-weight:bold;" onclick="setAnnotatorStatus(&quot;in_progress&quot;)">In Progress</button>';
                    infoHtml += '<button id="annBtnDone" style="padding:3px 10px;border:1px solid #4caf50;border-radius:4px;background:' + (annotatorStatus==='completed'?'#4caf50':'transparent') + ';color:' + (annotatorStatus==='completed'?'#fff':'#4caf50') + ';cursor:pointer;font-size:0.78em;font-weight:bold;" onclick="setAnnotatorStatus(&quot;completed&quot;)">Completed</button></div>';
                } else {
                    var _asc = {not_started:'#888',in_progress:'#ff9800',completed:'#4caf50'};
                    var _asl = {not_started:'Not Started',in_progress:'In Progress',completed:'Completed'};
                    infoHtml += '<div class="info-item"><span style="color:#888;font-size:0.8em;">Annotator:</span> <b style="color:' + (_asc[annotatorStatus]||'#888') + ';font-size:0.85em;">' + (_asl[annotatorStatus]||'Not Started') + '</b></div>';
                }
                // Show error count for annotator
                const errorCount = Object.keys(stepErrors).length;
                if (isAnnotatorMode && errorCount > 0) {
                    infoHtml += '<div class="info-item" style="color:#f44336;font-weight:bold;">&#9888; ' + errorCount + ' step(s) marked as errors by reviewer</div>';
                }
                // Deleted steps restore
                if (taskData.deleted_step_count > 0) {
                    infoHtml += '<div class="info-item" style="color:#ff9800;font-size:0.85em;">' + taskData.deleted_step_count + ' deleted step(s) <button style="padding:2px 8px;border:1px solid #ff9800;border-radius:3px;background:transparent;color:#ff9800;cursor:pointer;font-size:0.8em;" onclick="restoreDeletedSteps()">Restore All</button></div>';
                }
                document.getElementById('annotatorInfo').innerHTML = infoHtml || '<div class="info-item"><span>No annotator info</span></div>';

                // Human-provided data section
                let hdHtml = '';
                if (info.step_by_step_instruction || humanKP.length > 0) {
                    hdHtml = '<div class="human-data-section"><h4>Human-Provided Data</h4>';
                    if (info.step_by_step_instruction) {
                        hdHtml += '<div class="data-field"><span class="data-label">Step-by-Step Instruction (from annotator)</span>' +
                            '<div class="data-value">' + info.step_by_step_instruction.replace(/\\\\n/g, '\\n').replace(/\\n/g, '<br>') + '</div></div>';
                    }
                    if (humanKP.length > 0) {
                        hdHtml += '<div class="data-field"><span class="data-label">Knowledge Points (from annotator)</span><div>';
                        humanKP.forEach(kp => { hdHtml += '<span class="kp-tag">' + kp + '</span>'; });
                        hdHtml += '</div></div>';
                    }
                    hdHtml += '</div>';
                }
                document.getElementById('humanDataSection').innerHTML = hdHtml;

                reviewStatus = taskData.review_status || 'unreviewed';
                updateReviewUI();
                updateReviewProgress();

                setLoadProgress(80, 'Rendering step sidebar...');
                renderStepSidebar();
                setLoadProgress(90, 'Rendering ' + (taskData.steps || []).length + ' steps...');
                renderStep(0);
                if (viewMode === 'grid') renderGrid();

                setLoadProgress(100, 'Done!');
                document.getElementById('loadingOverlay').style.display = 'none';
            } catch (err) {
                document.getElementById('contentArea').innerHTML =
                    '<div style="padding:40px;color:#f44336;">Failed to load: ' + err.message + '</div>';
                document.getElementById('loadingOverlay').style.display = 'none';
            }
        }

        // ========== Step sidebar ==========

        function renderStepSidebar() {
            const steps = taskData.steps || [];
            let html = '<div class="step-sidebar-header">Steps (' + steps.length + ')</div>';
            steps.forEach((step, i) => {
                const origIdx = step.original_index != null ? step.original_index : i;
                const aiR = getAiStepResult(origIdx);
                const hasError = !!stepErrors[String(origIdx)];
                let aiDot = '';
                if (aiR) {
                    const dotColor = {correct:'#4caf50', wrong:'#f44336', redundant:'#ff9800', suspicious:'#ffc107'}[aiR.correctness] || '#888';
                    const justDotColor = {good:'#4caf50', acceptable:'#8bc34a', poor:'#f44336', missing:'#ff9800'}[aiR.justification_quality] || '#888';
                    aiDot = '<span style="width:6px;height:6px;border-radius:50%;background:' + dotColor + ';display:inline-block;margin-left:4px;flex-shrink:0;" title="AI: ' + (aiR.correctness || 'unknown') + '"></span>';
                    aiDot += '<span style="width:6px;height:6px;border-radius:50%;background:' + justDotColor + ';display:inline-block;margin-left:2px;flex-shrink:0;" title="Justification: ' + (aiR.justification_quality || 'unknown') + '"></span>';
                }
                // Error mark indicator
                let errorDot = '';
                if (hasError) {
                    errorDot = '<span style="width:8px;height:8px;border-radius:50%;background:#f44336;display:inline-block;margin-left:3px;flex-shrink:0;box-shadow:0 0 4px #f44336;" title="Reviewer marked error"></span>';
                }
                html += '<div class="step-item' + (i === 0 ? ' active' : '') + (hasError ? ' has-error' : '') + '" onclick="selectStep(' + i + ')" id="step-' + i + '" style="display:flex;align-items:center;">' +
                    '<span class="step-num">' + i + '</span> ' +
                    '<span class="step-action" style="flex:1;">' + (step.action || '') + '</span>' +
                    errorDot +
                    aiDot +
                    '</div>';
            });
            document.getElementById('stepSidebar').innerHTML = html;
        }

        function selectStep(idx) {
            currentStep = idx;
            document.getElementById('stepCounter').textContent = 'Step ' + idx + ' / ' + ((taskData.steps || []).length - 1);
            document.querySelectorAll('.step-item').forEach((el, i) => {
                el.classList.toggle('active', i === idx);
            });
            if (viewMode === 'single') {
                // Scroll to the step in stacked view
                const card = document.getElementById('stacked-step-' + idx);
                if (card) {
                    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    document.querySelectorAll('.stacked-step-card').forEach(c => c.classList.remove('active'));
                    card.classList.add('active');
                }
            }
        }

        // ========== Render step (stacked all steps) ==========

        function renderStep(idx) {
            // Render ALL steps in stacked view
            renderAllStepsStacked();
            // Scroll to the specified step
            currentStep = idx;
            document.getElementById('stepCounter').textContent = 'Step ' + idx + ' / ' + ((taskData.steps || []).length - 1);
            const targetCard = document.getElementById('stacked-step-' + idx);
            if (targetCard) {
                targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                document.querySelectorAll('.stacked-step-card').forEach(c => c.classList.remove('active'));
                targetCard.classList.add('active');
            }
            renderAnnotationPanels();
        }

        function renderAllStepsStacked() {
            const steps = taskData.steps || [];
            const videoWidth = taskData.video_width || 1920;
            const videoHeight = taskData.video_height || 1080;
            const container = document.getElementById('stepsStacked');
            if (!container) return;

            let html = '';
            steps.forEach((step, idx) => {
                const origIdx = step.original_index != null ? step.original_index : idx;
                const adj = coordAdjustments[String(origIdx)];
                const isAdjusted = !!adj;
                const origCoord = isAdjusted ? (adj.original || step.coordinate) : step.coordinate;
                const imgUrl = '/oss_frame/' + encodeURIComponent(folderName) + '/' + parseFloat(step.video_time).toFixed(1) + '?folder=' + encodeURIComponent(ossFolder);

                const hasStepError = !!stepErrors[String(origIdx)];
                html += '<div class="stacked-step-card' + (hasStepError ? ' has-error' : '') + '" id="stacked-step-' + idx + '" onclick="onStackedStepClick(' + idx + ')">';

                // Header
                html += '<div class="stacked-step-header"><h4>Step ' + idx + ': ' + (step.action || '');
                if (isAdjusted) html += ' <span class="adjusted-badge">Adjusted</span>';
                html += '</h4>';
                // Mark Error button (reviewer only) or Delete button
                if (!isAnnotatorMode) {
                    html += '<button class="btn-mark-error' + (hasStepError ? ' marked' : '') + '" onclick="event.stopPropagation();toggleStepError(' + idx + ')">' + (hasStepError ? 'Unmark Error' : 'Mark Error') + '</button>';
                }
                html += '<button class="delete-step-btn" onclick="event.stopPropagation();confirmDeleteStep(' + idx + ')" title="Delete this step">Delete</button>';
                html += '</div>';

                // Error banner (shown for both reviewer and annotator)
                if (hasStepError) {
                    const errNote = (stepErrors[String(origIdx)] && stepErrors[String(origIdx)].note) || '';
                    html += '<div class="step-error-banner"><span class="error-icon">&#9888;</span><div>';
                    html += '<span class="error-text">Marked as error by reviewer</span>';
                    if (errNote) html += '<div style="color:#ffab91;font-size:0.8em;margin-top:2px;">' + errNote + '</div>';
                    if (!isAnnotatorMode) {
                        html += '<input class="step-error-note-input" placeholder="Note for annotator..." value="' + errNote.replace(/"/g, '&quot;') + '" onclick="event.stopPropagation()" onchange="saveStepErrorNote(' + idx + ', this.value)" />';
                    }
                    html += '</div></div>';
                }

                // Resolution info
                html += '<div class="image-info" id="stacked-info-' + idx + '" style="margin-bottom:4px;"><div class="resolution-info">Res: <b>' + videoWidth + ' x ' + videoHeight + '</b></div></div>';

                // Screenshot with markers + drag lines
                html += '<div class="stacked-img-container" id="stacked-img-' + idx + '" onclick="event.stopPropagation();handleStackedImageClick(event,' + idx + ')">';
                html += '<img id="stacked-img-el-' + idx + '" src="' + imgUrl + '" data-idx="' + idx + '" />';
                html += '<svg class="drag-line" id="stacked-drag-' + idx + '" style="display:none;position:absolute;top:0;left:0;width:100%;height:100%;"><line id="stacked-dragline-' + idx + '" /><polygon id="stacked-dragarrow-' + idx + '" /></svg>';
                html += '</div>';

                // Frame adjustment controls
                html += '<div class="frame-adjust-controls">';
                html += '<button class="frame-btn" onclick="event.stopPropagation();shiftFrame(' + idx + ',-0.8)" title="-0.8s">&#9664;&#9664;</button>';
                html += '<button class="frame-btn" onclick="event.stopPropagation();shiftFrame(' + idx + ',-0.2)" title="-0.2s">&#9664;</button>';
                html += '<span class="frame-time" id="frame-time-' + idx + '">t=' + step.video_time.toFixed(2) + 's</span>';
                html += '<button class="frame-btn" onclick="event.stopPropagation();shiftFrame(' + idx + ',0.2)" title="+0.2s">&#9654;</button>';
                html += '<button class="frame-btn" onclick="event.stopPropagation();shiftFrame(' + idx + ',0.8)" title="+0.8s">&#9654;&#9654;</button>';
                html += '</div>';

                // Code - editable for type/press operations
                if (step.action === 'type' || step.action === 'press') {
                    html += '<div class="code code-editable"><input type="text" class="code-edit-input" value="' + (step.code || '').replace(/"/g, '&quot;') + '" onchange="saveCode(' + idx + ', this.value)" title="Edit code" /></div>';
                } else {
                    html += '<div class="code">' + (step.code || '') + '</div>';
                }

                // Coordinate with fine-tune
                if (step.has_coordinate) {
                    html += '<div style="margin-bottom:6px;font-size:0.85em;">';
                    if (isAdjusted) {
                        html += 'Coord: <span class="current-coord" id="stacked-coord-' + idx + '">(' + step.coordinate.x + ', ' + step.coordinate.y + ')</span>';
                        html += '<span class="original-coord">(' + origCoord.x + ', ' + origCoord.y + ')</span>';
                    } else {
                        html += 'Coord: <span id="stacked-coord-' + idx + '">(' + step.coordinate.x + ', ' + step.coordinate.y + ')</span>';
                    }
                    html += ' <button class="finetune-btn" onclick="event.stopPropagation();toggleStackedFinetune(' + idx + ')">Fine-tune</button></div>';
                    // Finetune panel (hidden by default)
                    html += '<div class="coord-adjust-panel" id="stacked-finetune-' + idx + '">';
                    html += '<h5>Coordinate Fine-tuning</h5>';
                    html += '<div class="coord-adjust-controls">';
                    if (step.action === 'drag' && step.drag_to) {
                        html += '<div style="margin-bottom:4px;color:#888;font-size:0.78em;">Start (From):</div>';
                    }
                    html += '<div class="coord-field"><label>X:</label><input type="number" class="coord-input" id="stacked-cx-' + idx + '" value="' + step.coordinate.x + '" min="0" max="' + videoWidth + '" oninput="updateStackedMarker(' + idx + ')"></div>';
                    html += '<div class="coord-field"><label>Y:</label><input type="number" class="coord-input" id="stacked-cy-' + idx + '" value="' + step.coordinate.y + '" min="0" max="' + videoHeight + '" oninput="updateStackedMarker(' + idx + ')"></div>';
                    if (step.action === 'drag' && step.drag_to) {
                        html += '</div><div class="coord-adjust-controls" style="margin-top:4px;">';
                        html += '<div style="margin-bottom:4px;color:#888;font-size:0.78em;">End (To):</div>';
                        html += '<div class="coord-field"><label>X2:</label><input type="number" class="coord-input" id="stacked-dx-' + idx + '" value="' + step.drag_to.x + '" min="0" max="' + videoWidth + '" oninput="updateStackedMarker(' + idx + ')"></div>';
                        html += '<div class="coord-field"><label>Y2:</label><input type="number" class="coord-input" id="stacked-dy-' + idx + '" value="' + step.drag_to.y + '" min="0" max="' + videoHeight + '" oninput="updateStackedMarker(' + idx + ')"></div>';
                    }
                    html += '<button class="finetune-btn save" onclick="event.stopPropagation();saveStackedCoordinate(' + idx + ')">Save</button>';
                    html += '<button class="finetune-btn cancel" onclick="event.stopPropagation();cancelStackedFinetune(' + idx + ')">Cancel</button>';
                    html += '<span style="color:#888;font-size:0.75em;margin-left:8px;">Click image to set</span>';
                    html += '</div></div>';
                }

                // Description
                if (step.description) html += '<div class="description">' + step.description + '</div>';

                // Justification
                html += '<div class="justification-edit-area">';
                html += '<label style="color:#888;font-size:0.78em;display:block;margin-bottom:3px;">Justification:</label>';
                html += '<textarea class="justification-input" onchange="saveJustification(' + idx + ', this.value)" placeholder="Step justification...">' + (step.justification || '') + '</textarea>';
                html += '</div>';

                // AI check
                html += renderAiResultHtml(origIdx);

                html += '</div>';
            });

            container.innerHTML = html;

            // Add coordinate markers after images load
            // Use requestAnimationFrame to ensure browser has done layout first
            requestAnimationFrame(() => {
                steps.forEach((step, idx) => {
                    if (!step.has_coordinate || !step.coordinate) return;
                    const imgEl = document.getElementById('stacked-img-el-' + idx);
                    if (!imgEl) return;
                    const addMarkers = function() {
                        // Guard: if image has no layout dimensions yet, defer again
                        if (imgEl.clientWidth === 0) {
                            requestAnimationFrame(() => addMarkersToContainer('stacked-img-' + idx, imgEl, step, videoWidth, videoHeight));
                            return;
                        }
                        addMarkersToContainer('stacked-img-' + idx, imgEl, step, videoWidth, videoHeight);
                    };
                    if (imgEl.complete && imgEl.naturalWidth > 0) addMarkers();
                    else imgEl.onload = addMarkers;
                });
            });
        }

        function addMarkersToContainer(containerId, imgEl, step, videoWidth, videoHeight) {
            const container = document.getElementById(containerId);
            if (!container) return;
            // If image has no dimensions yet, defer to next frame
            if (imgEl.clientWidth === 0 || imgEl.clientHeight === 0) {
                requestAnimationFrame(() => addMarkersToContainer(containerId, imgEl, step, videoWidth, videoHeight));
                return;
            }
            container.querySelectorAll('.coord-marker').forEach(m => m.remove());
            const scaleX = imgEl.clientWidth / videoWidth;
            const scaleY = imgEl.clientHeight / videoHeight;

            // Update resolution info with scale if info bar exists
            // containerId is like "stacked-img-3" -> info id is "stacked-info-3"
            const infoId = containerId.replace('stacked-img-', 'stacked-info-');
            const infoEl = document.getElementById(infoId);
            if (infoEl) {
                infoEl.innerHTML = '<div class="resolution-info">Res: <b>' + videoWidth + ' x ' + videoHeight + '</b> | Scale: <b>' + (scaleX * 100).toFixed(0) + '%</b></div>';
            }

            if (step.action === 'drag' && step.drag_to) {
                const sx = step.coordinate.x * scaleX;
                const sy = step.coordinate.y * scaleY;
                const ex = step.drag_to.x * scaleX;
                const ey = step.drag_to.y * scaleY;
                // Start marker
                const m1 = document.createElement('div');
                m1.className = 'coord-marker';
                m1.setAttribute('data-label', 'from (' + step.coordinate.x + ',' + step.coordinate.y + ')');
                m1.style.left = sx + 'px';
                m1.style.top = sy + 'px';
                m1.style.display = 'block';
                container.appendChild(m1);
                // End marker
                const m2 = document.createElement('div');
                m2.className = 'coord-marker drag-end';
                m2.setAttribute('data-label', 'to (' + step.drag_to.x + ',' + step.drag_to.y + ')');
                m2.style.left = ex + 'px';
                m2.style.top = ey + 'px';
                m2.style.display = 'block';
                container.appendChild(m2);
                // SVG drag line with arrow
                const dragId = containerId.replace('stacked-img-', 'stacked-drag-');
                const lineId = containerId.replace('stacked-img-', 'stacked-dragline-');
                const arrowId = containerId.replace('stacked-img-', 'stacked-dragarrow-');
                const dragSvg = document.getElementById(dragId);
                const linePath = document.getElementById(lineId);
                const arrow = document.getElementById(arrowId);
                if (dragSvg && linePath && arrow) {
                    linePath.setAttribute('x1', sx);
                    linePath.setAttribute('y1', sy);
                    linePath.setAttribute('x2', ex);
                    linePath.setAttribute('y2', ey);
                    const angle = Math.atan2(ey - sy, ex - sx);
                    const aLen = 10;
                    const ax1 = ex - aLen * Math.cos(angle - 0.4);
                    const ay1 = ey - aLen * Math.sin(angle - 0.4);
                    const ax2 = ex - aLen * Math.cos(angle + 0.4);
                    const ay2 = ey - aLen * Math.sin(angle + 0.4);
                    arrow.setAttribute('points', ex + ',' + ey + ' ' + ax1 + ',' + ay1 + ' ' + ax2 + ',' + ay2);
                    dragSvg.style.display = 'block';
                }
            } else if (step.action === 'scroll' && step.scroll_info) {
                // Scroll arrow
                const px = step.coordinate.x * scaleX;
                const py = step.coordinate.y * scaleY;
                const si = step.scroll_info;
                const arrowLen = 40;
                let endX = px, endY = py;
                let label = 'Scroll';
                if (si.dy > 0) { endY = py + arrowLen; label = 'Scroll Down x' + si.dy; }
                else if (si.dy < 0) { endY = py - arrowLen; label = 'Scroll Up x' + Math.abs(si.dy); }
                if (si.dx > 0) { endX = px + arrowLen; label = 'Scroll Right x' + si.dx; }
                else if (si.dx < 0) { endX = px - arrowLen; label = 'Scroll Left x' + Math.abs(si.dx); }
                // Draw SVG arrow for scroll
                const scrollSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                scrollSvg.setAttribute('class', 'drag-line');
                scrollSvg.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;';
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', px); line.setAttribute('y1', py);
                line.setAttribute('x2', endX); line.setAttribute('y2', endY);
                line.setAttribute('stroke', '#ff0040'); line.setAttribute('stroke-width', '2');
                line.setAttribute('marker-end', 'url(#scrollArrowHead)');
                // Arrowhead marker def
                const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
                marker.setAttribute('id', 'scrollArrowHead'); marker.setAttribute('markerWidth', '10');
                marker.setAttribute('markerHeight', '7'); marker.setAttribute('refX', '10');
                marker.setAttribute('refY', '3.5'); marker.setAttribute('orient', 'auto');
                const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
                polygon.setAttribute('fill', '#ff0040');
                marker.appendChild(polygon); defs.appendChild(marker); scrollSvg.appendChild(defs);
                scrollSvg.appendChild(line);
                // Label text
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', px + 15); text.setAttribute('y', py - 8);
                text.setAttribute('fill', '#ff0040'); text.setAttribute('font-size', '11');
                text.setAttribute('font-weight', 'bold');
                text.textContent = label;
                scrollSvg.appendChild(text);
                container.appendChild(scrollSvg);
                // Also show position marker
                const m = document.createElement('div');
                m.className = 'coord-marker';
                m.setAttribute('data-label', label + ' (' + step.coordinate.x + ',' + step.coordinate.y + ')');
                m.style.left = px + 'px'; m.style.top = py + 'px'; m.style.display = 'block';
                m.style.borderColor = '#ff9800'; // Orange for scroll
                container.appendChild(m);
                const dragId = containerId.replace('stacked-img-', 'stacked-drag-');
                const dragSvg = document.getElementById(dragId);
                if (dragSvg) dragSvg.style.display = 'none';
            } else {
                // Click actions (single, double, right, triple)
                const px = step.coordinate.x * scaleX;
                const py = step.coordinate.y * scaleY;
                const clickType = step.click_type || 'click';
                const m = document.createElement('div');
                m.className = 'coord-marker';
                let label = step.action + ' (' + step.coordinate.x + ',' + step.coordinate.y + ')';
                if (clickType === 'doubleClick') { label = 'dblclick (' + step.coordinate.x + ',' + step.coordinate.y + ')'; m.style.boxShadow = '0 0 8px rgba(255,0,64,0.5), 0 0 16px rgba(255,0,64,0.3)'; }
                else if (clickType === 'rightClick') { label = 'rightclick (' + step.coordinate.x + ',' + step.coordinate.y + ')'; m.style.borderStyle = 'dashed'; }
                else if (clickType === 'tripleClick') { label = 'triple (' + step.coordinate.x + ',' + step.coordinate.y + ')'; m.style.boxShadow = '0 0 8px rgba(255,0,64,0.5), 0 0 16px rgba(255,0,64,0.3), 0 0 24px rgba(255,0,64,0.2)'; }
                m.setAttribute('data-label', label);
                m.style.left = px + 'px'; m.style.top = py + 'px'; m.style.display = 'block';
                container.appendChild(m);
                // Add second ring for double/triple click
                if (clickType === 'doubleClick' || clickType === 'tripleClick') {
                    const m2 = document.createElement('div');
                    m2.className = 'coord-marker';
                    m2.style.left = px + 'px'; m2.style.top = py + 'px'; m2.style.display = 'block';
                    m2.style.width = '40px'; m2.style.height = '40px'; m2.style.opacity = '0.5';
                    m2.style.pointerEvents = 'none';
                    container.appendChild(m2);
                }
                if (clickType === 'tripleClick') {
                    const m3 = document.createElement('div');
                    m3.className = 'coord-marker';
                    m3.style.left = px + 'px'; m3.style.top = py + 'px'; m3.style.display = 'block';
                    m3.style.width = '52px'; m3.style.height = '52px'; m3.style.opacity = '0.3';
                    m3.style.pointerEvents = 'none';
                    container.appendChild(m3);
                }
                // Hide drag line
                const dragId = containerId.replace('stacked-img-', 'stacked-drag-');
                const dragSvg = document.getElementById(dragId);
                if (dragSvg) dragSvg.style.display = 'none';
            }
        }

        function onStackedStepClick(idx) {
            currentStep = idx;
            document.querySelectorAll('.stacked-step-card').forEach(c => c.classList.remove('active'));
            document.getElementById('stacked-step-' + idx).classList.add('active');
            document.querySelectorAll('.step-item').forEach((el, i) => el.classList.toggle('active', i === idx));
            document.getElementById('stepCounter').textContent = 'Step ' + idx + ' / ' + ((taskData.steps || []).length - 1);
        }

        // Stacked view coordinate fine-tuning
        let stackedFinetuneIdx = null;

        function toggleStackedFinetune(idx) {
            const panel = document.getElementById('stacked-finetune-' + idx);
            if (!panel) return;
            if (stackedFinetuneIdx === idx) {
                panel.classList.remove('show');
                stackedFinetuneIdx = null;
            } else {
                // Close any other open panel
                if (stackedFinetuneIdx !== null) {
                    const prev = document.getElementById('stacked-finetune-' + stackedFinetuneIdx);
                    if (prev) prev.classList.remove('show');
                }
                panel.classList.add('show');
                stackedFinetuneIdx = idx;
            }
        }

        function updateStackedMarker(idx) {
            const step = taskData.steps[idx];
            const xInput = document.getElementById('stacked-cx-' + idx);
            const yInput = document.getElementById('stacked-cy-' + idx);
            const imgEl = document.getElementById('stacked-img-el-' + idx);
            if (!xInput || !yInput || !imgEl) return;
            const x = parseInt(xInput.value) || 0;
            const y = parseInt(yInput.value) || 0;
            const videoWidth = taskData.video_width || 1920;
            const videoHeight = taskData.video_height || 1080;
            // Update marker with temp coordinate
            const tempStep = Object.assign({}, step, { coordinate: { x, y } });
            addMarkersToContainer('stacked-img-' + idx, imgEl, tempStep, videoWidth, videoHeight);
        }

        async function saveStackedCoordinate(idx) {
            const step = taskData.steps[idx];
            const xInput = document.getElementById('stacked-cx-' + idx);
            const yInput = document.getElementById('stacked-cy-' + idx);
            if (!xInput || !yInput) return;
            const x = parseInt(xInput.value) || 0;
            const y = parseInt(yInput.value) || 0;
            const origIdx = step.original_index != null ? step.original_index : idx;
            const adj = coordAdjustments[String(origIdx)];
            let origX, origY;
            if (adj && adj.original) { origX = adj.original.x; origY = adj.original.y; }
            else { origX = step.coordinate.x; origY = step.coordinate.y; }

            // Handle drag-end coordinates
            let dragToX = null, dragToY = null, origDragToX = null, origDragToY = null;
            if (step.action === 'drag' && step.drag_to) {
                const dxInput = document.getElementById('stacked-dx-' + idx);
                const dyInput = document.getElementById('stacked-dy-' + idx);
                if (dxInput && dyInput) {
                    dragToX = parseInt(dxInput.value) || 0;
                    dragToY = parseInt(dyInput.value) || 0;
                    if (adj && adj.original_drag_to) { origDragToX = adj.original_drag_to.x; origDragToY = adj.original_drag_to.y; }
                    else { origDragToX = step.drag_to.x; origDragToY = step.drag_to.y; }
                }
            }

            var body = { folder_name: folderName, oss_folder: ossFolder, step_index: origIdx, x: x, y: y, original_x: origX, original_y: origY };
            if (dragToX !== null) {
                body.drag_to_x = dragToX;
                body.drag_to_y = dragToY;
                body.original_drag_to_x = origDragToX;
                body.original_drag_to_y = origDragToY;
            }

            await fetch('/api/oss/update_coordinate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            });
            var adjData = { x: x, y: y, original: { x: origX, y: origY } };
            if (dragToX !== null) {
                adjData.drag_to = { x: dragToX, y: dragToY };
                adjData.original_drag_to = { x: origDragToX, y: origDragToY };
            }
            coordAdjustments[String(origIdx)] = adjData;
            step.coordinate = { x: x, y: y };
            if (step.action === 'drag' && dragToX !== null) {
                step.drag_to = { x: dragToX, y: dragToY };
                step.code = 'pyautogui.moveTo(' + x + ', ' + y + '); pyautogui.dragTo(' + dragToX + ', ' + dragToY + ')';
            } else if (step.action === 'click') {
                if (step.code.includes('doubleClick')) step.code = 'pyautogui.doubleClick(' + x + ', ' + y + ')';
                else if (step.code.includes('rightClick')) step.code = 'pyautogui.rightClick(' + x + ', ' + y + ')';
                else if (step.code.includes('clicks=3')) step.code = 'pyautogui.click(' + x + ', ' + y + ', clicks=3)';
                else step.code = 'pyautogui.click(' + x + ', ' + y + ')';
            }
            stackedFinetuneIdx = null;
            renderAllStepsStacked();
            renderStepSidebar();
        }

        function cancelStackedFinetune(idx) {
            const panel = document.getElementById('stacked-finetune-' + idx);
            if (panel) panel.classList.remove('show');
            stackedFinetuneIdx = null;
            // Restore original marker
            const step = taskData.steps[idx];
            const imgEl = document.getElementById('stacked-img-el-' + idx);
            if (step && imgEl) {
                addMarkersToContainer('stacked-img-' + idx, imgEl, step, taskData.video_width || 1920, taskData.video_height || 1080);
            }
        }

        function handleStackedImageClick(event, idx) {
            if (stackedFinetuneIdx !== idx) return;
            const imgEl = document.getElementById('stacked-img-el-' + idx);
            if (!imgEl) return;
            const rect = imgEl.getBoundingClientRect();
            const videoWidth = taskData.video_width || 1920;
            const videoHeight = taskData.video_height || 1080;
            const scaleX = imgEl.clientWidth / videoWidth;
            const scaleY = imgEl.clientHeight / videoHeight;
            const x = Math.round((event.clientX - rect.left) / scaleX);
            const y = Math.round((event.clientY - rect.top) / scaleY);
            const xInput = document.getElementById('stacked-cx-' + idx);
            const yInput = document.getElementById('stacked-cy-' + idx);
            if (xInput) xInput.value = x;
            if (yInput) yInput.value = y;
            updateStackedMarker(idx);
        }

        // ========== View mode: single / grid ==========

        function setViewMode(mode) {
            viewMode = mode;
            document.getElementById('viewSingle').classList.toggle('active', mode === 'single');
            document.getElementById('viewGrid').classList.toggle('active', mode === 'grid');
            document.getElementById('singleStepView').style.display = mode === 'single' ? '' : 'none';
            document.getElementById('gridStepView').style.display = mode === 'grid' ? '' : 'none';
            // Hide prev/next in both modes (stacked view scrolls, grid has pagination)
            document.getElementById('prevBtn').style.display = 'none';
            document.getElementById('nextBtn').style.display = 'none';
            document.getElementById('stepCounter').style.display = mode === 'single' ? '' : 'none';
            document.getElementById('pageSizeSelect').style.display = mode === 'grid' ? '' : 'none';
            if (mode === 'single') {
                renderAllStepsStacked();
            } else if (mode === 'grid') {
                gridPage = 0;
                renderGrid();
            }
        }

        function changePageSize() {
            gridPageSize = parseInt(document.getElementById('pageSizeSelect').value) || 30;
            gridPage = 0;
            renderGrid();
        }

        function renderGrid() {
            const steps = taskData ? taskData.steps || [] : [];
            const totalPages = Math.max(1, Math.ceil(steps.length / gridPageSize));
            if (gridPage >= totalPages) gridPage = totalPages - 1;
            const start = gridPage * gridPageSize;
            const end = Math.min(start + gridPageSize, steps.length);
            const pageSteps = steps.slice(start, end);
            const videoWidth = taskData ? taskData.video_width || 1920 : 1920;
            const videoHeight = taskData ? taskData.video_height || 1080 : 1080;

            let html = '';
            pageSteps.forEach((step, localIdx) => {
                const globalIdx = start + localIdx;
                const origIdx = step.original_index != null ? step.original_index : globalIdx;
                const adj = coordAdjustments[String(origIdx)];
                const isAdjusted = !!adj;
                const imgUrl = '/oss_frame/' + encodeURIComponent(folderName) + '/' + parseFloat(step.video_time).toFixed(1) + '?folder=' + encodeURIComponent(ossFolder);

                // AI check class for card border color
                const aiR = getAiStepResult(origIdx);
                let aiCardClass = '';
                if (aiR) aiCardClass = ' ai-' + (aiR.correctness || 'unknown');
                const hasGridError = !!stepErrors[String(origIdx)];
                if (hasGridError) aiCardClass += ' has-error';

                html += '<div class="grid-card' + aiCardClass + '" data-step="' + globalIdx + '" onclick="openDetail(' + globalIdx + ')">';
                html += '<div class="grid-card-img" id="grid-img-' + globalIdx + '">';
                html += '<img src="' + imgUrl + '" loading="lazy" data-step="' + globalIdx + '" />';
                html += '</div>';
                html += '<div class="grid-card-body">';
                html += '<div class="grid-card-header"><span class="grid-card-step">Step ' + globalIdx + '</span>';
                html += '<span class="grid-card-action">' + (step.action || '') + '</span></div>';
                if (step.code) html += '<div class="grid-card-code">' + step.code + '</div>';
                if (step.justification) html += '<div class="grid-card-justification">' + step.justification + '</div>';
                let badges = '';
                if (hasGridError) badges += '<span class="badge" style="background:#f44336;color:#fff;">ERROR</span>';
                if (isAdjusted) badges += '<span class="badge badge-adjusted">Adjusted</span>';
                if (aiR) {
                    const ccls = {correct:'ai-correct', wrong:'ai-wrong', redundant:'ai-redundant', suspicious:'ai-suspicious'}[aiR.correctness] || 'ai-unknown';
                    badges += '<span class="badge ai-badge ' + ccls + '">' + (aiR.correctness || '?').toUpperCase() + '</span>';
                    if (aiR.justification_quality === 'poor' || aiR.justification_quality === 'missing') {
                        badges += '<span class="badge ai-badge ai-poor">Just:' + aiR.justification_quality.toUpperCase() + '</span>';
                    }
                }
                if (badges) html += '<div class="grid-card-badges">' + badges + '</div>';
                html += '</div></div>';
            });
            document.getElementById('stepsGrid').innerHTML = html;

            // Render coord markers on grid images after images load
            pageSteps.forEach((step, localIdx) => {
                const globalIdx = start + localIdx;
                if (step.has_coordinate && step.coordinate) {
                    const container = document.getElementById('grid-img-' + globalIdx);
                    const img = container ? container.querySelector('img') : null;
                    if (img) {
                        const addMarkers = function() {
                            // Remove old markers
                            container.querySelectorAll('.grid-marker').forEach(m => m.remove());
                            const scaleX = img.clientWidth / videoWidth;
                            const scaleY = img.clientHeight / videoHeight;
                            if (step.action === 'drag' && step.drag_to) {
                                const m1 = document.createElement('div');
                                m1.className = 'grid-marker';
                                m1.style.left = (step.coordinate.x * scaleX) + 'px';
                                m1.style.top = (step.coordinate.y * scaleY) + 'px';
                                container.appendChild(m1);
                                const m2 = document.createElement('div');
                                m2.className = 'grid-marker drag-end';
                                m2.style.left = (step.drag_to.x * scaleX) + 'px';
                                m2.style.top = (step.drag_to.y * scaleY) + 'px';
                                container.appendChild(m2);
                            } else {
                                const m = document.createElement('div');
                                m.className = 'grid-marker';
                                m.style.left = (step.coordinate.x * scaleX) + 'px';
                                m.style.top = (step.coordinate.y * scaleY) + 'px';
                                container.appendChild(m);
                            }
                        };
                        if (img.complete) addMarkers();
                        else img.onload = addMarkers;
                    }
                }
            });

            // Page navigation
            let navHtml = '';
            if (totalPages > 1) {
                navHtml += '<button class="grid-page-btn" onclick="gridPrevPage()" ' + (gridPage <= 0 ? 'disabled' : '') + '>Prev Page</button>';
                navHtml += '<span class="grid-page-info">Page ' + (gridPage + 1) + ' / ' + totalPages + ' (' + steps.length + ' steps)</span>';
                navHtml += '<button class="grid-page-btn" onclick="gridNextPage()" ' + (gridPage >= totalPages - 1 ? 'disabled' : '') + '>Next Page</button>';
            } else {
                navHtml += '<span class="grid-page-info">' + steps.length + ' steps</span>';
            }
            document.getElementById('gridPageNav').innerHTML = navHtml;
            document.getElementById('gridPageNavBottom').innerHTML = navHtml;
        }

        function gridPrevPage() { if (gridPage > 0) { gridPage--; renderGrid(); } }
        function gridNextPage() {
            const steps = taskData ? taskData.steps || [] : [];
            const totalPages = Math.ceil(steps.length / gridPageSize);
            if (gridPage < totalPages - 1) { gridPage++; renderGrid(); }
        }

        // ========== Detail overlay (grid card click) ==========

        function openDetail(idx) {
            detailStepIdx = idx;
            renderDetailOverlay(idx);
            document.getElementById('detailOverlay').classList.add('show');
        }

        function closeDetail() {
            document.getElementById('detailOverlay').classList.remove('show');
            detailStepIdx = -1;
            // Re-render grid in case justification was edited
            if (viewMode === 'grid') renderGrid();
        }

        function detailPrev() {
            if (detailStepIdx > 0) { detailStepIdx--; renderDetailOverlay(detailStepIdx); }
        }

        function detailNext() {
            const steps = taskData ? taskData.steps || [] : [];
            if (detailStepIdx < steps.length - 1) { detailStepIdx++; renderDetailOverlay(detailStepIdx); }
        }

        let detailFinetuneActive = false;

        function renderDetailOverlay(idx) {
            const steps = taskData ? taskData.steps || [] : [];
            if (idx < 0 || idx >= steps.length) return;
            const step = steps[idx];
            const origIdx = step.original_index != null ? step.original_index : idx;
            const adj = coordAdjustments[String(origIdx)];
            const isAdjusted = !!adj;
            const origCoord = isAdjusted ? (adj.original || step.coordinate) : step.coordinate;
            const videoWidth = taskData.video_width || 1920;
            const videoHeight = taskData.video_height || 1080;
            detailFinetuneActive = false;

            document.getElementById('detailOverlayTitle').textContent = 'Step ' + idx + ': ' + (step.action || '');
            document.getElementById('detailPrevBtn').disabled = idx <= 0;
            document.getElementById('detailNextBtn').disabled = idx >= steps.length - 1;

            const imgUrl = '/oss_frame/' + encodeURIComponent(folderName) + '/' + parseFloat(step.video_time).toFixed(1) + '?folder=' + encodeURIComponent(ossFolder);

            let html = '<div class="image-info" id="detailResInfo"><div class="resolution-info">Res: <b>' + videoWidth + ' x ' + videoHeight + '</b></div></div>';
            html += '<div style="position:relative;display:inline-block;max-width:100%;background:#000;border-radius:6px;overflow:hidden;margin-bottom:10px;" id="detailImgContainer" onclick="handleDetailImageClick(event,' + idx + ')">';
            html += '<img id="detailImg" src="' + imgUrl + '" style="max-width:100%;max-height:60vh;display:block;" />';
            html += '<svg class="drag-line" id="detailDragSvg" style="display:none;position:absolute;top:0;left:0;width:100%;height:100%;"><line id="detailDragLine" /><polygon id="detailDragArrow" /></svg>';
            html += '</div>';

            // Frame adjustment controls
            html += '<div class="frame-adjust-controls">';
            html += '<button class="frame-btn" onclick="shiftFrame(' + idx + ',-0.8)" title="-0.8s">&#9664;&#9664;</button>';
            html += '<button class="frame-btn" onclick="shiftFrame(' + idx + ',-0.2)" title="-0.2s">&#9664;</button>';
            html += '<span class="frame-time" id="detail-frame-time">t=' + step.video_time.toFixed(2) + 's</span>';
            html += '<button class="frame-btn" onclick="shiftFrame(' + idx + ',0.2)" title="+0.2s">&#9654;</button>';
            html += '<button class="frame-btn" onclick="shiftFrame(' + idx + ',0.8)" title="+0.8s">&#9654;&#9654;</button>';
            html += '</div>';

            // Error banner in detail
            const hasDetailError = !!stepErrors[String(origIdx)];
            if (hasDetailError) {
                const errNote = (stepErrors[String(origIdx)] && stepErrors[String(origIdx)].note) || '';
                html += '<div class="step-error-banner"><span class="error-icon">&#9888;</span><div>';
                html += '<span class="error-text">Marked as error by reviewer</span>';
                if (errNote) html += '<div style="color:#ffab91;font-size:0.8em;margin-top:2px;">' + errNote + '</div>';
                if (!isAnnotatorMode) {
                    html += '<input class="step-error-note-input" placeholder="Note for annotator..." value="' + errNote.replace(/"/g, '&quot;') + '" onchange="saveStepErrorNote(' + idx + ', this.value)" />';
                }
                html += '</div></div>';
            }

            html += '<div class="step-details" style="margin-bottom:10px;">';
            html += '<h4>Step ' + idx + ': ' + (step.action || '');
            if (isAdjusted) html += ' <span class="adjusted-badge">Adjusted</span>';
            if (!isAnnotatorMode) {
                html += ' <button class="btn-mark-error' + (hasDetailError ? ' marked' : '') + '" onclick="toggleStepError(' + idx + ');renderDetailOverlay(' + idx + ')">' + (hasDetailError ? 'Unmark Error' : 'Mark Error') + '</button>';
            }
            html += ' <button class="delete-step-btn" onclick="confirmDeleteStep(' + idx + ')" title="Delete this step">Delete Step</button>';
            html += '</h4>';
            if (step.action === 'type' || step.action === 'press') {
                html += '<div class="code code-editable"><input type="text" class="code-edit-input" value="' + (step.code || '').replace(/"/g, '&quot;') + '" onchange="saveCode(' + idx + ', this.value)" title="Edit code" /></div>';
            } else {
                html += '<div class="code">' + (step.code || '') + '</div>';
            }
            if (step.has_coordinate) {
                html += '<div style="margin-bottom:6px;font-size:0.85em;">';
                if (isAdjusted) {
                    html += 'Coord: <span class="current-coord" id="detail-coord-display">(' + step.coordinate.x + ', ' + step.coordinate.y + ')</span>';
                    html += '<span class="original-coord">(' + origCoord.x + ', ' + origCoord.y + ')</span>';
                } else {
                    html += 'Coord: <span id="detail-coord-display">(' + step.coordinate.x + ', ' + step.coordinate.y + ')</span>';
                }
                html += ' <button class="finetune-btn" onclick="toggleDetailFinetune(' + idx + ')">Fine-tune</button></div>';
                // Finetune panel
                html += '<div class="coord-adjust-panel" id="detail-finetune-panel">';
                html += '<h5>Coordinate Fine-tuning</h5>';
                html += '<div class="coord-adjust-controls">';
                if (step.action === 'drag' && step.drag_to) {
                    html += '<div style="margin-bottom:4px;color:#888;font-size:0.78em;">Start (From):</div>';
                }
                html += '<div class="coord-field"><label>X:</label><input type="number" class="coord-input" id="detail-cx" value="' + step.coordinate.x + '" min="0" max="' + videoWidth + '" oninput="updateDetailMarker(' + idx + ')"></div>';
                html += '<div class="coord-field"><label>Y:</label><input type="number" class="coord-input" id="detail-cy" value="' + step.coordinate.y + '" min="0" max="' + videoHeight + '" oninput="updateDetailMarker(' + idx + ')"></div>';
                if (step.action === 'drag' && step.drag_to) {
                    html += '</div><div class="coord-adjust-controls" style="margin-top:4px;">';
                    html += '<div style="margin-bottom:4px;color:#888;font-size:0.78em;">End (To):</div>';
                    html += '<div class="coord-field"><label>X2:</label><input type="number" class="coord-input" id="detail-dx" value="' + step.drag_to.x + '" min="0" max="' + videoWidth + '" oninput="updateDetailMarker(' + idx + ')"></div>';
                    html += '<div class="coord-field"><label>Y2:</label><input type="number" class="coord-input" id="detail-dy" value="' + step.drag_to.y + '" min="0" max="' + videoHeight + '" oninput="updateDetailMarker(' + idx + ')"></div>';
                }
                html += '<button class="finetune-btn save" onclick="saveDetailCoordinate(' + idx + ')">Save</button>';
                html += '<button class="finetune-btn cancel" onclick="cancelDetailFinetune(' + idx + ')">Cancel</button>';
                html += '<span style="color:#888;font-size:0.75em;margin-left:8px;">Click image to set</span>';
                html += '</div></div>';
            }
            if (step.description) html += '<div class="description">' + step.description + '</div>';
            html += '<div class="justification-edit-area">';
            html += '<label style="color:#888;font-size:0.78em;display:block;margin-bottom:3px;">Justification:</label>';
            html += '<textarea class="justification-input" onchange="saveJustification(' + idx + ', this.value)" placeholder="Step justification...">' + (step.justification || '') + '</textarea>';
            html += '</div>';
            html += renderAiResultHtml(origIdx);
            html += '</div>';

            document.getElementById('detailOverlayBody').innerHTML = html;

            // Add markers + drag lines + update scale info
            const detailImg = document.getElementById('detailImg');
            const doDetailMarkers = function() {
                const container = document.getElementById('detailImgContainer');
                if (!container) return;
                container.querySelectorAll('.coord-marker').forEach(m => m.remove());
                const scaleX = detailImg.clientWidth / videoWidth;
                const scaleY = detailImg.clientHeight / videoHeight;
                // Update res info with scale
                const resInfo = document.getElementById('detailResInfo');
                if (resInfo) resInfo.innerHTML = '<div class="resolution-info">Res: <b>' + videoWidth + ' x ' + videoHeight + '</b> | Scale: <b>' + (scaleX * 100).toFixed(0) + '%</b></div>';
                if (step.has_coordinate && step.coordinate) {
                    if (step.action === 'drag' && step.drag_to) {
                        const sx = step.coordinate.x * scaleX, sy = step.coordinate.y * scaleY;
                        const ex = step.drag_to.x * scaleX, ey = step.drag_to.y * scaleY;
                        const m1 = document.createElement('div'); m1.className = 'coord-marker';
                        m1.setAttribute('data-label', 'from (' + step.coordinate.x + ',' + step.coordinate.y + ')');
                        m1.style.left = sx + 'px'; m1.style.top = sy + 'px'; m1.style.display = 'block';
                        container.appendChild(m1);
                        const m2 = document.createElement('div'); m2.className = 'coord-marker drag-end';
                        m2.setAttribute('data-label', 'to (' + step.drag_to.x + ',' + step.drag_to.y + ')');
                        m2.style.left = ex + 'px'; m2.style.top = ey + 'px'; m2.style.display = 'block';
                        container.appendChild(m2);
                        const dSvg = document.getElementById('detailDragSvg');
                        const dLine = document.getElementById('detailDragLine');
                        const dArrow = document.getElementById('detailDragArrow');
                        if (dSvg && dLine && dArrow) {
                            dLine.setAttribute('x1', sx); dLine.setAttribute('y1', sy);
                            dLine.setAttribute('x2', ex); dLine.setAttribute('y2', ey);
                            const angle = Math.atan2(ey - sy, ex - sx), aLen = 10;
                            dArrow.setAttribute('points', ex+','+ey+' '+(ex-aLen*Math.cos(angle-0.4))+','+(ey-aLen*Math.sin(angle-0.4))+' '+(ex-aLen*Math.cos(angle+0.4))+','+(ey-aLen*Math.sin(angle+0.4)));
                            dSvg.style.display = 'block';
                        }
                    } else {
                        const m = document.createElement('div'); m.className = 'coord-marker';
                        m.setAttribute('data-label', step.action + ' (' + step.coordinate.x + ',' + step.coordinate.y + ')');
                        m.style.left = (step.coordinate.x * scaleX) + 'px'; m.style.top = (step.coordinate.y * scaleY) + 'px';
                        m.style.display = 'block'; container.appendChild(m);
                        const dSvg = document.getElementById('detailDragSvg');
                        if (dSvg) dSvg.style.display = 'none';
                    }
                }
            };
            if (detailImg) {
                if (detailImg.complete && detailImg.naturalWidth > 0) doDetailMarkers();
                else detailImg.onload = doDetailMarkers;
            }
        }

        function toggleDetailFinetune(idx) {
            const panel = document.getElementById('detail-finetune-panel');
            if (!panel) return;
            detailFinetuneActive = !detailFinetuneActive;
            panel.classList.toggle('show', detailFinetuneActive);
        }

        function updateDetailMarker(idx) {
            const step = taskData.steps[idx];
            const xInput = document.getElementById('detail-cx');
            const yInput = document.getElementById('detail-cy');
            const imgEl = document.getElementById('detailImg');
            if (!xInput || !yInput || !imgEl) return;
            const x = parseInt(xInput.value) || 0;
            const y = parseInt(yInput.value) || 0;
            const videoWidth = taskData.video_width || 1920;
            const videoHeight = taskData.video_height || 1080;
            const container = document.getElementById('detailImgContainer');
            if (!container) return;
            container.querySelectorAll('.coord-marker').forEach(m => m.remove());
            const scaleX = imgEl.clientWidth / videoWidth;
            const scaleY = imgEl.clientHeight / videoHeight;
            const m = document.createElement('div'); m.className = 'coord-marker';
            m.setAttribute('data-label', step.action + ' (' + x + ',' + y + ')');
            m.style.left = (x * scaleX) + 'px'; m.style.top = (y * scaleY) + 'px';
            m.style.display = 'block'; container.appendChild(m);
        }

        function handleDetailImageClick(event, idx) {
            if (!detailFinetuneActive) return;
            const imgEl = document.getElementById('detailImg');
            if (!imgEl) return;
            const rect = imgEl.getBoundingClientRect();
            const videoWidth = taskData.video_width || 1920;
            const videoHeight = taskData.video_height || 1080;
            const scaleX = imgEl.clientWidth / videoWidth;
            const scaleY = imgEl.clientHeight / videoHeight;
            const x = Math.round((event.clientX - rect.left) / scaleX);
            const y = Math.round((event.clientY - rect.top) / scaleY);
            const xInput = document.getElementById('detail-cx');
            const yInput = document.getElementById('detail-cy');
            if (xInput) xInput.value = x;
            if (yInput) yInput.value = y;
            updateDetailMarker(idx);
        }

        async function saveDetailCoordinate(idx) {
            const step = taskData.steps[idx];
            const xInput = document.getElementById('detail-cx');
            const yInput = document.getElementById('detail-cy');
            if (!xInput || !yInput) return;
            const x = parseInt(xInput.value) || 0;
            const y = parseInt(yInput.value) || 0;
            const origIdx = step.original_index != null ? step.original_index : idx;
            const adj = coordAdjustments[String(origIdx)];
            let origX, origY;
            if (adj && adj.original) { origX = adj.original.x; origY = adj.original.y; }
            else { origX = step.coordinate.x; origY = step.coordinate.y; }

            let dragToX = null, dragToY = null, origDragToX = null, origDragToY = null;
            if (step.action === 'drag' && step.drag_to) {
                const dxI = document.getElementById('detail-dx');
                const dyI = document.getElementById('detail-dy');
                if (dxI && dyI) {
                    dragToX = parseInt(dxI.value) || 0;
                    dragToY = parseInt(dyI.value) || 0;
                    if (adj && adj.original_drag_to) { origDragToX = adj.original_drag_to.x; origDragToY = adj.original_drag_to.y; }
                    else { origDragToX = step.drag_to.x; origDragToY = step.drag_to.y; }
                }
            }

            var body = { folder_name: folderName, oss_folder: ossFolder, step_index: origIdx, x: x, y: y, original_x: origX, original_y: origY };
            if (dragToX !== null) { body.drag_to_x = dragToX; body.drag_to_y = dragToY; body.original_drag_to_x = origDragToX; body.original_drag_to_y = origDragToY; }

            await fetch('/api/oss/update_coordinate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            });
            var adjData = { x: x, y: y, original: { x: origX, y: origY } };
            if (dragToX !== null) { adjData.drag_to = { x: dragToX, y: dragToY }; adjData.original_drag_to = { x: origDragToX, y: origDragToY }; }
            coordAdjustments[String(origIdx)] = adjData;
            step.coordinate = { x: x, y: y };
            if (step.action === 'drag' && dragToX !== null) {
                step.drag_to = { x: dragToX, y: dragToY };
                step.code = 'pyautogui.moveTo(' + x + ', ' + y + '); pyautogui.dragTo(' + dragToX + ', ' + dragToY + ')';
            } else if (step.action === 'click') {
                if (step.code.includes('doubleClick')) step.code = 'pyautogui.doubleClick(' + x + ', ' + y + ')';
                else if (step.code.includes('rightClick')) step.code = 'pyautogui.rightClick(' + x + ', ' + y + ')';
                else if (step.code.includes('clicks=3')) step.code = 'pyautogui.click(' + x + ', ' + y + ', clicks=3)';
                else step.code = 'pyautogui.click(' + x + ', ' + y + ')';
            }
            detailFinetuneActive = false;
            renderDetailOverlay(idx);
            renderGrid();
            renderStepSidebar();
        }

        function cancelDetailFinetune(idx) {
            detailFinetuneActive = false;
            renderDetailOverlay(idx);
        }

        // ========== Annotation panels ==========

        function renderAnnotationPanels() {
            const ann = annotation || {};
            const scores = ann.scores || {};

            // Annotators only see knowledge panel, not verdict/evaluation
            if (isAnnotatorMode) {
                document.getElementById('verdictPanel').innerHTML = '';
                document.getElementById('evaluationPanel').innerHTML = '';
            } else {
            document.getElementById('verdictPanel').innerHTML =
                '<h4>Case Verdict</h4>' +
                '<div class="mark-buttons">' +
                '<button class="mark-btn pass' + (ann.mark === 'pass' ? ' active' : '') + '" onclick="setMark(\\'pass\\')">PASS</button>' +
                '<button class="mark-btn fail' + (ann.mark === 'fail' ? ' active' : '') + '" onclick="setMark(\\'fail\\')">FAIL</button>' +
                '<button class="mark-btn unclear' + (!ann.mark ? ' active' : '') + '" onclick="setMark(null)">UNCLEAR</button>' +
                '</div>';

            const failMode = ann.mark === 'fail';
            let evalHtml = '<div class="case-evaluation' + (failMode ? ' fail-mode' : '') + '">';
            evalHtml += '<h4>Evaluation' + (failMode ? ' - Fail Reason' : ' - Pass Reason') + '</h4>';
            evalHtml += '<div class="rating-section">';
            const dims = [
                { key: 'correctness', label: 'Correctness', primary: true },
                { key: 'difficulty', label: 'Difficulty', primary: false },
                { key: 'knowledge_richness', label: 'Knowledge Richness', primary: false },
                { key: 'task_value', label: 'Task Value', primary: false }
            ];
            dims.forEach(d => {
                evalHtml += '<div class="rating-item"><label>' + d.label;
                if (d.primary) evalHtml += ' <span class="label-primary">Primary</span>';
                evalHtml += '</label><div class="rating-stars">';
                for (let n = 1; n <= 5; n++) {
                    evalHtml += '<button class="rating-star' + ((scores[d.key] || 0) >= n ? ' active' : '') + '" onclick="setCaseRating(\\'' + d.key + '\\',' + n + ')">' + n + '</button>';
                }
                evalHtml += '</div></div>';
            });
            evalHtml += '</div>';
            evalHtml += '<textarea class="pass-reason-input" id="pass-reason" placeholder="' +
                (failMode ? 'Describe why failed...' : 'Describe why passed...') +
                '" onchange="updatePassReason(this.value)">' + (ann.pass_reason || '') + '</textarea>';
            evalHtml += '</div>';
            document.getElementById('evaluationPanel').innerHTML = evalHtml;
            } // end if not annotator mode

            // Knowledge panel (visible to both reviewer and annotator)
            let kHtml = '<div class="knowledge-panel">';
            kHtml += '<h4>Knowledge Points & Apps</h4>';
            kHtml += '<div class="knowledge-section">';
            const fields = [
                { key: 'osworld_overlap', label: 'OSWorld Overlap', inputId: 'osworld-input' },
                { key: 'custom_nodes', label: 'Custom Nodes', inputId: 'custom-input' },
                { key: 'related_apps', label: 'Related Apps', inputId: 'apps-input' }
            ];
            fields.forEach(f => {
                const tags = ann[f.key] || [];
                kHtml += '<div class="knowledge-item"><label>' + f.label + '</label>';
                kHtml += '<div class="knowledge-tags">';
                tags.forEach(tag => {
                    kHtml += '<span class="knowledge-tag">' + tag + '<span class="remove-tag" onclick="removeTag(\\'' + f.key + '\\',\\'' + tag.replace(/'/g, "\\\\'") + '\\')">x</span></span>';
                });
                kHtml += '</div>';
                kHtml += '<div class="knowledge-input-row">';
                kHtml += '<input type="text" class="knowledge-input" id="' + f.inputId + '" placeholder="Add...">';
                kHtml += '<button class="knowledge-add-btn" onclick="addTag(\\'' + f.key + '\\',\\'' + f.inputId + '\\')">+</button>';
                kHtml += '</div></div>';
            });
            kHtml += '</div>';
            kHtml += '<label style="color:#888;font-size:0.8em;display:block;margin-bottom:3px;">Step-by-Step Instructions:</label>';
            kHtml += '<textarea class="step-instructions-input" id="step-instructions" placeholder="Step-by-step instructions..." onchange="updateStepInstructions(this.value)">' + (ann.step_by_step_instructions || '') + '</textarea>';
            kHtml += '</div>';
            document.getElementById('knowledgePanel').innerHTML = kHtml;
        }

        // ========== Annotation actions ==========

        async function setMark(mark) {
            annotation.mark = mark;
            await saveOssAnnotation();
            renderAnnotationPanels();
            if (mark === 'pass') reviewStatus = 'reviewed';
            else if (mark === 'fail') reviewStatus = 'rejected';
            else reviewStatus = 'unreviewed';
            updateReviewUI();
            updateRecSidebarStatus();
        }

        async function setCaseRating(dim, val) {
            if (!annotation.scores) annotation.scores = {};
            annotation.scores[dim] = val;
            await saveOssAnnotation();
            renderAnnotationPanels();
        }

        async function updatePassReason(reason) {
            annotation.pass_reason = reason;
            await saveOssAnnotation();
        }

        async function updateStepInstructions(text) {
            annotation.step_by_step_instructions = text;
            await saveOssAnnotation();
        }

        async function addTag(field, inputId) {
            const input = document.getElementById(inputId);
            const value = input.value.trim();
            if (!value) return;
            if (!annotation[field]) annotation[field] = [];
            if (!annotation[field].includes(value)) {
                annotation[field].push(value);
                await saveOssAnnotation();
                renderAnnotationPanels();
            }
            input.value = '';
        }

        async function removeTag(field, value) {
            if (!annotation[field]) return;
            const idx = annotation[field].indexOf(value);
            if (idx > -1) {
                annotation[field].splice(idx, 1);
                await saveOssAnnotation();
                renderAnnotationPanels();
            }
        }

        async function saveOssAnnotation() {
            try {
                await fetch('/api/oss/annotate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        folder_name: folderName,
                        oss_folder: ossFolder,
                        mark: annotation.mark || null,
                        scores: annotation.scores || {},
                        pass_reason: annotation.pass_reason || '',
                        osworld_overlap: annotation.osworld_overlap || [],
                        custom_nodes: annotation.custom_nodes || [],
                        related_apps: annotation.related_apps || [],
                        step_by_step_instructions: annotation.step_by_step_instructions || ''
                    })
                });
            } catch (err) { console.error('Save failed:', err); }
        }

        // ========== Step Error Marking (Reviewer) ==========

        async function toggleStepError(stepIdx) {
            const step = taskData && taskData.steps[stepIdx];
            if (!step) return;
            const origIdx = step.original_index != null ? step.original_index : stepIdx;
            const si = String(origIdx);
            const isCurrentlyError = !!stepErrors[si];

            if (isCurrentlyError) {
                delete stepErrors[si];
            } else {
                stepErrors[si] = {note: ''};
            }

            try {
                await fetch('/api/oss/mark_step_error', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        folder_name: folderName, oss_folder: ossFolder,
                        step_index: origIdx, is_error: !isCurrentlyError
                    })
                });
            } catch (err) { console.error('Mark error failed:', err); }

            // Update UI in-place
            updateStepErrorUI(stepIdx);
            renderStepSidebar();
            updateReviewProgress();
        }

        async function saveStepErrorNote(stepIdx, note) {
            const step = taskData && taskData.steps[stepIdx];
            if (!step) return;
            const origIdx = step.original_index != null ? step.original_index : stepIdx;
            const si = String(origIdx);
            if (!stepErrors[si]) return;
            stepErrors[si].note = note;

            try {
                await fetch('/api/oss/mark_step_error', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        folder_name: folderName, oss_folder: ossFolder,
                        step_index: origIdx, is_error: true, note: note
                    })
                });
            } catch (err) { console.error('Save error note failed:', err); }
        }

        async function restoreDeletedSteps() {
            if (!confirm('Restore all deleted steps?')) return;
            const deleted = (annotation.deleted_steps || []).slice();
            for (const origIdx of deleted) {
                try {
                    await fetch('/api/oss/undelete_step', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ folder_name: folderName, oss_folder: ossFolder, original_index: origIdx })
                    });
                } catch(e) {}
            }
            loadTask();
        }

        async function setAnnotatorStatus(s) {
            if (annotatorStatus === s) s = 'not_started';
            annotatorStatus = s;
            var b1 = document.getElementById('annBtnIP'), b2 = document.getElementById('annBtnDone');
            if (b1) { b1.style.background = s==='in_progress'?'#ff9800':'transparent'; b1.style.color = s==='in_progress'?'#000':'#ff9800'; }
            if (b2) { b2.style.background = s==='completed'?'#4caf50':'transparent'; b2.style.color = s==='completed'?'#fff':'#4caf50'; }
            try { await fetch('/api/oss/set_annotator_status', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({folder_name:folderName,oss_folder:ossFolder,status:s}) }); } catch(e) {}
        }

        function updateStepErrorUI(stepIdx) {
            const step = taskData && taskData.steps[stepIdx];
            if (!step) return;
            const origIdx = step.original_index != null ? step.original_index : stepIdx;
            const si = String(origIdx);
            const isError = !!stepErrors[si];

            // Update stacked card
            const card = document.getElementById('stacked-step-' + stepIdx);
            if (card) {
                card.classList.toggle('has-error', isError);
                // Update button
                const btn = card.querySelector('.btn-mark-error');
                if (btn) {
                    btn.classList.toggle('marked', isError);
                    btn.textContent = isError ? 'Unmark Error' : 'Mark Error';
                }
                // Update/remove error banner
                let banner = card.querySelector('.step-error-banner');
                if (isError && !banner) {
                    // Insert banner after header
                    const header = card.querySelector('.stacked-step-header');
                    if (header) {
                        banner = document.createElement('div');
                        banner.className = 'step-error-banner';
                        banner.innerHTML = '<span class="error-icon">&#9888;</span><div><span class="error-text">Marked as error by reviewer</span>' +
                            (isAnnotatorMode ? '' : '<input class="step-error-note-input" placeholder="Note for annotator..." value="' + ((stepErrors[si] && stepErrors[si].note) || '').replace(/"/g, '&quot;') + '" onchange="saveStepErrorNote(' + stepIdx + ', this.value)" />') +
                            '</div>';
                        header.after(banner);
                    }
                } else if (!isError && banner) {
                    banner.remove();
                } else if (isError && banner && !isAnnotatorMode) {
                    const noteInput = banner.querySelector('.step-error-note-input');
                    if (noteInput) noteInput.value = (stepErrors[si] && stepErrors[si].note) || '';
                }
            }
        }

        function updateReviewProgress() {
            const steps = taskData ? taskData.steps || [] : [];
            const totalSteps = steps.length;
            const errorCount = Object.keys(stepErrors).length;
            // Count "reviewed" as steps that have been explicitly checked (either marked error or have AI result)
            // For the reviewer progress, we track how many steps they have reviewed by looking at steps with any AI result or error mark
            const reviewedCount = steps.filter((s, i) => {
                const origIdx = s.original_index != null ? s.original_index : i;
                return !!stepErrors[String(origIdx)] || !!getAiStepResult(origIdx);
            }).length;

            const pct = totalSteps > 0 ? Math.round((reviewedCount / totalSteps) * 100) : 0;
            document.getElementById('reviewProgressText').innerHTML =
                'Errors: <b style="color:#f44336">' + errorCount + '</b> | Reviewed: ' + reviewedCount + '/' + totalSteps;
            document.getElementById('reviewProgressFill').style.width = pct + '%';
            document.getElementById('reviewProgressContainer').style.display = 'flex';
        }

        function updateRecSidebarStatus() {
            // Update the recording sidebar to reflect new status
            const rec = allRecordings.find(r => r.folder_name === folderName);
            if (rec) {
                rec.review_status = reviewStatus;
                renderRecordingSidebar();
            }
        }

        // ========== Editable fields ==========

        async function saveJustification(stepIdx, value) {
            // Update local data and use original_index for overlay reference
            const step = taskData && taskData.steps[stepIdx];
            if (step) {
                step.justification = value;
                const origIdx = step.original_index != null ? step.original_index : stepIdx;
                try {
                    await fetch('/api/oss/update_justification', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            folder_name: folderName, oss_folder: ossFolder,
                            step_index: origIdx, justification: value
                        })
                    });
                } catch (err) { console.error('Save justification failed:', err); }
            }
        }

        function codeToDescription(code) {
            // Reverse-map pyautogui code to human-readable description
            var m;
            m = code.match(/pyautogui\.write\(['"](.*?)['"]\)/);
            if (m) return '\u2328\ufe0f Type: ' + m[1];
            m = code.match(/pyautogui\.hotkey\((.+)\)/);
            if (m) return '\u2328\ufe0f Press: ' + m[1].replace(/['"]/g,'').replace(/,\s*/g,' + ');
            m = code.match(/pyautogui\.press\(['"](.*?)['"]\)/);
            if (m) return '\u2328\ufe0f Press: ' + m[1];
            m = code.match(/pyautogui\.moveTo\((\d+),\s*(\d+)\).*dragTo\((\d+),\s*(\d+)\)/);
            if (m) return 'Drag from (' + m[1] + ', ' + m[2] + ') to (' + m[3] + ', ' + m[4] + ')';
            m = code.match(/pyautogui\.doubleClick\((\d+),\s*(\d+)\)/);
            if (m) return 'Double left Click at (' + m[1] + ', ' + m[2] + ')';
            m = code.match(/pyautogui\.rightClick\((\d+),\s*(\d+)\)/);
            if (m) return 'Right Click at (' + m[1] + ', ' + m[2] + ')';
            m = code.match(/pyautogui\.click\((\d+),\s*(\d+)\)/);
            if (m) return 'Single left Click at (' + m[1] + ', ' + m[2] + ')';
            m = code.match(/pyautogui\.scroll\((-?\d+)\)/);
            if (m) { var n = parseInt(m[1]); return n < 0 ? '\u2b07\ufe0f\u00d7' + Math.abs(n) + ' Scroll Down' : '\u2b06\ufe0f\u00d7' + n + ' Scroll Up'; }
            return code;
        }

        async function saveCode(stepIdx, value) {
            const step = taskData && taskData.steps[stepIdx];
            if (step) {
                step.code = value;
                // Sync description from code
                step.description = codeToDescription(value);
                // Update description display in stacked view
                var card = document.getElementById('stacked-step-' + stepIdx);
                if (card) {
                    var descEl = card.querySelector('.description');
                    if (descEl) descEl.textContent = step.description;
                }
                const origIdx = step.original_index != null ? step.original_index : stepIdx;
                try {
                    await fetch('/api/oss/update_code', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            folder_name: folderName, oss_folder: ossFolder,
                            step_index: origIdx, code: value
                        })
                    });
                } catch (err) { console.error('Save code failed:', err); }
            }
        }

        let _frameShiftTimer = null;
        function shiftFrame(stepIdx, delta) {
            const step = taskData && taskData.steps[stepIdx];
            if (!step) return;
            step.video_time = Math.max(0, step.video_time + delta);
            const origIdx = step.original_index != null ? step.original_index : stepIdx;
            const newUrl = '/oss_frame/' + encodeURIComponent(folderName) + '/' + step.video_time + '?folder=' + encodeURIComponent(ossFolder);

            // Update stacked view image
            const stackedImg = document.getElementById('stacked-img-el-' + stepIdx);
            if (stackedImg) {
                stackedImg.src = newUrl;
                // Re-add markers after image loads
                stackedImg.onload = function() {
                    const videoWidth = taskData.video_width || 1920;
                    const videoHeight = taskData.video_height || 1080;
                    addMarkersToContainer('stacked-img-' + stepIdx, stackedImg, step, videoWidth, videoHeight);
                };
            }
            // Update detail overlay image
            const detailImg = document.getElementById('detailImg');
            if (detailImg && detailStepIdx === stepIdx) detailImg.src = newUrl;

            // Update time displays
            const stackedTime = document.getElementById('frame-time-' + stepIdx);
            if (stackedTime) stackedTime.textContent = 't=' + step.video_time.toFixed(2) + 's';
            const detailTime = document.getElementById('detail-frame-time');
            if (detailTime && detailStepIdx === stepIdx) detailTime.textContent = 't=' + step.video_time.toFixed(2) + 's';

            // Debounced save
            clearTimeout(_frameShiftTimer);
            _frameShiftTimer = setTimeout(async () => {
                try {
                    await fetch('/api/oss/update_video_time', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            folder_name: folderName, oss_folder: ossFolder,
                            step_index: origIdx, video_time: step.video_time
                        })
                    });
                } catch (err) { console.error('Save video time failed:', err); }
            }, 300);
        }

        async function saveQuery(value) {
            try {
                await fetch('/api/oss/update_query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        folder_name: folderName, oss_folder: ossFolder, query: value
                    })
                });
                if (taskData && taskData.annotator_info) taskData.annotator_info.query = value;
            } catch (err) { console.error('Save query failed:', err); }
        }

        // ========== Delete step ==========

        let deleteConfirmStep = null;
        function confirmDeleteStep(stepIdx) {
            if (deleteConfirmStep === stepIdx) {
                // Second click - actually delete
                doDeleteStep(stepIdx);
                deleteConfirmStep = null;
            } else {
                // First click - ask for confirmation
                deleteConfirmStep = stepIdx;
                const btn = document.querySelector('.delete-step-btn');
                if (btn) {
                    btn.textContent = 'CONFIRM DELETE?';
                    btn.style.background = '#f44336';
                    btn.style.color = '#fff';
                }
                setTimeout(() => {
                    if (deleteConfirmStep === stepIdx) {
                        deleteConfirmStep = null;
                        const btn = document.querySelector('.delete-step-btn');
                        if (btn) {
                            btn.textContent = 'Delete Step';
                            btn.style.background = '';
                            btn.style.color = '';
                        }
                    }
                }, 3000);
            }
        }

        async function doDeleteStep(stepIdx) {
            // Use original_index for the overlay reference (not display index)
            const step = taskData && taskData.steps[stepIdx];
            if (!step) return;
            const origIdx = step.original_index != null ? step.original_index : stepIdx;
            try {
                await fetch('/api/oss/delete_step', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        folder_name: folderName, oss_folder: ossFolder, original_index: origIdx
                    })
                });
                // Remove step from local display data
                if (taskData && taskData.steps) {
                    taskData.steps.splice(stepIdx, 1);
                    // Re-index for display (keep original_index intact)
                    taskData.steps.forEach((s, i) => { s.index = i; });
                    if (currentStep >= taskData.steps.length) currentStep = Math.max(0, taskData.steps.length - 1);
                    renderStepSidebar();
                    // Close detail overlay if open
                    if (document.getElementById('detailOverlay').classList.contains('show')) {
                        closeDetail();
                    }
                    if (taskData.steps.length > 0) {
                        if (viewMode === 'single') renderAllStepsStacked();
                        if (viewMode === 'grid') renderGrid();
                    } else {
                        document.getElementById('stepsStacked').innerHTML = '<div style="padding:20px;color:#888;">No steps remaining</div>';
                    }
                }
            } catch (err) { console.error('Delete step failed:', err); }
        }

        // ========== Export ==========

        async function exportCase() {
            const btn = document.querySelector('.export-case-btn');
            if (btn) { btn.textContent = 'Exporting...'; btn.disabled = true; }
            try {
                const url = '/api/oss/export_case/' + encodeURIComponent(folderName) + '?folder=' + encodeURIComponent(ossFolder);
                const resp = await fetch(url);
                if (!resp.ok) {
                    const errData = await resp.json().catch(() => ({}));
                    alert('Export failed: ' + (errData.error || resp.statusText));
                    return;
                }
                const blob = await resp.blob();
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = folderName + '.zip';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(a.href);
                alert('Export downloaded successfully!');
            } catch (err) {
                alert('Export failed: ' + err.message);
            } finally {
                if (btn) { btn.textContent = 'Export'; btn.disabled = false; }
            }
        }

        // ========== Reset to Original ==========

        let resetConfirmed = false;
        function confirmResetCase() {
            const btn = document.querySelector('.btn-reset');
            if (resetConfirmed) {
                // Second click - actually reset
                doResetCase();
                resetConfirmed = false;
                if (btn) { btn.textContent = 'Reset to Original'; btn.classList.remove('confirming'); }
            } else {
                // First click - ask for confirmation
                resetConfirmed = true;
                if (btn) { btn.textContent = 'CONFIRM RESET?'; btn.classList.add('confirming'); }
                setTimeout(() => {
                    if (resetConfirmed) {
                        resetConfirmed = false;
                        if (btn) { btn.textContent = 'Reset to Original'; btn.classList.remove('confirming'); }
                    }
                }, 3000);
            }
        }

        async function doResetCase() {
            try {
                await fetch('/api/oss/reset_case', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ folder_name: folderName, oss_folder: ossFolder })
                });
                // Reload everything fresh from OSS
                taskData = null;
                currentStep = 0;
                finetuneActive = false;
                annotation = {};
                coordAdjustments = {};
                reviewStatus = 'unreviewed';
                // Reset AI check state
                if (aiCheckPolling) { clearInterval(aiCheckPolling); aiCheckPolling = null; }
                aiCheckResults = null;
                aiCheckStartTime = null;
                showAiProgress(false);
                document.getElementById('aiCheckBtn').classList.remove('running');
                document.getElementById('aiCheckBtn').textContent = 'AI Check';
                document.getElementById('aiCheckStatus').textContent = '';
                updateReviewUI();
                document.getElementById('loadingOverlay').style.display = 'flex';
                await loadTask();
                loadRecordingSidebar();
            } catch (err) { console.error('Reset failed:', err); }
        }

        // ========== Coordinate fine-tuning ==========

        function toggleFinetune() {
            finetuneActive = !finetuneActive;
            const panel = document.getElementById('finetune-panel');
            if (panel) panel.classList.toggle('show', finetuneActive);
        }

        function updateMarkerPreview() {
            const xInput = document.getElementById('coord-x');
            const yInput = document.getElementById('coord-y');
            const marker = document.getElementById('coordMarker');
            const img = document.getElementById('screenshot');
            if (!xInput || !yInput || !marker || !img) return;
            const x = parseInt(xInput.value) || 0;
            const y = parseInt(yInput.value) || 0;
            const scaleX = img.clientWidth / window._videoDims.width;
            const scaleY = img.clientHeight / window._videoDims.height;
            marker.style.left = (x * scaleX) + 'px';
            marker.style.top = (y * scaleY) + 'px';
            const step = taskData.steps[currentStep];
            marker.setAttribute('data-label', step.action + ' (' + x + ',' + y + ')');
            const cd = document.getElementById('coord-display');
            if (cd) cd.textContent = '(' + x + ', ' + y + ')';
        }

        async function saveCoordinate() {
            const xInput = document.getElementById('coord-x');
            const yInput = document.getElementById('coord-y');
            if (!xInput || !yInput) return;
            const x = parseInt(xInput.value) || 0;
            const y = parseInt(yInput.value) || 0;
            const step = taskData.steps[currentStep];
            const origIdx = step.original_index != null ? step.original_index : currentStep;
            let origX, origY;
            const adj = coordAdjustments[String(origIdx)];
            if (adj && adj.original) { origX = adj.original.x; origY = adj.original.y; }
            else { origX = window._origCoords.x; origY = window._origCoords.y; }

            await fetch('/api/oss/update_coordinate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    folder_name: folderName, oss_folder: ossFolder,
                    step_index: origIdx, x: x, y: y,
                    original_x: origX, original_y: origY
                })
            });
            coordAdjustments[String(origIdx)] = { x: x, y: y, original: { x: origX, y: origY } };
            step.coordinate = { x: x, y: y };
            if (step.action === 'click') {
                if (step.code.includes('doubleClick')) step.code = 'pyautogui.doubleClick(' + x + ', ' + y + ')';
                else if (step.code.includes('rightClick')) step.code = 'pyautogui.rightClick(' + x + ', ' + y + ')';
                else if (step.code.includes('clicks=3')) step.code = 'pyautogui.click(' + x + ', ' + y + ', clicks=3)';
                else step.code = 'pyautogui.click(' + x + ', ' + y + ')';
            }
            finetuneActive = false;
            renderStep(currentStep);
        }

        function cancelFinetune() { finetuneActive = false; renderStep(currentStep); }

        function handleImageClick(event) {
            if (!finetuneActive) return;
            const img = document.getElementById('screenshot');
            const rect = img.getBoundingClientRect();
            const scaleX = img.clientWidth / window._videoDims.width;
            const scaleY = img.clientHeight / window._videoDims.height;
            const x = Math.max(0, Math.min(window._videoDims.width, Math.round((event.clientX - rect.left) / scaleX)));
            const y = Math.max(0, Math.min(window._videoDims.height, Math.round((event.clientY - rect.top) / scaleY)));
            const xInput = document.getElementById('coord-x');
            const yInput = document.getElementById('coord-y');
            if (xInput) xInput.value = x;
            if (yInput) yInput.value = y;
            updateMarkerPreview();
        }

        // ========== AI Check ==========

        let aiCheckResults = null; // { status, steps: { origIdx: {...} }, ... }
        let aiCheckPolling = null;
        let aiCheckStartTime = null;

        function showAiProgress(show) {
            document.getElementById('aiProgressOverlay').classList.toggle('show', show);
        }

        function updateAiProgress(progress, total, steps) {
            const pct = total > 0 ? Math.round(progress / total * 100) : 0;
            document.getElementById('aiProgressPct').textContent = pct + '%';
            document.getElementById('aiProgressFill').style.width = pct + '%';
            document.getElementById('aiProgressSteps').textContent = 'Step ' + progress + ' / ' + total;

            // ETA calculation
            if (aiCheckStartTime && progress > 0) {
                const elapsed = (Date.now() - aiCheckStartTime) / 1000;
                const perStep = elapsed / progress;
                const remaining = Math.round(perStep * (total - progress));
                if (remaining > 60) {
                    document.getElementById('aiProgressEta').textContent = 'Est. ' + Math.ceil(remaining / 60) + ' min remaining';
                } else if (remaining > 0) {
                    document.getElementById('aiProgressEta').textContent = 'Est. ' + remaining + 's remaining';
                } else {
                    document.getElementById('aiProgressEta').textContent = 'Almost done...';
                }
            } else {
                document.getElementById('aiProgressEta').textContent = 'Estimating...';
            }

            // Live issue counter
            let issues = 0, ok = 0;
            if (steps) {
                Object.values(steps).forEach(s => {
                    if (s.correctness === 'wrong' || s.correctness === 'redundant' || s.correctness === 'suspicious') issues++;
                    else if (s.correctness === 'correct') ok++;
                    if (s.justification_quality === 'poor' || s.justification_quality === 'missing') issues++;
                });
            }
            const issuesDiv = document.getElementById('aiProgressIssues');
            if (progress > 0) {
                issuesDiv.innerHTML = '<span class="ok-count">' + ok + '</span> steps OK &nbsp;&middot;&nbsp; <span class="issue-count">' + issues + '</span> issues found so far';
            } else {
                issuesDiv.innerHTML = '';
            }
        }

        function startAiCheck(force) {
            const btn = document.getElementById('aiCheckBtn');
            btn.classList.add('running');
            btn.textContent = 'Checking...';
            document.getElementById('aiCheckStatus').textContent = '';
            aiCheckStartTime = Date.now();

            fetch('/api/oss/ai_check', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    folder_name: folderName, oss_folder: ossFolder, force: !!force
                })
            }).then(r => r.json()).then(data => {
                if (data.status === 'already_completed' && !force) {
                    aiCheckResults = data.results;
                    onAiCheckComplete();
                    return;
                }
                // Show overlay and start polling
                showAiProgress(true);
                updateAiProgress(0, data.total || 0, null);
                document.getElementById('aiProgressSubtitle').textContent =
                    'Analyzing ' + (data.total || '?') + ' steps with Gemini (sequential, with annotated screenshots)...';
                aiCheckPolling = setInterval(pollAiCheck, 1500);
            }).catch(err => {
                btn.classList.remove('running');
                btn.textContent = 'AI Check';
                document.getElementById('aiCheckStatus').textContent = 'Error: ' + err.message;
            });
        }

        function pollAiCheck() {
            fetch('/api/oss/ai_check_status?folder_name=' + encodeURIComponent(folderName) + '&folder=' + encodeURIComponent(ossFolder))
            .then(r => r.json()).then(data => {
                if (data.status === 'running') {
                    updateAiProgress(data.progress, data.total, data.steps);
                    // Update partial results live
                    if (data.steps) {
                        if (!aiCheckResults) aiCheckResults = { status: 'running', steps: {} };
                        aiCheckResults.steps = data.steps;
                        refreshAiDisplay();
                    }
                } else if (data.status === 'completed') {
                    clearInterval(aiCheckPolling);
                    aiCheckPolling = null;
                    aiCheckResults = data;
                    onAiCheckComplete();
                } else if (data.status === 'failed') {
                    clearInterval(aiCheckPolling);
                    aiCheckPolling = null;
                    showAiProgress(false);
                    document.getElementById('aiCheckBtn').classList.remove('running');
                    document.getElementById('aiCheckBtn').textContent = 'AI Check (Failed)';
                    document.getElementById('aiCheckStatus').textContent = data.error || 'Check failed';
                }
            });
        }

        function onAiCheckComplete() {
            showAiProgress(false);
            const btn = document.getElementById('aiCheckBtn');
            btn.classList.remove('running');
            btn.textContent = 'AI Check \u2713';
            // Count issues
            let issues = 0;
            const steps = (aiCheckResults && aiCheckResults.steps) || {};
            Object.values(steps).forEach(s => {
                if (s.correctness === 'wrong' || s.correctness === 'redundant' || s.correctness === 'suspicious') issues++;
                if (s.justification_quality === 'poor' || s.justification_quality === 'missing') issues++;
            });
            document.getElementById('aiCheckStatus').textContent = issues > 0 ? issues + ' issues found' : 'All OK';
            document.getElementById('aiCheckStatus').style.color = issues > 0 ? '#ff9800' : '#4caf50';
            refreshAiDisplay();
            updateReviewProgress();
        }

        function refreshAiDisplay() {
            // Re-render current view to show AI results
            if (viewMode === 'single') {
                renderStep(currentStep);
            } else {
                renderGrid();
            }
            renderStepSidebar();
        }

        function getAiStepResult(origIdx) {
            if (!aiCheckResults || !aiCheckResults.steps) return null;
            return aiCheckResults.steps[String(origIdx)] || null;
        }

        function renderAiResultHtml(origIdx) {
            const r = getAiStepResult(origIdx);
            if (!r) return '';

            const ccls = {correct:'ai-correct', wrong:'ai-wrong', redundant:'ai-redundant', suspicious:'ai-suspicious'}[r.correctness] || 'ai-unknown';
            const qcls = {good:'ai-good', acceptable:'ai-acceptable', poor:'ai-poor', missing:'ai-missing'}[r.justification_quality] || 'ai-unknown';

            let html = '<div class="ai-result">';
            html += '<div class="ai-badges">';
            html += '<span class="ai-badge ' + ccls + '">AI: ' + (r.correctness || 'unknown').toUpperCase() + '</span>';
            html += '<span class="ai-badge ' + qcls + '">Justification: ' + (r.justification_quality || 'unknown').toUpperCase() + '</span>';
            if (r.flags && r.flags.length > 0) {
                r.flags.forEach(function(f) { html += '<span class="ai-flag">' + f + '</span>'; });
            }
            html += '</div>';
            // Correctness reason (short verdict)
            if (r.correctness_reason) {
                html += '<div class="ai-reason"><b>\u5224\u5b9a\u7ed3\u679c (Verdict):</b> ' + r.correctness_reason + '</div>';
            }
            // AI reasoning (detailed analysis)
            if (r.reasoning) {
                html += '<div class="ai-reasoning"><b>AI \u5206\u6790:</b> ' + r.reasoning + '</div>';
            }
            if (r.justification_issues && r.justification_issues.length > 0) {
                html += '<div class="ai-issues">';
                r.justification_issues.forEach(function(issue) { html += '<div class="ai-issue">\u2022 ' + issue + '</div>'; });
                html += '</div>';
            }
            if (r.rewritten_justification) {
                const isMissing = r.justification_quality === 'missing';
                html += '<div class="ai-rewrite"><label>' + (isMissing ? 'AI Written Justification (original was empty):' : 'AI Suggested Rewrite:') + '</label>';
                html += '<div class="ai-rewrite-text">' + r.rewritten_justification + '</div>';
                html += '<button class="ai-apply-btn" onclick="applyAiRewrite(' + origIdx + ')">' + (isMissing ? 'Apply AI Justification' : 'Apply This') + '</button>';
                html += '</div>';
            }
            html += '</div>';
            return html;
        }

        function applyAiRewrite(origIdx) {
            const r = getAiStepResult(origIdx);
            if (!r || !r.rewritten_justification) return;
            // Find the step by original_index
            const steps = taskData ? taskData.steps || [] : [];
            const stepIdx = steps.findIndex(s => (s.original_index != null ? s.original_index : s.index) === origIdx);
            if (stepIdx < 0) return;
            steps[stepIdx].justification = r.rewritten_justification;
            saveJustification(stepIdx, r.rewritten_justification);

            // Update in-place: find the stacked step card's justification textarea
            const card = document.getElementById('stacked-step-' + stepIdx);
            if (card) {
                const textarea = card.querySelector('.justification-input');
                if (textarea) textarea.value = r.rewritten_justification;
                // Replace the apply button with "Adopted" badge
                const applyBtn = card.querySelector('.ai-apply-btn');
                if (applyBtn) {
                    applyBtn.textContent = 'Adopted';
                    applyBtn.disabled = true;
                    applyBtn.style.background = '#4caf50';
                    applyBtn.style.cursor = 'default';
                }
            }

            // Also update detail overlay if open
            if (document.getElementById('detailOverlay').classList.contains('show') && detailStepIdx === stepIdx) {
                const detailTextarea = document.querySelector('#detailOverlayBody .justification-input');
                if (detailTextarea) detailTextarea.value = r.rewritten_justification;
                const detailBtn = document.querySelector('#detailOverlayBody .ai-apply-btn');
                if (detailBtn) {
                    detailBtn.textContent = 'Adopted';
                    detailBtn.disabled = true;
                    detailBtn.style.background = '#4caf50';
                    detailBtn.style.cursor = 'default';
                }
            }
        }

        // ========== Navigation ==========

        function prevStep() { if (currentStep > 0) selectStep(currentStep - 1); }
        function nextStep() { if (taskData && currentStep < taskData.steps.length - 1) selectStep(currentStep + 1); }

        function updateReviewUI() {
            document.getElementById('reviewStatus').textContent = reviewStatus;
            document.getElementById('reviewStatus').style.color =
                reviewStatus === 'reviewed' ? '#4caf50' :
                reviewStatus === 'rejected' ? '#f44336' : '#888';
            ['reviewed', 'rejected', 'unreviewed'].forEach(s => {
                const btn = document.getElementById('btn' + s.charAt(0).toUpperCase() + s.slice(1));
                if (btn) btn.classList.toggle('active', reviewStatus === s);
            });
        }

        async function setReview(status) {
            reviewStatus = status;
            updateReviewUI();
            updateRecSidebarStatus();
            try {
                await fetch('/api/oss/review', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ folder_name: folderName, oss_folder: ossFolder, status: status })
                });
            } catch (err) { console.error('Save failed:', err); }
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
            // Esc closes detail overlay
            if (e.key === 'Escape' && document.getElementById('detailOverlay').classList.contains('show')) {
                closeDetail();
                return;
            }
            // Arrow keys in detail overlay navigate steps
            if (document.getElementById('detailOverlay').classList.contains('show')) {
                if (e.key === 'ArrowLeft') detailPrev();
                if (e.key === 'ArrowRight') detailNext();
                return;
            }
            if (e.key === 'ArrowLeft') prevStep();
            if (e.key === 'ArrowRight') nextStep();
            // 'g' toggles grid view
            if (e.key === 'g') setViewMode(viewMode === 'grid' ? 'single' : 'grid');
            if (currentMode === 'annotation') {
                if (e.key === '1') setMark('pass');
                if (e.key === '2') setMark('fail');
                if (e.key === '3') setMark(null);
            }
        });

        // ========== Init ==========
        // Hide page size selector by default (shown when grid mode is selected)
        document.getElementById('pageSizeSelect').style.display = 'none';
        if (!directMode) loadRecordingSidebar();
        loadTask();
    </script>
</body>
</html>
'''

# ============================================================================
# Authentication Routes
# ============================================================================

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Reviewer Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f1a; color: #e0e0e0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .login-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid #333; border-radius: 12px; padding: 40px; max-width: 400px; width: 90%; box-shadow: 0 8px 30px rgba(0,0,0,0.4); }
        .login-card h1 { color: #00d9ff; font-size: 1.4em; text-align: center; margin-bottom: 8px; }
        .login-card .subtitle { color: #888; font-size: 0.85em; text-align: center; margin-bottom: 24px; }
        .form-group { margin-bottom: 16px; }
        .form-group label { display: block; color: #888; font-size: 0.82em; margin-bottom: 4px; font-weight: bold; }
        .form-group input { width: 100%; padding: 10px 14px; border: 1px solid #444; border-radius: 6px; background: #0d1117; color: #e0e0e0; font-size: 0.95em; }
        .form-group input:focus { border-color: #00d9ff; outline: none; }
        .submit-btn { width: 100%; padding: 12px; background: linear-gradient(135deg, #00d9ff 0%, #0088cc 100%); border: none; border-radius: 6px; color: #000; font-weight: bold; font-size: 1em; cursor: pointer; margin-top: 8px; }
        .submit-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(0,217,255,0.3); }
        .error-msg { color: #f44336; text-align: center; margin-top: 12px; font-size: 0.85em; }
        .annotator-link { text-align: center; margin-top: 20px; }
        .annotator-link a { color: #ff9800; font-size: 0.85em; text-decoration: none; }
        .annotator-link a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="login-card">
        <h1>Reviewer Login</h1>
        <div class="subtitle">Enter credentials to access the review dashboard</div>
        <form method="POST">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" placeholder="Username" required autofocus />
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" placeholder="Password" required />
            </div>
            <button type="submit" class="submit-btn">Login</button>
            {% if error %}<div class="error-msg">{{ error }}</div>{% endif %}
        </form>
        <div class="annotator-link">
            <a href="/annotator_login">Annotator? Login here</a>
        </div>
    </div>
</body>
</html>
'''

ANNOTATOR_LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Annotator Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f1a; color: #e0e0e0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .login-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid #333; border-radius: 12px; padding: 40px; max-width: 400px; width: 90%; box-shadow: 0 8px 30px rgba(0,0,0,0.4); }
        .login-card h1 { color: #ff9800; font-size: 1.4em; text-align: center; margin-bottom: 8px; }
        .login-card .subtitle { color: #888; font-size: 0.85em; text-align: center; margin-bottom: 24px; }
        .form-group { margin-bottom: 16px; }
        .form-group label { display: block; color: #888; font-size: 0.82em; margin-bottom: 4px; font-weight: bold; }
        .form-group input { width: 100%; padding: 10px 14px; border: 1px solid #444; border-radius: 6px; background: #0d1117; color: #e0e0e0; font-size: 0.95em; }
        .form-group input:focus { border-color: #ff9800; outline: none; }
        .submit-btn { width: 100%; padding: 12px; background: linear-gradient(135deg, #ff9800 0%, #e68900 100%); border: none; border-radius: 6px; color: #000; font-weight: bold; font-size: 1em; cursor: pointer; margin-top: 8px; }
        .submit-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(255,152,0,0.3); }
        .error-msg { color: #f44336; text-align: center; margin-top: 12px; font-size: 0.85em; }
        .reviewer-link { text-align: center; margin-top: 20px; }
        .reviewer-link a { color: #00d9ff; font-size: 0.85em; text-decoration: none; }
        .reviewer-link a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="login-card">
        <h1>Annotator Login</h1>
        <div class="subtitle">Enter your username to access your tasks</div>
        <form method="POST">
            <div class="form-group">
                <label>OSS Folder</label>
                <input type="text" name="folder" placeholder="e.g. recordings_0303" value="{{ folder }}" required />
            </div>
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" placeholder="Your annotator username" value="{{ username }}" required />
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" placeholder="Password" required />
            </div>
            <button type="submit" class="submit-btn">Login</button>
            {% if error %}<div class="error-msg">{{ error }}</div>{% endif %}
        </form>
        <div class="reviewer-link">
            <a href="/login">Reviewer? Login here</a>
        </div>
    </div>
</body>
</html>
'''


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
    return render_template_string(LOGIN_TEMPLATE, error=error)


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
    return render_template_string(ANNOTATOR_LOGIN_TEMPLATE, error=error, folder=folder, username=username)


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
    return render_template_string(DASHBOARD_TEMPLATE)

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
    return render_template_string(OSS_REVIEW_TEMPLATE, folder_name=folder_name, page_mode='reviewer')


@app.route('/annotator/<path:folder_name>')
def annotator_page(folder_name):
    """Annotator page - allows corrections with AI check. No review decision access."""
    # Require either reviewer or annotator login
    if not session.get('reviewer_logged_in') and not session.get('annotator_username'):
        return redirect('/annotator_login')
    return render_template_string(OSS_REVIEW_TEMPLATE, folder_name=folder_name, page_mode='annotator')


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
    return render_template_string(ANNOTATOR_DASHBOARD_TEMPLATE)


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

ANNOTATOR_DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Annotator Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a14;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 16px 24px;
            border-bottom: 2px solid #ff9800;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .header h1 { font-size: 1.3em; color: #ff9800; }
        .header-form { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        .header-form input {
            padding: 8px 12px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #0d1117;
            color: #e0e0e0;
            font-size: 0.9em;
            width: 220px;
        }
        .header-form input:focus { border-color: #ff9800; outline: none; }
        .header-form button {
            padding: 8px 20px;
            border: none;
            border-radius: 6px;
            background: #ff9800;
            color: #000;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
        }
        .header-form button:hover { background: #ffb74d; }
        .header-form button:disabled { background: #555; color: #888; cursor: not-allowed; }
        .back-link { color: #888; text-decoration: none; font-size: 0.85em; border: 1px solid #555; padding: 5px 12px; border-radius: 4px; }
        .back-link:hover { color: #ff9800; border-color: #ff9800; }
        .content { max-width: 1200px; margin: 20px auto; padding: 0 20px; }
        .summary-bar {
            display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap;
        }
        .summary-stat {
            background: #16213e; border-radius: 8px; padding: 12px 20px;
            border: 1px solid #252542; text-align: center; min-width: 120px;
        }
        .summary-stat .num { font-size: 1.6em; font-weight: bold; color: #00d9ff; }
        .summary-stat .label { font-size: 0.75em; color: #888; margin-top: 2px; }
        .task-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 12px;
        }
        .task-card {
            background: #16213e;
            border: 1px solid #252542;
            border-radius: 8px;
            padding: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .task-card:hover { border-color: #ff9800; transform: translateY(-2px); box-shadow: 0 4px 15px rgba(255,152,0,0.15); }
        .task-card.has-errors { border-left: 3px solid #f44336; }
        .task-card .task-title { color: #e0e0e0; font-weight: bold; font-size: 0.9em; word-break: break-all; }
        .task-card .task-query { color: #888; font-size: 0.8em; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .task-card .task-badges { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
        .task-card .badge {
            padding: 2px 8px; border-radius: 10px; font-size: 0.72em; font-weight: bold;
        }
        .badge.reviewed { background: rgba(76,175,80,0.2); color: #4caf50; }
        .badge.rejected { background: rgba(244,67,54,0.2); color: #f44336; }
        .badge.unreviewed { background: rgba(136,136,136,0.2); color: #888; }
        .badge.errors { background: rgba(244,67,54,0.2); color: #f44336; }
        .badge.ai { background: rgba(156,39,176,0.2); color: #ce93d8; }
        .badge.corrected { background: rgba(33,150,243,0.2); color: #42a5f5; }
        .error-msg { color: #f44336; text-align: center; padding: 20px; }
        .loading { color: #888; text-align: center; padding: 40px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Annotator Dashboard</h1>
        <div class="header-form">
            <input type="text" id="ossFolder" placeholder="OSS Folder (e.g. recordings_0303)" />
            <input type="text" id="username" placeholder="Your username" />
            <button id="loadBtn" onclick="loadTasks()">Load My Tasks</button>
            <button id="refreshBtn" style="background:#555;display:none;" onclick="loadTasks(true)">Refresh</button>
        </div>
        <a class="back-link" href="/logout">Logout</a>
    </div>
    <div class="content">
        <div id="summaryBar" class="summary-bar" style="display:none;"></div>
        <div id="taskGrid" class="task-grid">
            <div class="loading">Enter your OSS folder and username, then click "Load My Tasks"</div>
        </div>
    </div>

    <script>
        // Auto-fill from URL params
        const params = new URLSearchParams(window.location.search);
        if (params.get('folder')) document.getElementById('ossFolder').value = params.get('folder');
        if (params.get('user')) document.getElementById('username').value = params.get('user');

        // Auto-load if both params present
        if (params.get('folder') && params.get('user')) loadTasks();

        async function loadTasks(forceRefresh) {
            const folder = document.getElementById('ossFolder').value.trim();
            const user = document.getElementById('username').value.trim();
            if (!folder || !user) { alert('Both fields required'); return; }

            const btn = document.getElementById('loadBtn');
            btn.disabled = true;
            btn.textContent = 'Loading...';
            document.getElementById('taskGrid').innerHTML = '<div class="loading">Loading your tasks...</div>';

            try {
                const resp = await fetch('/api/oss/annotator_tasks?folder=' + encodeURIComponent(folder) + '&user=' + encodeURIComponent(user) + (forceRefresh ? '&refresh=1' : ''));
                const data = await resp.json();

                if (data.error) {
                    document.getElementById('taskGrid').innerHTML = '<div class="error-msg">' + data.error + '</div>';
                    document.getElementById('summaryBar').style.display = 'none';
                    return;
                }

                const recs = data.recordings || [];
                // Update URL
                history.replaceState(null, '', '/annotator_dashboard?folder=' + encodeURIComponent(folder) + '&user=' + encodeURIComponent(user));

                // Summary
                const total = recs.length;
                const reviewed = recs.filter(r => r.review_status === 'reviewed').length;
                const rejected = recs.filter(r => r.review_status === 'rejected').length;
                const withErrors = recs.filter(r => r.error_count > 0).length;
                const withAi = recs.filter(r => r.has_ai_results).length;
                let summaryHtml = '';
                summaryHtml += '<div class="summary-stat"><div class="num">' + total + '</div><div class="label">Total Tasks</div></div>';
                summaryHtml += '<div class="summary-stat"><div class="num" style="color:#4caf50">' + reviewed + '</div><div class="label">Reviewed</div></div>';
                summaryHtml += '<div class="summary-stat"><div class="num" style="color:#f44336">' + rejected + '</div><div class="label">Rejected</div></div>';
                summaryHtml += '<div class="summary-stat"><div class="num" style="color:#ff9800">' + withErrors + '</div><div class="label">With Errors</div></div>';
                if (withAi > 0) summaryHtml += '<div class="summary-stat"><div class="num" style="color:#ce93d8">' + withAi + '</div><div class="label">AI Checked</div></div>';
                document.getElementById('summaryBar').innerHTML = summaryHtml;
                document.getElementById('summaryBar').style.display = 'flex';

                // Task cards - group by version, show newest first
                function getBase(tid) {
                    const m = tid.match(/^(.+)_(\d{3})$/);
                    return m ? { base: m[1], ver: parseInt(m[2]) } : { base: tid, ver: -1 };
                }
                const groups = {};
                recs.forEach(rec => {
                    const tid = rec.task_name || rec.task_id || rec.folder_name;
                    const p = getBase(tid);
                    if (!groups[p.base]) groups[p.base] = [];
                    groups[p.base].push({ rec, ver: p.ver });
                });

                let gridHtml = '';
                Object.keys(groups).sort().forEach(base => {
                    const vers = groups[base];
                    vers.sort((a, b) => b.ver - a.ver);
                    vers.forEach((v, vi) => {
                        const rec = v.rec;
                        const tid = rec.task_name || rec.task_id || rec.folder_name;
                        const status = rec.review_status || 'unreviewed';
                        const hasErr = rec.error_count > 0;
                        const cardUrl = '/annotator/' + encodeURIComponent(rec.folder_name) + '?folder=' + encodeURIComponent(folder) + '&direct=1';
                        gridHtml += '<div class="task-card' + (hasErr ? ' has-errors' : '') + (vi > 0 ? ' older-version' : '') + '" onclick="window.location.href=\\'' + cardUrl + '\\'">';
                        gridHtml += '<div class="task-title">' + tid + '</div>';
                        if (rec.query) gridHtml += '<div class="task-query">' + rec.query + '</div>';
                        gridHtml += '<div class="task-badges">';
                        gridHtml += '<span class="badge ' + status + '">' + status.charAt(0).toUpperCase() + status.slice(1) + '</span>';
                        if (hasErr) gridHtml += '<span class="badge errors">' + rec.error_count + ' errors</span>';
                        if (rec.has_ai_results) gridHtml += '<span class="badge ai">AI Checked</span>';
                        if (rec.has_corrections) gridHtml += '<span class="badge corrected">Has Edits</span>';
                        if (vi > 0) gridHtml += '<span class="badge unreviewed">Older Version</span>';
                        gridHtml += '</div></div>';
                    });
                });

                document.getElementById('taskGrid').innerHTML = gridHtml || '<div class="loading">No tasks found for this user</div>';
            } catch (err) {
                document.getElementById('taskGrid').innerHTML = '<div class="error-msg">Failed to load: ' + err.message + '</div>';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Load My Tasks';
                document.getElementById('refreshBtn').style.display = 'inline-block';
            }
        }
    </script>
</body>
</html>
'''

DIRECT_ACCESS_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Annotator Entrance</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .access-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #333;
            border-radius: 12px;
            padding: 32px 40px;
            max-width: 480px;
            width: 90%;
            box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        }
        .access-card h1 {
            color: #00d9ff;
            font-size: 1.4em;
            margin-bottom: 6px;
            text-align: center;
        }
        .access-card .subtitle {
            color: #888;
            font-size: 0.85em;
            text-align: center;
            margin-bottom: 24px;
        }
        .form-group {
            margin-bottom: 16px;
        }
        .form-group label {
            display: block;
            color: #888;
            font-size: 0.82em;
            margin-bottom: 4px;
            font-weight: bold;
        }
        .form-group input {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #0d1117;
            color: #e0e0e0;
            font-size: 0.95em;
            transition: border-color 0.2s;
        }
        .form-group input:focus {
            border-color: #00d9ff;
            outline: none;
        }
        .form-group input::placeholder { color: #555; }
        .form-group .hint {
            color: #666;
            font-size: 0.72em;
            margin-top: 3px;
        }
        .submit-btn {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #00d9ff 0%, #0088cc 100%);
            border: none;
            border-radius: 6px;
            color: #000;
            font-weight: bold;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.2s;
            margin-top: 8px;
        }
        .submit-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(0,217,255,0.3); }
        .submit-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .error-msg {
            color: #f44336;
            font-size: 0.85em;
            text-align: center;
            margin-top: 12px;
            display: none;
        }
        .info-box {
            background: #0d1117;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 12px 14px;
            margin-top: 20px;
            font-size: 0.8em;
            color: #888;
            line-height: 1.5;
        }
        .info-box b { color: #ffc107; }
    </style>
</head>
<body>
    <div class="access-card">
        <h1 style="color:#ff9800;">Annotator Entrance</h1>
        <div class="subtitle">Enter your folder, task ID and username to view reviewer feedback and correct errors</div>

        <div class="form-group">
            <label>OSS Folder</label>
            <input type="text" id="ossFolder" placeholder="e.g. recordings_0303" value="{{ default_folder }}" />
            <div class="hint">The OSS recording folder name</div>
        </div>

        <div class="form-group">
            <label>Task ID</label>
            <input type="text" id="taskId" placeholder="e.g. Annotater07_vlc_scenario_fc4f43fb9122_22833104_000" value="{{ task_id }}" />
            <div class="hint">The task ID shown in the review page (e.g. Annotater07_vlc_scenario_...)</div>
        </div>

        <div class="form-group">
            <label>Username</label>
            <input type="text" id="username" placeholder="Your annotator username" value="{{ username }}" />
            <div class="hint">Must match the annotator who created this recording</div>
        </div>

        <button class="submit-btn" id="submitBtn" onclick="openCase()">Open Case</button>
        <div class="error-msg" id="errorMsg"></div>

        <div class="info-box">
            <b>Note:</b> This page is for annotators to view reviewer feedback and correct errors in their recordings.
            You can only access cases assigned to your username. Steps marked as errors by the reviewer
            will be highlighted — fix them and your changes are saved automatically.
            <br><br>
            <b>Want to see all your tasks?</b> <a href="/annotator_dashboard" style="color:#ff9800;">Open Annotator Dashboard</a>
        </div>
    </div>

    <script>
        // Auto-fill from URL params
        const params = new URLSearchParams(window.location.search);
        if (params.get('folder')) document.getElementById('ossFolder').value = params.get('folder');
        if (params.get('task')) document.getElementById('taskId').value = params.get('task');
        if (params.get('user')) document.getElementById('username').value = params.get('user');

        async function openCase() {
            const folder = document.getElementById('ossFolder').value.trim();
            const taskId = document.getElementById('taskId').value.trim();
            const username = document.getElementById('username').value.trim();
            const errEl = document.getElementById('errorMsg');

            if (!folder || !taskId || !username) {
                errEl.textContent = 'All fields are required.';
                errEl.style.display = 'block';
                return;
            }

            const btn = document.getElementById('submitBtn');
            btn.disabled = true;
            btn.textContent = 'Verifying...';
            errEl.style.display = 'none';

            try {
                // Verify the case exists and username matches
                const resp = await fetch('/api/oss/verify_access?folder=' + encodeURIComponent(folder) +
                    '&task_id=' + encodeURIComponent(taskId) +
                    '&user=' + encodeURIComponent(username));
                const data = await resp.json();

                if (data.error) {
                    errEl.textContent = data.error;
                    errEl.style.display = 'block';
                    btn.disabled = false;
                    btn.textContent = 'Open Case';
                    return;
                }

                if (!data.access_granted) {
                    errEl.textContent = 'Access denied. Username does not match the recording annotator.';
                    errEl.style.display = 'block';
                    btn.disabled = false;
                    btn.textContent = 'Open Case';
                    return;
                }

                // Redirect to the annotator page using the resolved folder_name
                const folderName = data.folder_name;
                window.location.href = '/annotator/' + encodeURIComponent(folderName) +
                    '?folder=' + encodeURIComponent(folder) + '&direct=1';
            } catch (err) {
                errEl.textContent = 'Connection error: ' + err.message;
                errEl.style.display = 'block';
                btn.disabled = false;
                btn.textContent = 'Open Case';
            }
        }

        // Submit on Enter key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') openCase();
        });
    </script>
</body>
</html>
'''


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
