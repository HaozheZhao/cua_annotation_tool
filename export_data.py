"""
Export recording data from OSS recordings_0303 folder.
Creates two outputs:
  1. export_text_only/ - All text data (no images), zipped
  2. export_full/      - All data including screenshots, zipped
"""

import os
import sys
import json
import re
import csv
import shutil
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
import oss_client

try:
    import cv2
except ImportError:
    print("WARNING: cv2 not available - screenshots will be skipped")
    cv2 = None

OSS_FOLDER = "recordings_0303"
EXPORT_DIR = Path(__file__).parent / "export"
TEXT_DIR = EXPORT_DIR / "export_text_only"
FULL_DIR = EXPORT_DIR / "export_full"
CACHE_DIR = Path(__file__).parent / "oss_cache"


def download_recording(folder_name):
    """Download all metadata + video for a recording."""
    prefix = f"{OSS_FOLDER}/{folder_name}"
    local_dir = CACHE_DIR / folder_name
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download metadata files
    oss_client.download_recording_metadata_files(prefix, str(local_dir))

    # Download video
    oss_client.download_video(prefix, str(local_dir))

    return local_dir


def extract_text_data(local_dir, folder_name):
    """Extract all text data from a recording into a structured dict."""
    local_dir = Path(local_dir)

    # Load annotator info
    annotator_info = {}
    ai_file = local_dir / "annotator_info.json"
    if ai_file.exists():
        with open(ai_file) as f:
            annotator_info = json.load(f)

    # Load metadata
    metadata = {}
    meta_file = local_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            metadata = json.load(f)

    # Load task name
    task_name = ""
    tn_file = local_dir / "task_name.json"
    if tn_file.exists():
        with open(tn_file) as f:
            tn_data = json.load(f)
            task_name = tn_data.get("task_name", "")

    # Load knowledge points
    knowledge_points = []
    kp_file = local_dir / "knowledge_points.json"
    if kp_file.exists():
        with open(kp_file) as f:
            kp_data = json.load(f)
            if isinstance(kp_data, list):
                knowledge_points = kp_data

    # Load events
    complete_events = []
    complete_file = local_dir / "reduced_events_complete.jsonl"
    if complete_file.exists():
        with open(complete_file) as f:
            complete_events = [json.loads(line) for line in f if line.strip()]

    vis_events = []
    vis_file = local_dir / "reduced_events_vis.jsonl"
    if vis_file.exists():
        with open(vis_file) as f:
            vis_events = [json.loads(line) for line in f if line.strip()]

    # Build operations list
    video_start_ts = metadata.get("video_start_timestamp", 0)
    operations = []
    for i, ce in enumerate(complete_events):
        ve = vis_events[i] if i < len(vis_events) else {}
        coord = ce.get("coordinate", {})
        x, y = coord.get("x", 0), coord.get("y", 0)
        action = ce.get("action", "")
        description = ve.get("description", ce.get("description", ""))
        justification = ce.get("justification", "")

        # Compute video time for screenshot extraction
        start_time = ce.get("start_time", 0)
        pre_move = ce.get("pre_move", {})
        if pre_move and pre_move.get("start_time") and pre_move.get("end_time"):
            pm_start = pre_move["start_time"]
            pm_end = pre_move["end_time"]
            capture_time = pm_start + (pm_end - pm_start) * 0.8
        elif pre_move and pre_move.get("end_time"):
            capture_time = max(0, pre_move["end_time"] - 0.3)
        else:
            capture_time = max(0, start_time - 0.1)
        video_time = capture_time - video_start_ts

        # Parse drag coordinates
        drag_to = None
        if action == "drag":
            m = re.search(r"Drag from \((\d+),\s*(\d+)\) to \((\d+),\s*(\d+)\)", description)
            if m:
                drag_to = {"x": int(m.group(3)), "y": int(m.group(4))}

        op = {
            "step_index": i,
            "action": action,
            "description": description,
            "justification": justification,
            "coordinate": {"x": x, "y": y},
            "video_time_sec": round(video_time, 3),
        }

        # Action-specific fields
        if action == "click" or action == "mouse_press":
            op["click_type"] = ce.get("click_type", 1)  # 1=single, 2=double
            op["button"] = ce.get("button", "left")
        elif action == "drag":
            if drag_to:
                op["drag_to"] = drag_to
            op["click_type"] = ce.get("click_type", 1)
            op["button"] = ce.get("button", "left")
            # Include drag trace (list of [x,y] points)
            if ce.get("drag_trace"):
                op["drag_trace"] = ce["drag_trace"]
        elif action == "type":
            op["typed_text"] = ce.get("resolved_text", ce.get("raw_text", ""))
            op["raw_text"] = ce.get("raw_text", "")
            op["key_names"] = ce.get("key_names", [])
            op["has_editing"] = ce.get("has_editing", False)
            if ce.get("resolved_description"):
                op["resolved_description"] = ce["resolved_description"]
        elif action == "press" or action == "long_press":
            op["key_name"] = ce.get("key_name", "")
            # Reconstruct full key combo from description (e.g. "Press: $ctrl$ + c")
            desc_text = ce.get("description", "")
            key_combo = desc_text.replace("⌨️ Press: ", "").replace("⌨️ Long Press: ", "").strip()
            if key_combo:
                op["key_combo"] = key_combo
        elif action == "scroll":
            if ce.get("trace"):
                trace = ce["trace"]
                # Extract scroll direction and amount
                total_dx = sum(t.get("dx", 0) for t in trace)
                total_dy = sum(t.get("dy", 0) for t in trace)
                op["scroll_dx"] = total_dx
                op["scroll_dy"] = total_dy
                op["scroll_steps"] = len(trace)
                # Include scroll position (from first trace point)
                if trace:
                    op["coordinate"] = {"x": trace[0].get("x", 0), "y": trace[0].get("y", 0)}

        operations.append(op)

    return {
        "folder_name": folder_name,
        "task_name": task_name,
        "annotator": annotator_info.get("username", "Unknown"),
        "task_id": annotator_info.get("task_id", ""),
        "query": annotator_info.get("query", ""),
        "step_by_step_instruction": annotator_info.get("step_by_step_instruction", ""),
        "upload_timestamp": annotator_info.get("upload_timestamp", ""),
        "knowledge_points": knowledge_points,
        "video_resolution": {
            "width": metadata.get("video_width", 1920),
            "height": metadata.get("video_height", 1080),
        },
        "total_steps": len(operations),
        "operations": operations,
    }


def extract_screenshots(local_dir, operations, output_dir):
    """Extract screenshot frames from video for each operation."""
    local_dir = Path(local_dir)
    output_dir = Path(output_dir)

    if cv2 is None:
        return

    # Find video
    video_path = None
    for f in local_dir.glob("*.mp4"):
        if "video_clips" not in str(f):
            video_path = f
            break

    if not video_path:
        return

    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    for op in operations:
        video_time = op["video_time_sec"]
        frame_num = int(video_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            filename = f"step_{op['step_index']:03d}_{op['action']}.jpg"
            cv2.imwrite(str(screenshots_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    cap.release()


CSV_FIELDNAMES = [
    "step_index", "action", "description", "justification",
    "coordinate_x", "coordinate_y",
    # click/drag
    "click_type", "button",
    # drag
    "drag_to_x", "drag_to_y",
    # type
    "typed_text", "raw_text", "has_editing",
    # press/long_press
    "key_combo",
    # scroll
    "scroll_dx", "scroll_dy", "scroll_steps",
    # common
    "video_time_sec",
]


def write_operations_csv(csv_path, operations, include_screenshot=False):
    """Write operations to CSV with all action-specific fields."""
    fieldnames = list(CSV_FIELDNAMES)
    if include_screenshot:
        fieldnames.append("screenshot_path")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for op in operations:
            row = {
                "step_index": op["step_index"],
                "action": op["action"],
                "description": op["description"],
                "justification": op["justification"],
                "coordinate_x": op["coordinate"]["x"],
                "coordinate_y": op["coordinate"]["y"],
                "video_time_sec": op["video_time_sec"],
                # click/drag fields
                "click_type": op.get("click_type", ""),
                "button": op.get("button", ""),
                # drag
                "drag_to_x": op["drag_to"]["x"] if op.get("drag_to") else "",
                "drag_to_y": op["drag_to"]["y"] if op.get("drag_to") else "",
                # type
                "typed_text": op.get("typed_text", ""),
                "raw_text": op.get("raw_text", ""),
                "has_editing": op.get("has_editing", ""),
                # press/long_press
                "key_combo": op.get("key_combo", ""),
                # scroll
                "scroll_dx": op.get("scroll_dx", ""),
                "scroll_dy": op.get("scroll_dy", ""),
                "scroll_steps": op.get("scroll_steps", ""),
            }
            if include_screenshot:
                row["screenshot_path"] = op.get("screenshot_path", "")
            writer.writerow(row)


def make_zip(source_dir, zip_path):
    """Create a zip file from a directory."""
    source_dir = Path(source_dir)
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(source_dir.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(source_dir.parent))


def main():
    print(f"Listing recordings in {OSS_FOLDER}...")
    recordings = oss_client.list_recordings(OSS_FOLDER)
    total = len(recordings)
    print(f"Found {total} recordings.\n")

    if total == 0:
        print("No recordings found. Check OSS credentials and folder name.")
        return

    # Clean export dirs
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    TEXT_DIR.mkdir(parents=True)
    FULL_DIR.mkdir(parents=True)

    all_summary = []

    for idx, folder_name in enumerate(recordings):
        print(f"[{idx+1}/{total}] Processing: {folder_name}")

        # Download
        try:
            local_dir = download_recording(folder_name)
        except Exception as e:
            print(f"  ERROR downloading: {e}")
            continue

        # Extract text data
        try:
            data = extract_text_data(local_dir, folder_name)
        except Exception as e:
            print(f"  ERROR extracting text: {e}")
            continue

        # --- Text-only export ---
        rec_text_dir = TEXT_DIR / folder_name
        rec_text_dir.mkdir(parents=True, exist_ok=True)

        # Save full recording data as JSON
        with open(rec_text_dir / "recording_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save operations as CSV
        if data["operations"]:
            write_operations_csv(rec_text_dir / "operations.csv", data["operations"])

        # Save raw JSON files as-is
        for src_name in ["annotator_info.json", "knowledge_points.json", "metadata.json", "task_name.json"]:
            src = local_dir / src_name
            if src.exists():
                shutil.copy2(src, rec_text_dir / src_name)

        # Copy raw event files
        for src_name in ["reduced_events_complete.jsonl", "reduced_events_vis.jsonl"]:
            src = local_dir / src_name
            if src.exists():
                shutil.copy2(src, rec_text_dir / src_name)

        # --- Full export (text + screenshots) ---
        rec_full_dir = FULL_DIR / folder_name
        # Copy text dir contents
        shutil.copytree(rec_text_dir, rec_full_dir)

        # Extract screenshots and add paths to data
        try:
            extract_screenshots(local_dir, data["operations"], rec_full_dir)
            n_screenshots = 0
            # Add screenshot relative path to each operation
            full_data = json.loads(json.dumps(data))  # deep copy
            for op in full_data["operations"]:
                filename = f"step_{op['step_index']:03d}_{op['action']}.jpg"
                screenshot_file = rec_full_dir / "screenshots" / filename
                if screenshot_file.exists():
                    op["screenshot_path"] = f"screenshots/{filename}"
                    n_screenshots += 1
                else:
                    op["screenshot_path"] = ""
            # Overwrite recording_data.json with screenshot paths
            with open(rec_full_dir / "recording_data.json", "w", encoding="utf-8") as f:
                json.dump(full_data, f, indent=2, ensure_ascii=False)
            # Overwrite operations.csv with screenshot_path column
            if full_data["operations"]:
                write_operations_csv(rec_full_dir / "operations.csv", full_data["operations"], include_screenshot=True)
            print(f"  {data['total_steps']} steps, {n_screenshots} screenshots, annotator={data['annotator']}")
        except Exception as e:
            print(f"  Screenshots error: {e}, text data OK")

        # Summary entry
        all_summary.append({
            "folder_name": folder_name,
            "annotator": data["annotator"],
            "task_id": data["task_id"],
            "task_name": data["task_name"],
            "query": data["query"],
            "step_by_step_instruction": data["step_by_step_instruction"],
            "knowledge_points": data["knowledge_points"],
            "total_steps": data["total_steps"],
            "upload_timestamp": data["upload_timestamp"],
        })

    # Write summary files
    print(f"\nWriting summary files...")

    # Summary JSON
    for d in [TEXT_DIR, FULL_DIR]:
        with open(d / "summary.json", "w", encoding="utf-8") as f:
            json.dump(all_summary, f, indent=2, ensure_ascii=False)

    # Summary CSV
    if all_summary:
        summary_fields = ["folder_name", "annotator", "task_id", "task_name", "query",
                         "step_by_step_instruction", "knowledge_points", "total_steps", "upload_timestamp"]
        for d in [TEXT_DIR, FULL_DIR]:
            with open(d / "summary.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=summary_fields)
                writer.writeheader()
                for entry in all_summary:
                    row = dict(entry)
                    row["knowledge_points"] = "; ".join(row["knowledge_points"])
                    writer.writerow(row)

    # Create zip files
    print("Creating zip files...")
    text_zip = EXPORT_DIR / "export_text_only.zip"
    full_zip = EXPORT_DIR / "export_full.zip"

    make_zip(TEXT_DIR, text_zip)
    print(f"  Text-only: {text_zip} ({text_zip.stat().st_size / 1024 / 1024:.1f} MB)")

    make_zip(FULL_DIR, full_zip)
    print(f"  Full data: {full_zip} ({full_zip.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\nDone! {len(all_summary)} recordings exported.")
    print(f"  Text-only zip: {text_zip}")
    print(f"  Full zip:      {full_zip}")


if __name__ == "__main__":
    main()
