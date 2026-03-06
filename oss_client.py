"""
OSS Client wrapper for cua_annotation_tool.
Provides functions to list, download, and stream recordings from Alibaba Cloud OSS.
"""

import os
import json
import time
import threading
from pathlib import Path

import oss2

# OSS Configuration - uses environment variables with defaults
OSS_ACCESS_KEY_ID = os.environ.get("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
OSS_BUCKET_NAME = os.environ.get("OSS_BUCKET_NAME", "")
OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT", "oss-cn-shanghai.aliyuncs.com")

# Cache settings
_cache = {}
_cache_lock = threading.Lock()
CACHE_TTL = 60  # seconds


def _get_bucket():
    """Get an authenticated OSS bucket instance."""
    auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
    return oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)


def _cache_get(key):
    """Get a cached value if not expired."""
    with _cache_lock:
        if key in _cache:
            value, ts = _cache[key]
            if time.time() - ts < CACHE_TTL:
                return value
            del _cache[key]
    return None


def _cache_set(key, value):
    """Set a cached value with current timestamp."""
    with _cache_lock:
        _cache[key] = (value, time.time())


def clear_cache():
    """Clear the entire cache."""
    with _cache_lock:
        _cache.clear()


def list_recordings(folder="recordings_new"):
    """
    List recording folders under a given OSS prefix.
    Returns a list of folder names (immediate subdirectories).
    Results are cached for CACHE_TTL seconds.
    """
    cache_key = f"list:{folder}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    bucket = _get_bucket()
    prefix = folder.rstrip("/") + "/"
    folders = set()

    # List objects with delimiter to get "directories"
    for obj in oss2.ObjectIteratorV2(bucket, prefix=prefix, delimiter="/"):
        if obj.is_prefix():
            # Extract folder name from prefix
            name = obj.key[len(prefix):].rstrip("/")
            if name:
                folders.add(name)

    result = sorted(folders)
    _cache_set(cache_key, result)
    return result


def get_recording_metadata(prefix):
    """
    Download and parse annotator_info.json from a recording folder.
    Returns the parsed JSON dict, or None if not found.
    """
    bucket = _get_bucket()
    key = prefix.rstrip("/") + "/annotator_info.json"

    try:
        result = bucket.get_object(key)
        content = result.read().decode("utf-8")
        return json.loads(content)
    except oss2.exceptions.NoSuchKey:
        return None
    except Exception:
        return None


def download_recording_metadata_files(prefix, local_dir):
    """
    Download JSONL/JSON metadata files from a recording to a local directory.
    Downloads: reduced_events_complete.jsonl, reduced_events_vis.jsonl, metadata.json,
    task_name.json, annotator_info.json
    Returns dict of {filename: local_path} for successfully downloaded files.
    """
    bucket = _get_bucket()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    target_files = [
        "reduced_events_complete.jsonl",
        "reduced_events_vis.jsonl",
        "metadata.json",
        "task_name.json",
        "annotator_info.json",
        "knowledge_points.json",
    ]

    downloaded = {}
    remote_prefix = prefix.rstrip("/") + "/"

    for filename in target_files:
        remote_key = remote_prefix + filename
        local_path = local_dir / filename

        # Skip if already downloaded
        if local_path.exists():
            downloaded[filename] = str(local_path)
            continue

        try:
            result = bucket.get_object(remote_key)
            with open(local_path, "wb") as f:
                for chunk in result:
                    f.write(chunk)
            downloaded[filename] = str(local_path)
        except oss2.exceptions.NoSuchKey:
            continue
        except Exception:
            continue

    return downloaded


def download_video(prefix, local_dir):
    """
    Download the full_video.mp4 (or any .mp4 file) from a recording.
    Returns the local path to the downloaded video, or None if not found.
    This is a lazy download - skips if already cached locally.
    """
    bucket = _get_bucket()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_prefix = prefix.rstrip("/") + "/"

    # First check if video already exists locally
    for f in local_dir.glob("*.mp4"):
        return str(f)

    # Find .mp4 files in the remote recording
    video_key = None
    for obj in oss2.ObjectIteratorV2(bucket, prefix=remote_prefix):
        if not obj.is_prefix() and obj.key.endswith(".mp4") and "video_clips" not in obj.key:
            video_key = obj.key
            break

    if not video_key:
        return None

    # Extract filename from key
    filename = video_key.split("/")[-1]
    local_path = local_dir / filename

    if local_path.exists():
        return str(local_path)

    # Download using resumable download for large files
    try:
        oss2.resumable_download(
            bucket,
            video_key,
            str(local_path),
            multiget_threshold=10 * 1024 * 1024,
            num_threads=4,
        )
        return str(local_path)
    except Exception as e:
        # Fallback to simple download
        try:
            result = bucket.get_object(video_key)
            with open(local_path, "wb") as f:
                for chunk in result:
                    f.write(chunk)
            return str(local_path)
        except Exception:
            return None


def get_presigned_url(key, expires=3600):
    """
    Generate a presigned URL for streaming a file directly from OSS.
    """
    bucket = _get_bucket()
    try:
        url = bucket.sign_url("GET", key, expires)
        return url
    except Exception:
        return None


def parse_folder_name_metadata(folder_name):
    """
    Parse metadata from a recording folder name.
    Folder names follow the pattern: {timestamp}_{task_name}_{username}_{recording_id}
    or: {timestamp}_{task_name}_{recording_id}
    Returns a dict with inferred metadata.
    """
    parts = folder_name.split("_")

    metadata = {
        "username": "Unknown",
        "task_id": "",
        "query": "",
        "upload_timestamp": "",
        "oss_upload_folder": "",
        "folder_name": folder_name,
    }

    if len(parts) >= 1:
        # First part is typically a timestamp like "20250101-120000"
        metadata["upload_timestamp"] = parts[0]

    if len(parts) >= 2:
        # Second part is the task name
        metadata["task_id"] = parts[1]

    # Try to detect username vs recording_id
    # Recording IDs typically look like "recording_XXXXXX" or numeric
    if len(parts) >= 4:
        # Pattern: timestamp_taskname_username_recordingid
        metadata["username"] = parts[2]
    elif len(parts) == 3:
        # Pattern: timestamp_taskname_recordingid (no username)
        metadata["username"] = "Unknown"

    return metadata
