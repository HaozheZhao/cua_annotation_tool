# CUA Annotation Tool

A web-based human verification system for CUA-SFT (Computer Use Agent - Supervised Fine-Tuning) data annotation.

## Features

- **Task Navigation**: Browse through tasks with sidebar navigation
- **Step-by-step Review**: View each action step with screenshots and coordinates
- **Coordinate Fine-tuning**: Adjust click coordinates with real-time marker preview
- **Case Evaluation**: Rate cases on Correctness, Difficulty, Knowledge Richness, and Task Value
- **Knowledge Points**: Tag OSWorld overlap, custom nodes, and related apps
- **Pass/Fail Marking**: Mark cases as Pass, Fail, or Unclear with justification
- **Progress Tracking**: Visual progress bar showing annotation status
- **Export**: Export passed cases with screenshots and JSON metadata

## Installation

```bash
pip install flask opencv-python oss2
```

## Quick Start

```bash
python app.py --data /path/to/your/data --csv /path/to/your/tasks.csv
```

Then open http://localhost:5000 in your browser.

---

## Data Setup

### 1. Data Folder Structure

Your data folder should contain **numbered subfolders** (one per task):

```
data/
├── 1/
│   ├── metadata.json
│   ├── reduced_events_complete.jsonl
│   ├── reduced_events_vis.jsonl (optional)
│   └── recording.mp4
├── 2/
│   ├── metadata.json
│   ├── reduced_events_complete.jsonl
│   └── recording.mp4
├── 3/
│   └── ...
└── ...
```

#### Required files per task folder:

| File | Description |
|------|-------------|
| `metadata.json` | Contains video start timestamp |
| `reduced_events_complete.jsonl` | Event details (one JSON per line) |
| `*.mp4` | Screen recording video |

**metadata.json example:**
```json
{
  "video_start_timestamp": 1706123456.789
}
```

**reduced_events_complete.jsonl example:**
```json
{"action": "click", "coordinate": {"x": 500, "y": 300}, "start_time": 1706123460.0, "end_time": 1706123461.0, "pre_move": {"start_time": 1706123459.0, "end_time": 1706123460.0}, "justification": "Click on submit button"}
{"action": "type", "start_time": 1706123462.0, "end_time": 1706123465.0, "justification": "Type search query"}
```

### 2. CSV File Structure

Create a CSV file with task assignments:

```csv
task_id,instruction,worker_id,worker_name,osworld_overlap,custom_nodes,related_apps
1,Search for weather in New York,001,Alice,weather_search,search_node,Browser;Weather App
2,Open calculator and compute 25*4,002,Bob,,calculator_node,Calculator
3,Send an email to test@example.com,001,Alice,email_task,compose_email,Email Client
```

| Column | Required | Description |
|--------|----------|-------------|
| `task_id` | Yes | Must match folder name in data directory |
| `instruction` | Yes | Task instruction/description |
| `worker_id` | No | Worker identifier |
| `worker_name` | No | Worker name for display |
| `osworld_overlap` | No | Comma-separated tags for OSWorld overlap |
| `custom_nodes` | No | Comma-separated custom node tags |
| `related_apps` | No | Semicolon or comma-separated related applications |

---

## Usage

### Command Line Options

```bash
python app.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | `./data` | Path to data folder containing task subfolders |
| `--csv` | `./task_assignments.csv` | Path to CSV file with task assignments |
| `--output` | `./output` | Path where exports will be saved |
| `--oss-cache` | `./oss_cache` | Directory for cached OSS recordings |
| `--port` | `5000` | Server port number |
| `--host` | `0.0.0.0` | Server host |

### Examples

```bash
# Basic usage with default paths
python app.py

# Specify custom paths
python app.py --data /home/user/my_data --csv /home/user/tasks.csv

# Full options
python app.py \
  --data /path/to/data \
  --csv /path/to/task_assignments.csv \
  --output /path/to/output \
  --port 8080
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` | Previous step |
| `→` | Next step |
| `1` | Mark as Pass |
| `2` | Mark as Fail |
| `3` | Mark as Unclear |

---

## Export Format

Exported JSON for each passed case:

```json
{
  "task_id": 1,
  "instruction": "Search for weather in New York",
  "step_by_step_instructions": "1. Open browser\n2. Navigate to weather.com\n3. Search for New York",
  "knowledge_points": {
    "osworld_overlap": ["weather_search"],
    "custom_nodes": ["search_node"]
  },
  "related_apps": ["Browser", "Weather App"],
  "traj": [
    {
      "index": 0,
      "action_type": "click",
      "code": "pyautogui.click(500, 300)",
      "screenshot": "step_0.png"
    },
    {
      "index": 1,
      "action_type": "type",
      "code": "pyautogui.typewrite(\"New York\")",
      "screenshot": "step_1.png"
    }
  ]
}
```

## Output Files

After export, the output directory contains:

```
output/
├── annotations.json              # All annotation data
├── coordinate_adjustments.json   # Fine-tuned coordinates
├── all_tasks.json               # Combined export of all passed tasks
├── export.zip                   # ZIP archive of everything
└── task_1/                      # Per-task folders
    ├── task_1.json
    ├── step_0.png
    ├── step_1.png
    └── ...
```

---

## OSS Dashboard (Real-Time Viewing)

The tool includes an OSS Dashboard for reviewing recordings uploaded from AgentNet-Tool in real-time.

### Setup

Set environment variables for OSS access:

```bash
export OSS_ACCESS_KEY_ID=your_key_id
export OSS_ACCESS_KEY_SECRET=your_key_secret
export OSS_BUCKET_NAME=your_bucket_name
export OSS_ENDPOINT=oss-cn-shanghai.aliyuncs.com
```

### Usage

1. Start the server and navigate to `/dashboard`
2. Enter the OSS upload folder name (e.g., `recordings_new`)
3. Click **Load** to fetch recordings from OSS
4. View per-annotator summary cards with review status counts
5. Click an annotator card to expand their task list
6. Click **View** to review individual recordings
7. Mark recordings as **Reviewed**, **Rejected**, or clear the status

### OSS Dashboard Routes

| Route | Description |
|-------|-------------|
| `/dashboard` | Dashboard page with annotator overview |
| `/oss_review/<folder_name>` | Review page for a single OSS recording |
| `/api/oss/list?folder=X` | List recordings from OSS folder |
| `/api/oss/dashboard_data?folder=X` | Aggregated per-annotator statistics |
| `/api/oss/task/<folder>?folder=X` | Load recording data for review |
| `/oss_frame/<folder>/<time>?folder=X` | Extract video frame |
| `POST /api/oss/review` | Save review status |

---

## License

MIT
