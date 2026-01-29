# CUA Annotation Tool

A web-based human verification system for CUA-SFT (Computer Use Agent - Supervised Fine-Tuning) data annotation.

## Features

- **Task Navigation**: Browse through tasks with sidebar navigation
- **Step-by-step Review**: View each action step with screenshots and coordinates
- **Coordinate Fine-tuning**: Adjust click coordinates with real-time marker preview
- **Case Evaluation**: Rate cases on Correctness, Difficulty, Knowledge Richness, and Task Value
- **Pass/Fail Marking**: Mark cases as Pass, Fail, or Unclear with justification
- **Progress Tracking**: Visual progress bar showing annotation status
- **Export**: Export passed cases with screenshots and JSON metadata

## Installation

```bash
pip install flask opencv-python
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
task_id,instruction,worker_id,worker_name
1,Search for weather in New York,001,Alice
2,Open calculator and compute 25*4,002,Bob
3,Send an email to test@example.com,001,Alice
```

| Column | Required | Description |
|--------|----------|-------------|
| `task_id` | Yes | Must match folder name in data directory |
| `instruction` | Yes | Task instruction/description |
| `worker_id` | No | Worker identifier |
| `worker_name` | No | Worker name for display |

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

Exported JSON for each passed case includes annotator evaluation:

```json
{
  "task_id": 1,
  "instruction": "Search for weather in New York",
  "pass_reason": "Step-by-step aligns with screenshots; actions valid; final state verified.",
  "scores": {
    "correctness": 5,
    "difficulty": 3,
    "knowledge_richness": 4,
    "task_value": 4
  },
  "traj": [
    {
      "index": 0,
      "code": "pyautogui.click(500, 300)",
      "screenshot": "task_1/step_0.png",
      "justification": "Click on search box"
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

## License

MIT
