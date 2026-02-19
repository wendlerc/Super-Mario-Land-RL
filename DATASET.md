# Mario Dataset Documentation

## Dataset Overview

Total annotated gameplay data for Super Mario Land (Game Boy).

| Dataset | Location | Size | Episodes | Description |
|---------|----------|------|----------|-------------|
| `mario_long_episodes/` | `/data/workspace/mario_long_episodes/` | 1.1GB | 290 | Random agent, single level |
| `mario_multi_level/` | `/data/workspace/mario_multi_level/` | ~6MB | 6 | Random agent, multi-level (generating) |
| `mario_dataset/` | `/data/workspace/mario_dataset/` | 198MB | 15 | 5-minute chunks |
| **TOTAL** | | **~1.3GB** | **~310** | |

## Generation Status

**Currently generating:** 10GB target with `nohup` (background processes)
- `generate_multi_level.py` — Multi-level variety (Worlds 0-3, Levels 0-2)
- `record_long_episodes.py` — Long episodes until death

**Check progress:**
```bash
# Episode counts
ls /data/workspace/mario_multi_level/*.mp4 | wc -l
ls /data/workspace/mario_long_episodes/*.mp4 | wc -l

# Size
du -sh /data/workspace/mario_multi_level/
du -sh /data/workspace/mario_long_episodes/

# Logs
tail -f /data/workspace/mario_dataset.log
tail -f /data/workspace/mario_long.log
```

## Annotation Format

Each episode includes:

```json
{
  "metadata": {
    "episode": 0,
    "start_world": 3,
    "start_level": 2,
    "fps": 30,
    "total_frames": 246,
    "duration_sec": 8.2,
    "final_progress": 239,
    "final_score": 0,
    "completed": false
  },
  "frames": [
    {
      "frame": 0,
      "action": {"index": 1, "name": "A"},
      "reward": 2439.9,
      "state": {
        "world": 3,
        "level": 2,
        "progress": 244,
        "score": 0,
        "lives": 2,
        "time_left": 400
      },
      "done": false
    }
  ]
}
```

## Actions

| Index | Name | Description |
|-------|------|-------------|
| 0 | NOP | No action |
| 1 | A | Jump button |
| 2 | B | Run/fire button |
| 3 | UP | D-pad up |
| 4 | DOWN | D-pad down |
| 5 | LEFT | D-pad left |
| 6 | RIGHT | D-pad right |
| 7 | JUMP_RIGHT | Right + Jump |
| 8 | JUMP_LEFT | Left + Jump |

## Resume Capability

All scripts support resuming interrupted generation:
- Episodes are skipped if already exist
- Graceful shutdown on interrupt
- Safe to restart anytime

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_multi_level.py` | Multi-level dataset with resume |
| `record_long_episodes.py` | Long single-level episodes |
| `batch_record.py` | 5-minute chunk dataset |
| `record_annotated.py` | Single annotated episode |

## Background Processes

Currently running with `nohup`:
```bash
# Check if running
ps aux | grep "generate_multi_level\|record_long_episodes"

# Stop gracefully
pkill -f generate_multi_level
pkill -f record_long_episodes
```

## Target

**Goal:** 10GB annotated gameplay data
**Current:** ~1.3GB (13%)
**ETA:** ~12-15 hours at current rate

## Use Cases

- Behavioral cloning
- Imitation learning
- World models
- Action prediction
- Reinforcement learning research
