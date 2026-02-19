#!/usr/bin/env python3
"""
Batch recording script for hours of Mario gameplay with annotations
Splits into multiple files to manage storage
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import time

os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from pyboy import PyBoy
from pyboy.utils import WindowEvent

ACTION_NAMES = ["NOP", "A", "B", "UP", "DOWN", "LEFT", "RIGHT", "JUMP_RIGHT", "JUMP_LEFT"]
ACTIONS = [
    [WindowEvent.PASS], [WindowEvent.PRESS_BUTTON_A], [WindowEvent.PRESS_BUTTON_B],
    [WindowEvent.PRESS_ARROW_UP], [WindowEvent.PRESS_ARROW_DOWN],
    [WindowEvent.PRESS_ARROW_LEFT], [WindowEvent.PRESS_ARROW_RIGHT],
    [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],
    [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],
]
BUTTONS = [WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_DOWN, 
           WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_ARROW_LEFT,
           WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B]
RELEASE_BUTTONS = [WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_ARROW_DOWN,
                   WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_LEFT,
                   WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B]
RELEASE_LOOKUP = {btn: rel for btn, rel in zip(BUTTONS, RELEASE_BUTTONS)}


class MarioRecorder:
    def __init__(self, rom_path):
        self.pyboy = PyBoy(rom_path, window="null", sound=False)
        self.pyboy.game_wrapper.start_game()
        self.screen = self.pyboy.screen
        self.held = {btn: False for btn in BUTTONS}
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        
    def reset(self):
        self.pyboy.game_wrapper.reset_game()
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        
    def get_frame(self):
        return self.screen.ndarray[:, :, :3]
    
    def step(self, action_idx):
        action = ACTIONS[action_idx]
        for btn in self.held:
            if self.held[btn] and btn not in action:
                self.pyboy.send_input(RELEASE_LOOKUP[btn])
                self.held[btn] = False
        for btn in action:
            if btn != WindowEvent.PASS:
                self.pyboy.send_input(btn)
                self.held[btn] = True
        
        self.pyboy.tick(8)
        
        w = self.pyboy.game_wrapper
        world, level = w.world
        progress = w.level_progress
        score = w.score
        lives = w.lives_left
        time_left = w.time_left
        
        reward = 0
        if progress > self.prev_progress:
            reward += (progress - self.prev_progress) * 10
            self.prev_progress = progress
        if score > self.prev_score:
            reward += (score - self.prev_score) * 0.1
            self.prev_score = score
        if lives < self.prev_lives:
            reward -= 15
            self.prev_lives = lives
        reward -= 0.1
        
        done = w.game_over() or time_left == 0
        
        return reward, done, {
            'world': world, 'level': level, 'progress': progress,
            'score': score, 'lives': lives, 'time_left': time_left
        }
    
    def close(self):
        self.pyboy.stop()


def record_chunk(rom_path, output_dir, chunk_num, duration_sec=300, fps=30):
    """Record one chunk (default 5 minutes)"""
    
    recorder = MarioRecorder(rom_path)
    recorder.reset()
    
    frame = recorder.get_frame()
    h, w = frame.shape[:2]
    
    chunk_name = f"mario_batch_{chunk_num:04d}"
    video_path = os.path.join(output_dir, f"{chunk_name}.mp4")
    json_path = os.path.join(output_dir, f"{chunk_name}.json")
    
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    annotations = {
        'metadata': {
            'chunk': chunk_num,
            'duration_sec': duration_sec,
            'fps': fps,
            'created_at': datetime.now().isoformat(),
        },
        'episodes': []
    }
    
    max_frames = duration_sec * fps
    frame_count = 0
    ep_num = 0
    ep_data = {'episode': ep_num, 'frames': []}
    
    print(f"   Chunk {chunk_num}: Recording {duration_sec}s ({max_frames} frames)...")
    start_time = time.time()
    
    while frame_count < max_frames:
        action_idx = np.random.randint(0, len(ACTIONS))
        action_name = ACTION_NAMES[action_idx]
        
        frame = recorder.get_frame()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Overlay
        text = f"C{chunk_num} F{frame_count} A:{action_name}"
        cv2.putText(frame_bgr, text, (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        writer.write(frame_bgr)
        
        reward, done, info = recorder.step(action_idx)
        
        ep_data['frames'].append({
            'frame': frame_count,
            'episode': ep_num,
            'action': {'index': int(action_idx), 'name': action_name},
            'reward': float(reward),
            'state': {
                'world': int(info['world']), 'level': int(info['level']),
                'progress': int(info['progress']), 'score': int(info['score']),
                'lives': int(info['lives']), 'time_left': int(info['time_left'])
            },
            'done': bool(done)
        })
        
        frame_count += 1
        
        if done:
            annotations['episodes'].append(ep_data)
            ep_num += 1
            ep_data = {'episode': ep_num, 'frames': []}
            recorder.reset()
    
    if ep_data['frames']:
        annotations['episodes'].append(ep_data)
    
    writer.release()
    recorder.close()
    
    with open(json_path, 'w') as f:
        json.dump(annotations, f)
    
    elapsed = time.time() - start_time
    print(f"   ✅ Chunk {chunk_num} done in {elapsed:.1f}s")
    print(f"      Video: {video_path}, JSON: {json_path}")
    
    return video_path, json_path


def batch_record(
    rom_path='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb',
    output_dir='/data/workspace/mario_dataset',
    total_hours=2.5,
    chunk_minutes=5
):
    """
    Record hours of gameplay in chunks
    
    Args:
        total_hours: Total recording time (default 2.5 hours = ~10GB)
        chunk_minutes: Duration per file (default 5 min chunks)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_seconds = total_hours * 3600
    chunk_seconds = chunk_minutes * 60
    num_chunks = int(total_seconds / chunk_seconds)
    
    print(f"🎮 Mario Dataset Batch Recording")
    print(f"   Total duration: {total_hours} hours ({total_seconds}s)")
    print(f"   Chunk size: {chunk_minutes} minutes ({chunk_seconds}s)")
    print(f"   Number of chunks: {num_chunks}")
    print(f"   Output dir: {output_dir}")
    print()
    
    start = time.time()
    
    for i in range(num_chunks):
        chunk_start = time.time()
        print(f"\n[{i+1}/{num_chunks}] Starting chunk...")
        
        try:
            record_chunk(rom_path, output_dir, i, chunk_seconds, fps=30)
        except Exception as e:
            print(f"   ❌ Error in chunk {i}: {e}")
            continue
        
        # Progress
        elapsed = time.time() - start
        remaining_chunks = num_chunks - i - 1
        avg_time = elapsed / (i + 1)
        eta = remaining_chunks * avg_time
        
        print(f"   Progress: {i+1}/{num_chunks} | Elapsed: {elapsed/60:.1f}m | ETA: {eta/3600:.1f}h")
    
    total_elapsed = time.time() - start
    print(f"\n✅ Batch complete!")
    print(f"   Total time: {total_elapsed/3600:.2f} hours")
    print(f"   Output: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=2.5, help='Total hours to record')
    parser.add_argument('--chunk-minutes', type=float, default=5, help='Minutes per file')
    parser.add_argument('--output', default='/data/workspace/mario_dataset', help='Output directory')
    parser.add_argument('--rom', default='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb')
    args = parser.parse_args()
    
    batch_record(
        rom_path=args.rom,
        output_dir=args.output,
        total_hours=args.hours,
        chunk_minutes=args.chunk_minutes
    )
