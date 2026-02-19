#!/usr/bin/env python3
"""
Improved Mario dataset generator with:
- Level variety (random worlds/levels)
- Progress tracking (resume capability)
- Efficient encoding
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import time
import random
import signal

# Handle graceful shutdown
running = True
def signal_handler(sig, frame):
    global running
    print("\n⚠️  Interrupt received, finishing current episode...")
    running = False
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
        self.rom_path = rom_path
        self.pyboy = None
        self.screen = None
        self.held = None
        
    def start_level(self, world=0, level=0):
        """Start from specific world/level"""
        if self.pyboy:
            self.pyboy.stop()
        
        self.pyboy = PyBoy(self.rom_path, window="null", sound=False)
        self.pyboy.game_wrapper.set_world_level(world, level)
        self.pyboy.game_wrapper.start_game()
        self.screen = self.pyboy.screen
        self.held = {btn: False for btn in BUTTONS}
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        
        return self.pyboy.game_wrapper.world
        
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
        
        info = {
            'world': world, 'level': level, 'progress': progress,
            'score': score, 'lives': lives, 'time_left': time_left,
            'start_world': world, 'start_level': level
        }
        
        return reward, done, info
    
    def close(self):
        if self.pyboy:
            self.pyboy.stop()


def record_episode(recorder, output_dir, episode_num, start_world=0, start_level=0, fps=30):
    """Record one episode from specific level"""
    
    actual_world, actual_level = recorder.start_level(start_world, start_level)
    
    frame = recorder.get_frame()
    h, w = frame.shape[:2]
    
    ep_name = f"mario_ep{episode_num:05d}_w{actual_world}l{actual_level}"
    video_path = os.path.join(output_dir, f"{ep_name}.mp4")
    json_path = os.path.join(output_dir, f"{ep_name}.json")
    
    # Skip if already exists (resume capability)
    if os.path.exists(video_path) and os.path.exists(json_path):
        print(f"   ⏭️  Episode {episode_num} already exists, skipping")
        return None, None, 0
    
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    annotation = {
        'metadata': {
            'episode': episode_num,
            'start_world': int(actual_world),
            'start_level': int(actual_level),
            'fps': fps,
            'created_at': datetime.now().isoformat(),
        },
        'frames': []
    }
    
    frame_count = 0
    done = False
    max_frames = 30000  # ~16 min safety limit
    
    print(f"   🎮 EP{episode_num} (World {actual_world}-{actual_level}): Recording...")
    
    while running and not done and frame_count < max_frames:
        action_idx = np.random.randint(0, len(ACTIONS))
        action_name = ACTION_NAMES[action_idx]
        
        frame = recorder.get_frame()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        text = f"EP{episode_num} W{actual_world}-{actual_level} F{frame_count}"
        cv2.putText(frame_bgr, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        writer.write(frame_bgr)
        
        reward, done, info = recorder.step(action_idx)
        
        annotation['frames'].append({
            'frame': frame_count,
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
        
        if frame_count % (fps * 60) == 0:  # Every minute
            minutes = frame_count / fps / 60
            print(f"      {minutes:.0f}m | Progress: {info['progress']} | Lives: {info['lives']}")
    
    # Finalize
    annotation['metadata']['total_frames'] = frame_count
    annotation['metadata']['duration_sec'] = frame_count / fps
    annotation['metadata']['final_progress'] = info['progress']
    annotation['metadata']['final_score'] = info['score']
    annotation['metadata']['completed'] = not done
    
    writer.release()
    
    with open(json_path, 'w') as f:
        json.dump(annotation, f)
    
    duration = frame_count / fps
    print(f"   ✅ EP{episode_num}: {duration/60:.1f}min, {frame_count} frames, "
          f"progress: {info['progress']}, ended: {'completed' if not done else 'died'}")
    
    return video_path, json_path, frame_count


def generate_dataset_with_levels(
    rom_path='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb',
    output_dir='/data/workspace/mario_multi_level',
    target_size_gb=10,
    fps=30
):
    """Generate dataset starting from random worlds/levels"""
    
    os.makedirs(output_dir, exist_ok=True)
    target_bytes = target_size_gb * 1024 * 1024 * 1024
    
    print(f"🎮 Multi-Level Dataset Generator")
    print(f"   Target: {target_size_gb} GB")
    print(f"   Output: {output_dir}")
    print(f"   Press Ctrl+C to stop gracefully")
    print()
    
    recorder = MarioRecorder(rom_path)
    episode_num = 0
    total_bytes = 0
    total_frames = 0
    start_time = time.time()
    
    while running and total_bytes < target_bytes:
        # Random world (0-3) and level (0-2)
        start_world = random.randint(0, 3)
        start_level = random.randint(0, 2)
        
        print(f"\n[{episode_num+1}] Starting from World {start_world}-{start_level}...")
        
        try:
            video_path, json_path, frames = record_episode(
                recorder, output_dir, episode_num, start_world, start_level, fps
            )
            
            if video_path is None:  # Already exists
                episode_num += 1
                continue
            
            video_size = os.path.getsize(video_path)
            json_size = os.path.getsize(json_path)
            chunk_size = video_size + json_size
            
            total_bytes += chunk_size
            total_frames += frames
            episode_num += 1
            
            used_gb = total_bytes / (1024**3)
            pct = (total_bytes / target_bytes) * 100
            elapsed = time.time() - start_time
            
            print(f"   Progress: {used_gb:.2f} GB / {target_size_gb} GB ({pct:.1f}%)")
            print(f"   Episodes: {episode_num} | Frames: {total_frames:,} | Time: {elapsed/60:.1f}m")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    recorder.close()
    
    total_elapsed = time.time() - start_time
    print(f"\n✅ Dataset generation stopped")
    print(f"   Total: {total_bytes / (1024**3):.2f} GB")
    print(f"   Episodes: {episode_num}")
    print(f"   Frames: {total_frames:,}")
    print(f"   Time: {total_elapsed/3600:.2f} hours")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-gb', type=float, default=10)
    parser.add_argument('--output', default='/data/workspace/mario_multi_level')
    parser.add_argument('--rom', default='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb')
    args = parser.parse_args()
    
    generate_dataset_with_levels(
        rom_path=args.rom,
        output_dir=args.output,
        target_size_gb=args.target_gb
    )
