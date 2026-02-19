#!/usr/bin/env python3
"""
Long-episode Mario recorder - captures full levels
Records until episode ends (death or level complete)
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


def record_full_episode(rom_path, output_dir, episode_num, max_frames=30000, fps=30):
    """
    Record one full episode (until death or level complete)
    max_frames: safety limit (30k = ~16 minutes at 30fps)
    """
    
    recorder = MarioRecorder(rom_path)
    recorder.reset()
    
    frame = recorder.get_frame()
    h, w = frame.shape[:2]
    
    ep_name = f"mario_episode_{episode_num:04d}"
    video_path = os.path.join(output_dir, f"{ep_name}.mp4")
    json_path = os.path.join(output_dir, f"{ep_name}.json")
    
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    annotation = {
        'metadata': {
            'episode': episode_num,
            'fps': fps,
            'max_frames': max_frames,
            'created_at': datetime.now().isoformat(),
        },
        'frames': []
    }
    
    frame_count = 0
    done = False
    
    print(f"   🎮 Episode {episode_num}: Recording...")
    start_time = time.time()
    
    while not done and frame_count < max_frames:
        action_idx = np.random.randint(0, len(ACTIONS))
        action_name = ACTION_NAMES[action_idx]
        
        frame = recorder.get_frame()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Overlay
        text = f"EP{episode_num} F{frame_count} {action_name}"
        cv2.putText(frame_bgr, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
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
        
        # Progress every minute
        if frame_count % (fps * 60) == 0:
            minutes = frame_count / fps / 60
            print(f"      {minutes:.1f} min | Progress: {info['progress']} | Lives: {info['lives']}")
    
    # Update metadata
    annotation['metadata']['total_frames'] = frame_count
    annotation['metadata']['duration_sec'] = frame_count / fps
    annotation['metadata']['final_progress'] = info['progress']
    annotation['metadata']['final_score'] = info['score']
    annotation['metadata']['completed'] = not done  # True if hit max_frames without dying
    
    writer.release()
    recorder.close()
    
    with open(json_path, 'w') as f:
        json.dump(annotation, f)
    
    elapsed = time.time() - start_time
    duration = frame_count / fps
    print(f"   ✅ Episode {episode_num} complete: {duration/60:.1f} min, {frame_count} frames, ended: {'completed' if not done else 'died'}")
    
    return video_path, json_path, frame_count


def generate_dataset(
    rom_path='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb',
    output_dir='/data/workspace/mario_long_episodes',
    target_size_gb=10,
    fps=30
):
    """
    Generate episodes until we hit target size
    """
    
    os.makedirs(output_dir, exist_ok=True)
    target_bytes = target_size_gb * 1024 * 1024 * 1024
    
    print(f"🎮 Mario Long-Episode Dataset Generator")
    print(f"   Target size: {target_size_gb} GB")
    print(f"   Output: {output_dir}")
    print(f"   FPS: {fps}")
    print()
    
    episode_num = 0
    total_bytes = 0
    total_frames = 0
    start_time = time.time()
    
    while total_bytes < target_bytes:
        print(f"\n[{episode_num+1}] Starting new episode...")
        
        try:
            video_path, json_path, frames = record_full_episode(
                rom_path, output_dir, episode_num, max_frames=30000, fps=fps
            )
            
            # Calculate size
            video_size = os.path.getsize(video_path)
            json_size = os.path.getsize(json_path)
            chunk_size = video_size + json_size
            
            total_bytes += chunk_size
            total_frames += frames
            episode_num += 1
            
            # Progress
            used_gb = total_bytes / (1024**3)
            pct = (total_bytes / target_bytes) * 100
            elapsed = time.time() - start_time
            
            print(f"   Progress: {used_gb:.2f} GB / {target_size_gb} GB ({pct:.1f}%)")
            print(f"   Total episodes: {episode_num}, Total frames: {total_frames:,}")
            print(f"   Elapsed: {elapsed/60:.1f} min")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    total_elapsed = time.time() - start_time
    print(f"\n✅ Dataset complete!")
    print(f"   Total size: {total_bytes / (1024**3):.2f} GB")
    print(f"   Total episodes: {episode_num}")
    print(f"   Total frames: {total_frames:,}")
    print(f"   Time: {total_elapsed/3600:.2f} hours")
    print(f"   Output: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-gb', type=float, default=10, help='Target size in GB')
    parser.add_argument('--output', default='/data/workspace/mario_long_episodes')
    parser.add_argument('--rom', default='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    
    generate_dataset(
        rom_path=args.rom,
        output_dir=args.output,
        target_size_gb=args.target_gb,
        fps=args.fps
    )
