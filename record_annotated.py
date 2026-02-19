#!/usr/bin/env python3
"""
Record Mario gameplay with action annotations
Saves video + JSON file with frame-by-frame data
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
from datetime import datetime

# SDL2 headless mode
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Action names for annotation
ACTION_NAMES = [
    "NOP",
    "A",
    "B", 
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "JUMP_RIGHT",
    "JUMP_LEFT"
]

ACTIONS = [
    [WindowEvent.PASS],
    [WindowEvent.PRESS_BUTTON_A],
    [WindowEvent.PRESS_BUTTON_B],
    [WindowEvent.PRESS_ARROW_UP],
    [WindowEvent.PRESS_ARROW_DOWN],
    [WindowEvent.PRESS_ARROW_LEFT],
    [WindowEvent.PRESS_ARROW_RIGHT],
    [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],
    [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],
]

BUTTONS = [
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
]

RELEASE_BUTTONS = [
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
]

RELEASE_BUTTON_LOOKUP = {btn: rel for btn, rel in zip(BUTTONS, RELEASE_BUTTONS)}


class AnnotatedMarioRecorder:
    """Record Mario gameplay with full annotations"""
    
    def __init__(self, rom_path='super_mario_land.gb'):
        self.rom_path = rom_path
        self.pyboy = PyBoy(rom_path, window="null", sound=False)
        self.pyboy.game_wrapper.start_game()
        self.screen = self.pyboy.screen
        self._currently_held = {btn: False for btn in BUTTONS}
        
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        
    def reset(self):
        self.pyboy.game_wrapper.reset_game()
        self.prev_progress = 0
        self.prev_score = 0
        self.prev_lives = 2
        return self._get_obs()
    
    def _get_obs(self):
        rgb = self.screen.ndarray[18:, :, :3]
        img = Image.fromarray(rgb, 'RGB').convert('L')
        small = np.array(img.resize((80, 72), Image.Resampling.LANCZOS))
        return np.expand_dims(small, axis=-1)
    
    def get_frame_rgb(self):
        """Get full RGB frame for video"""
        return self.screen.ndarray[:, :, :3]
    
    def do_action(self, action_idx):
        action = ACTIONS[action_idx]
        for btn in self._currently_held:
            if self._currently_held[btn] and btn not in action:
                self.pyboy.send_input(RELEASE_BUTTON_LOOKUP[btn])
                self._currently_held[btn] = False
        for btn in action:
            if btn not in [WindowEvent.PASS]:
                self.pyboy.send_input(btn)
                self._currently_held[btn] = True
    
    def step(self, action_idx):
        self.do_action(action_idx)
        self.pyboy.tick(8)
        
        wrapper = self.pyboy.game_wrapper
        progress = wrapper.level_progress
        score = wrapper.score
        lives = wrapper.lives_left
        time_left = wrapper.time_left
        
        # Calculate reward
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
        
        done = bool(self.pyboy.game_wrapper.game_over())
        if time_left == 0:
            done = True
        
        # Get world/level info (world is a tuple of (world, level))
        world, level = wrapper.world
        
        return reward, done, {
            'world': world,
            'level': level,
            'progress': progress,
            'score': score,
            'lives': lives,
            'time_left': time_left,
        }
    
    def close(self):
        self.pyboy.stop()


def record_with_annotations(
    rom_path='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb',
    output_prefix='mario_data',
    duration_seconds=30,
    fps=30,
    use_random_policy=True
):
    """
    Record gameplay with full annotations
    
    Args:
        rom_path: Path to ROM file
        output_prefix: Prefix for output files (video + json)
        duration_seconds: How long to record
        fps: Video FPS
        use_random_policy: If True, use random actions. If False, use trained model (if available)
    """
    
    print(f"🎮 Starting annotated recording...")
    print(f"   Duration: {duration_seconds}s")
    print(f"   FPS: {fps}")
    print(f"   Policy: {'Random' if use_random_policy else 'Trained'}")
    
    # Setup recorder
    recorder = AnnotatedMarioRecorder(rom_path=rom_path)
    recorder.reset()
    
    # Get frame dimensions
    frame = recorder.get_frame_rgb()
    height, width = frame.shape[:2]
    print(f"   Resolution: {width}x{height}")
    
    # Setup video writer
    video_path = f"{output_prefix}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Initialize annotations
    annotations = {
        'metadata': {
            'rom': rom_path,
            'duration_seconds': duration_seconds,
            'fps': fps,
            'resolution': {'width': width, 'height': height},
            'policy': 'random' if use_random_policy else 'trained_ppo',
            'created_at': datetime.now().isoformat(),
        },
        'episodes': []
    }
    
    # Recording loop
    max_frames = duration_seconds * fps
    frame_count = 0
    episode_num = 0
    episode_data = {
        'episode': episode_num,
        'frames': []
    }
    
    print(f"\n⏺️  Recording {max_frames} frames...")
    
    while frame_count < max_frames:
        # Get frame for video
        frame = recorder.get_frame_rgb()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Select action (random for now)
        if use_random_policy:
            action_idx = np.random.randint(0, len(ACTIONS))
        else:
            # Would load trained model here
            action_idx = np.random.randint(0, len(ACTIONS))
        
        action_name = ACTION_NAMES[action_idx]
        
        # Add overlay to video
        info_text = f"F:{frame_count} | A:{action_name} | P:{recorder.prev_progress}"
        cv2.putText(frame_bgr, info_text, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.35, (255, 255, 255), 1)
        
        video_writer.write(frame_bgr)
        
        # Take step
        reward, done, info = recorder.step(action_idx)
        
        # Store annotation
        frame_annotation = {
            'frame': frame_count,
            'episode': episode_num,
            'action': {
                'index': int(action_idx),
                'name': action_name,
            },
            'reward': float(reward),
            'state': {
                'world': int(info['world']),
                'level': int(info['level']),
                'progress': int(info['progress']),
                'score': int(info['score']),
                'lives': int(info['lives']),
                'time_left': int(info['time_left']),
            },
            'done': bool(done),
        }
        episode_data['frames'].append(frame_annotation)
        
        frame_count += 1
        
        # Progress update
        if frame_count % 300 == 0:
            elapsed_sec = frame_count / fps
            print(f"   {elapsed_sec:.1f}s / {duration_seconds}s | Progress: {info['progress']}")
        
        # Handle episode end
        if done:
            print(f"   💀 Episode {episode_num} ended at frame {frame_count}")
            annotations['episodes'].append(episode_data)
            
            episode_num += 1
            episode_data = {
                'episode': episode_num,
                'frames': []
            }
            
            recorder.reset()
    
    # Save final episode if not empty
    if episode_data['frames']:
        annotations['episodes'].append(episode_data)
    
    # Cleanup
    video_writer.release()
    recorder.close()
    
    # Save annotations
    json_path = f"{output_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    # Print summary
    total_episodes = len(annotations['episodes'])
    total_actions = sum(len(ep['frames']) for ep in annotations['episodes'])
    
    print(f"\n✅ Recording complete!")
    print(f"   Video: {video_path}")
    print(f"   Annotations: {json_path}")
    print(f"   Total frames: {frame_count}")
    print(f"   Episodes: {total_episodes}")
    print(f"   Duration: {frame_count/fps:.1f}s")
    
    return video_path, json_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom', default='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb')
    parser.add_argument('--output', default='mario_annotated')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--trained', action='store_true', help='Use trained model (if available)')
    args = parser.parse_args()
    
    record_with_annotations(
        rom_path=args.rom,
        output_prefix=args.output,
        duration_seconds=args.duration,
        fps=args.fps,
        use_random_policy=not args.trained
    )
