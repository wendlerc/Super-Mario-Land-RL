#!/usr/bin/env python3
"""
Mixed Mario dataset: 95% trained agent, 5% random agent
Records full episodes until completion or death
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

os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from pyboy import PyBoy
from pyboy.utils import WindowEvent
from stable_baselines3 import PPO

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


class MarioEnv:
    """Simple env wrapper for SB3"""
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
        return self._get_obs()
    
    def _get_obs(self):
        rgb = self.screen.ndarray[18:, :, :3]
        img = Image.fromarray(rgb, 'RGB').convert('L')
        small = np.array(img.resize((80, 72), Image.Resampling.LANCZOS))
        return np.expand_dims(small, axis=-1)
    
    def get_frame_rgb(self):
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
        obs = self._get_obs()
        
        info = {
            'world': world, 'level': level, 'progress': progress,
            'score': score, 'lives': lives, 'time_left': time_left
        }
        
        return obs, reward, done, info
    
    def close(self):
        self.pyboy.stop()


def record_episode(env, model, output_dir, episode_num, use_trained=True, fps=30, max_frames=30000):
    """Record one episode with trained or random agent"""
    
    obs = env.reset()
    frame = env.get_frame_rgb()
    h, w = frame.shape[:2]
    
    agent_type = "trained" if use_trained else "random"
    ep_name = f"mario_ep{episode_num:04d}_{agent_type}"
    video_path = os.path.join(output_dir, f"{ep_name}.mp4")
    json_path = os.path.join(output_dir, f"{ep_name}.json")
    
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    annotation = {
        'metadata': {
            'episode': episode_num,
            'agent': agent_type,
            'fps': fps,
            'max_frames': max_frames,
            'created_at': datetime.now().isoformat(),
        },
        'frames': []
    }
    
    frame_count = 0
    done = False
    
    print(f"   🎮 Episode {episode_num} ({agent_type}): Recording...")
    start_time = time.time()
    
    while not done and frame_count < max_frames:
        # Select action
        if use_trained and model is not None:
            action_idx, _ = model.predict(obs, deterministic=False)
            action_idx = int(action_idx)
        else:
            action_idx = np.random.randint(0, len(ACTIONS))
        
        action_name = ACTION_NAMES[action_idx]
        
        # Record frame
        frame = env.get_frame_rgb()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Overlay info
        text = f"EP{episode_num} {agent_type[:3].upper()} F{frame_count} {action_name}"
        cv2.putText(frame_bgr, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        writer.write(frame_bgr)
        
        # Take step
        obs, reward, done, info = env.step(action_idx)
        
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
        
        # Progress update every minute
        if frame_count % (fps * 60) == 0:
            minutes = frame_count / fps / 60
            print(f"      {minutes:.1f} min | Progress: {info['progress']} | Lives: {info['lives']}")
    
    # Finalize
    annotation['metadata']['total_frames'] = frame_count
    annotation['metadata']['duration_sec'] = frame_count / fps
    annotation['metadata']['final_progress'] = info['progress']
    annotation['metadata']['final_score'] = info['score']
    annotation['metadata']['completed'] = not done
    
    writer.release()
    
    with open(json_path, 'w') as f:
        json.dump(annotation, f)
    
    elapsed = time.time() - start_time
    duration = frame_count / fps
    print(f"   ✅ {agent_type:7s} ep {episode_num}: {duration/60:.1f} min, {frame_count} frames, "
          f"progress: {info['progress']}, ended: {'completed' if not done else 'died'}")
    
    return video_path, json_path, frame_count, agent_type


def generate_mixed_dataset(
    rom_path='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb',
    model_path='/data/workspace/mario-land-rl/mario_ppo_model.zip',
    output_dir='/data/workspace/mario_mixed_dataset',
    target_size_gb=10,
    trained_ratio=0.95,
    fps=30
):
    """
    Generate mixed dataset: 95% trained, 5% random
    """
    
    os.makedirs(output_dir, exist_ok=True)
    target_bytes = target_size_gb * 1024 * 1024 * 1024
    
    # Load trained model
    print("🧠 Loading trained PPO model...")
    try:
        model = PPO.load(model_path)
        print(f"   ✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        model = None
        trained_ratio = 0.0  # Fall back to all random
    
    # Create environment
    env = MarioEnv(rom_path)
    
    print(f"\n🎮 Mixed Dataset Generator")
    print(f"   Target: {target_size_gb} GB")
    print(f"   Ratio: {trained_ratio*100:.0f}% trained, {(1-trained_ratio)*100:.0f}% random")
    print(f"   Output: {output_dir}")
    print()
    
    episode_num = 0
    total_bytes = 0
    total_frames = 0
    trained_count = 0
    random_count = 0
    start_time = time.time()
    
    while total_bytes < target_bytes:
        # Decide agent type (95% trained, 5% random)
        use_trained = random.random() < trained_ratio if model is not None else False
        
        print(f"\n[{episode_num+1}] Starting episode...")
        
        try:
            video_path, json_path, frames, agent_type = record_episode(
                env, model, output_dir, episode_num, 
                use_trained=use_trained, fps=fps
            )
            
            # Track stats
            if agent_type == "trained":
                trained_count += 1
            else:
                random_count += 1
            
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
            print(f"   Episodes: {episode_num} ({trained_count} trained, {random_count} random)")
            print(f"   Frames: {total_frames:,} | Time: {elapsed/60:.1f} min")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    env.close()
    
    total_elapsed = time.time() - start_time
    print(f"\n✅ Dataset complete!")
    print(f"   Total: {total_bytes / (1024**3):.2f} GB")
    print(f"   Episodes: {episode_num}")
    print(f"   Trained: {trained_count} ({trained_count/episode_num*100:.1f}%)")
    print(f"   Random: {random_count} ({random_count/episode_num*100:.1f}%)")
    print(f"   Frames: {total_frames:,}")
    print(f"   Time: {total_elapsed/3600:.2f} hours")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-gb', type=float, default=10)
    parser.add_argument('--trained-ratio', type=float, default=0.95)
    parser.add_argument('--output', default='/data/workspace/mario_mixed_dataset')
    parser.add_argument('--model', default='/data/workspace/mario-land-rl/mario_ppo_model.zip')
    parser.add_argument('--rom', default='/data/workspace/roms/Super Mario Land (World) (Rev 1).gb')
    args = parser.parse_args()
    
    generate_mixed_dataset(
        rom_path=args.rom,
        model_path=args.model,
        output_dir=args.output,
        target_size_gb=args.target_gb,
        trained_ratio=args.trained_ratio
    )
