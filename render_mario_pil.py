#!/usr/bin/env python3
"""Record a short clip of Mario-style gameplay using PIL"""
import os
from PIL import Image, ImageDraw, ImageFont
import subprocess
import shutil

# Game state
mario = {'x': 20, 'y': 120, 'vx': 0, 'vy': 0, 'on_ground': False}
enemies = []
blocks = []
camera = {'x': 0}
score = 0
time = 400

# Initialize blocks
for i in range(100):
    blocks.append({'x': i * 16, 'y': 136, 'type': 'ground'})
    if i % 5 == 3 and i > 5:
        blocks.append({'x': i * 16, 'y': 104, 'type': 'brick'})

# Add enemies
for i in range(5):
    enemies.append({'x': 150 + i * 80, 'y': 120, 'vx': -0.5, 'type': 'goomba', 'alive': True})

def draw_frame(draw, img, frame_num):
    global mario, enemies, camera, score, time
    
    width, height = img.size
    
    # Sky background
    draw.rectangle([0, 0, width, height], fill='#5c94fc')
    
    # Clouds
    cloud_offset = (camera['x'] * 0.3) % 200
    for i in range(3):
        cx = int(30 + i * 70 - cloud_offset)
        if -40 < cx < 200:
            draw.rounded_rectangle([cx, 30, cx + 24, 42], radius=6, fill='white')
            draw.rounded_rectangle([cx - 8, 36, cx + 32, 44], radius=4, fill='white')
    
    # Hills
    for i in range(5):
        hx = int(i * 100 - (camera['x'] * 0.5) % 100)
        if -80 < hx < 240:
            draw.polygon([(hx, 136), (hx + 40, 100), (hx + 80, 136)], fill='#00a800')
    
    # Ground and blocks
    for b in blocks:
        x = int(b['x'] - camera['x'])
        if x < -16 or x > 176:
            continue
        
        if b['type'] == 'ground':
            draw.rectangle([x, b['y'], x + 16, b['y'] + 8], fill='#8b4513')
            draw.rectangle([x, b['y'], x + 16, b['y'] + 2], fill='#0f0')
        elif b['type'] == 'brick':
            draw.rectangle([x, b['y'], x + 16, b['y'] + 16], fill='#b8860b', outline='#8b4513')
    
    # Enemies
    for e in enemies:
        if not e['alive']:
            continue
        x = int(e['x'] - camera['x'])
        if x < -16 or x > 176:
            continue
        
        if e['type'] == 'goomba':
            draw.rounded_rectangle([x, e['y'], x + 12, e['y'] + 12], radius=2, fill='#8b4513')
            draw.rectangle([x, e['y'] + 10, x + 3, e['y'] + 12], fill='black')
            draw.rectangle([x + 9, e['y'] + 10, x + 12, e['y'] + 12], fill='black')
    
    # Mario
    mx = int(mario['x'] - camera['x'])
    my = int(mario['y'])
    
    # Body/hat (red)
    draw.rectangle([mx, my + 4, mx + 8, my + 8], fill='#d03000')
    draw.rectangle([mx, my - 1, mx + 8, my + 2], fill='#d03000')
    
    # Head (skin tone)
    draw.rectangle([mx + 1, my, mx + 7, my + 4], fill='#ffccaa')
    
    # Overalls (blue)
    draw.rectangle([mx, my + 8, mx + 8, my + 16], fill='#0000aa')
    
    # UI background
    draw.rectangle([0, 0, 160, 16], fill='black')
    
    # UI text - using default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 8)
    except:
        font = ImageFont.load_default()
    
    draw.text((8, 0), 'MARIO', fill='white', font=font)
    draw.text((8, 8), str(score).zfill(6), fill='white', font=font)
    draw.text((72, 0), 'WORLD', fill='white', font=font)
    draw.text((78, 8), '1-1', fill='white', font=font)
    draw.text((128, 0), 'TIME', fill='white', font=font)
    draw.text((132, 8), str(int(time)), fill='white', font=font)
    
    # Update game state
    mario['vx'] = 1.5
    
    # Auto-jump occasionally
    if mario['on_ground'] and frame_num % 45 == 0:
        mario['vy'] = -4
        mario['on_ground'] = False
    
    mario['vy'] += 0.3
    mario['x'] += mario['vx']
    mario['y'] += mario['vy']
    
    # Ground collision
    mario['on_ground'] = False
    for b in blocks:
        if b['type'] == 'ground':
            if (mario['x'] + 8 > b['x'] and mario['x'] < b['x'] + 16 and
                mario['y'] + 16 >= b['y'] and mario['y'] + 16 <= b['y'] + 10):
                mario['y'] = b['y'] - 16
                mario['vy'] = 0
                mario['on_ground'] = True
    
    camera['x'] = mario['x'] - 60
    
    # Update enemies
    for e in enemies:
        if not e['alive']:
            continue
        e['x'] += e['vx']
        if abs(e['x'] - mario['x']) > 100:
            e['vx'] *= -1
        
        if abs(mario['x'] - e['x']) < 10 and abs(mario['y'] - e['y']) < 14:
            if mario['vy'] > 0 and mario['y'] < e['y']:
                e['alive'] = False
                mario['vy'] = -3
                score += 100
    
    if frame_num % 60 == 0:
        time -= 1
    if mario['vx'] != 0:
        score += 10

def main():
    fps = 30
    duration = 15
    total_frames = fps * duration
    
    frames_dir = '/data/workspace/mario_frames'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)
    
    print(f"Recording {duration}s at {fps}fps ({total_frames} frames)...")
    
    for i in range(total_frames):
        # Create image at native resolution (160x144)
        img = Image.new('RGB', (160, 144), '#5c94fc')
        draw = ImageDraw.Draw(img)
        
        draw_frame(draw, img, i)
        
        # Scale up for better video quality
        img_scaled = img.resize((480, 432), Image.NEAREST)
        
        img_scaled.save(f'{frames_dir}/frame_{i:04d}.png')
        
        if i % 30 == 0:
            print(f"Progress: {i}/{total_frames}")
    
    print('Frames done! Creating video...')
    
    # Create video with ffmpeg
    output = '/data/workspace/mario_clip.mp4'
    cmd = [
        'ffmpeg', '-framerate', str(fps), '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', output, '-y'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        return
    
    print(f'Video created: {output}')
    
    # Clean up frames
    shutil.rmtree(frames_dir)
    print('Cleaned up frames')

if __name__ == '__main__':
    main()
