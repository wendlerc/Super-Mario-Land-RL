#!/usr/bin/env python3
"""Render multiple short Mario-style gameplay clips with different scenarios"""
from PIL import Image, ImageDraw, ImageFont
import os
import shutil

# scenarios: jump_sequence, enemy_stomp, running, etc.

def render_jump_clip(clip_name):
    """Clip showing Mario jumping over obstacles"""
    frames = []
    mario_x, mario_y = 20, 120
    vy = 0
    
    for frame_idx in range(0, 45):  # 1.5 seconds at 30fps
        img = Image.new('RGB', (160, 144), '#5c94fc')
        draw = ImageDraw.Draw(img)
        
        # Sky
        draw.rectangle([0, 0, 160, 144], fill='#5c94fc')
        
        # Ground
        draw.rectangle([0, 136, 160, 144], fill='#8b4513')
        draw.rectangle([0, 136, 160, 138], fill='#0f0')
        
        # Pipe obstacle
        pipe_x = 80
        draw.rectangle([pipe_x, 120, pipe_x + 20, 136], fill='#0f0', outline='#003300')  # Pipe
        draw.rectangle([pipe_x, 116, pipe_x + 20, 120], fill='#0f0', outline='#003300')  # Pipe top
        
        # Mario physics - jump arc
        if 10 <= frame_idx <= 15:  # Jump!
            if frame_idx == 10:
                vy = -5
            mario_y += vy
            vy += 0.4  # gravity
        elif frame_idx > 15:
            # Come down
            mario_y += vy
            vy += 0.4
            if mario_y >= 120:
                mario_y = 120
                vy = 0
        
        mario_x += 1.5  # Running right
        
        # Mario body
        mx, my = int(mario_x), int(mario_y)
        draw.rectangle([mx, my + 4, mx + 8, my + 8], fill='#d03000')  # Red
        draw.rectangle([mx + 1, my, mx + 7, my + 4], fill='#ffccaa')  # Skin
        draw.rectangle([mx, my - 1, mx + 8, my + 2], fill='#d03000')  # Hat
        draw.rectangle([mx, my + 8, mx + 8, my + 16], fill='#0000aa')  # Overalls
        
        # UI
        draw.rectangle([0, 0, 160, 16], fill='black')
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 8)
        except:
            font = ImageFont.load_default()
        draw.text((8, 0), 'MARIO', fill='white', font=font)
        draw.text((8, 8), str(frame_idx * 10).zfill(6), fill='white', font=font)
        draw.text((72, 0), 'WORLD', fill='white', font=font)
        draw.text((78, 8), '1-1', fill='white', font=font)
        
        # Scale up
        img_scaled = img.resize((480, 432), Image.NEAREST)
        frames.append(img_scaled)
    
    return frames

def render_enemy_stomp_clip():
    """Clip showing Mario stomping an enemy"""
    frames = []
    mario_x, mario_y = 30, 120
    enemy_x, enemy_y = 100, 120
    enemy_alive = True
    vy = 0
    
    for frame_idx in range(60):  # 2 seconds
        img = Image.new('RGB', (160, 144), '#5c94fc')
        draw = ImageDraw.Draw(img)
        
        # Sky
        draw.rectangle([0, 0, 160, 144], fill='#5c94fc')
        
        # Ground
        draw.rectangle([0, 136, 160, 144], fill='#8b4513')
        draw.rectangle([0, 136, 160, 138], fill='#0f0')
        
        # Enemy (Goomba) - brown mushroom
        if enemy_alive:
            ex, ey = int(enemy_x), int(enemy_y)
            draw.rounded_rectangle([ex, ey, ex + 12, ey + 12], radius=2, fill='#8b4513')
            # Feet
            draw.rectangle([ex, ey + 10, ex + 3, ey + 12], fill='black')
            draw.rectangle([ex + 9, ey + 10, ex + 12, ey + 12], fill='black')
        else:
            # Squashed enemy
            draw.ellipse([int(enemy_x), enemy_y + 8, int(enemy_x) + 12, enemy_y + 12], fill='#8b4513')
        
        # Mario jump and stomp
        if 15 <= frame_idx <= 20 and enemy_alive:
            # Jump onto enemy
            mario_y = enemy_y - 16
            if frame_idx == 20:
                enemy_alive = False
                vy = -3  # Bounce up
        elif not enemy_alive:
            # Bounce up after stomp
            mario_y += vy
            vy += 0.3
            if mario_y >= 120:
                mario_y = 120
                vy = 0
        
        mario_x += 1.2
        
        # Mario
        mx, my = int(mario_x), int(mario_y)
        draw.rectangle([mx, my + 4, mx + 8, my + 8], fill='#d03000')
        draw.rectangle([mx + 1, my, mx + 7, my + 4], fill='#ffccaa')
        draw.rectangle([mx, my - 1, mx + 8, my + 2], fill='#d03000')
        draw.rectangle([mx, my + 8, mx + 8, my + 16], fill='#0000aa')
        
        # Score popup when enemy stomped
        if not enemy_alive and frame_idx < 35:
            draw.text((int(enemy_x), enemy_y - 20), '100', fill='white', font=font if 'font' in dir() else ImageFont.load_default())
        
        # UI
        draw.rectangle([0, 0, 160, 16], fill='black')
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 8)
        except:
            font = ImageFont.load_default()
        draw.text((8, 0), 'MARIO', fill='white', font=font)
        score_val = 100 if not enemy_alive else 0
        draw.text((8, 8), str(score_val).zfill(6), fill='white', font=font)
        
        img_scaled = img.resize((480, 432), Image.NEAREST)
        frames.append(img_scaled)
    
    return frames

def render_brick_break_clip():
    """Clip showing Mario hitting a brick block"""
    frames = []
    mario_x, mario_y = 20, 120
    brick_y = 104
    
    for frame_idx in range(45):
        img = Image.new('RGB', (160, 144), '#5c94fc')
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([0, 0, 160, 144], fill='#5c94fc')
        
        # Ground
        draw.rectangle([0, 136, 160, 144], fill='#8b4513')
        draw.rectangle([0, 136, 160, 138], fill='#0f0')
        
        # Question mark block
        brick_x = 60
        if frame_idx < 25 or frame_idx > 30:
            # Normal brick/question block
            draw.rectangle([brick_x, brick_y, brick_x + 16, brick_y + 16], fill='#b8860b', outline='#8b4513')
            if frame_idx < 10:
                draw.text((brick_x + 4, brick_y + 4), '?', fill='yellow', font=ImageFont.load_default())
        else:
            # Bump animation
            offset = 3 if frame_idx < 28 else 1
            draw.rectangle([brick_x, brick_y - offset, brick_x + 16, brick_y + 16 - offset], fill='#b8860b', outline='#8b4513')
            # Coin popping out
            coin_y = brick_y - 20 - (frame_idx - 25) * 3
            draw.ellipse([brick_x + 4, int(coin_y), brick_x + 12, int(coin_y) + 8], fill='#ffd700')
        
        # Mario jumps to hit block
        if 15 <= frame_idx < 25:
            mario_y = 88  # Up in the air
        elif frame_idx >= 25:
            mario_y = 120  # Back down
        
        mario_x += 1.3
        
        # Mario
        mx, my = int(mario_x), int(mario_y)
        draw.rectangle([mx, my + 4, mx + 8, my + 8], fill='#d03000')
        draw.rectangle([mx + 1, my, mx + 7, my + 4], fill='#ffccaa')
        draw.rectangle([mx, my - 1, mx + 8, my + 2], fill='#d03000')
        draw.rectangle([mx, my + 8, mx + 8, my + 16], fill='#0000aa')
        
        # UI
        draw.rectangle([0, 0, 160, 16], fill='black')
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 8)
        except:
            font = ImageFont.load_default()
        draw.text((8, 0), 'MARIO', fill='white', font=font)
        draw.text((8, 8), '000100', fill='white', font=font)
        
        img_scaled = img.resize((480, 432), Image.NEAREST)
        frames.append(img_scaled)
    
    return frames

def save_gif(frames, output_path, duration=67):
    """Save frames as GIF"""
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved: {output_path}")

# Generate all clips
print("Generating jump clip...")
jump_frames = render_jump_clip("jump")
save_gif(jump_frames, '/data/workspace/mario_jump.gif')

print("Generating enemy stomp clip...")
stomp_frames = render_enemy_stomp_clip()
save_gif(stomp_frames, '/data/workspace/mario_stomp.gif')

print("Generating brick hit clip...")
brick_frames = render_brick_break_clip()
save_gif(brick_frames, '/data/workspace/mario_brick.gif')

print("All clips generated!")
