#!/usr/bin/env python3
import imageio
import numpy as np
import cv2

# Read the original video
reader = imageio.get_reader('/data/workspace/mario-land-rl/mario_random.mp4')

# Write with better codec
writer = imageio.get_writer('/data/workspace/mario-land-rl/mario_random_fixed.mp4', 
                            fps=30, codec='libx264', quality=8)

for frame in reader:
    writer.append_data(frame)

writer.close()
reader.close()
print("✅ Video re-encoded successfully")
