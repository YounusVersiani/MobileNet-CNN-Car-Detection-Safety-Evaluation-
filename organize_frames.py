import cv2
import os

# Define output directories (raw strings for Windows)
static_frames_dir = r'C:\Users\ynver\OneDrive\Desktop\Projects\static_frames'  # Adjust this path
moving_frames_dir = r'C:\Users\ynver\OneDrive\Desktop\Projects\moving_frames'  #  Adjust this path

# Create directories if they don’t exist
os.makedirs(static_frames_dir, exist_ok=True)
os.makedirs(moving_frames_dir, exist_ok=True)

# Source KITTI data path
kitti_sync_dir = r'C:\Users\ynver\OneDrive\Desktop\Projects\2011_09_26\2011_09_26_drive_0001_sync\image_02\data'  # Adjust this path

# Process 108 frames (0 to 107)
for i in range(108):
    png_path = os.path.join(kitti_sync_dir, f'{i:010d}.png')  # e.g., C:\...\data\0000000000.png
    frame = cv2.imread(png_path)
    if frame is None:
        print(f"Failed to load {png_path} - Check path or file existence")
        continue
    
    # First 50 frames as static (0-49)
    if i < 50:
        jpg_path = os.path.join(static_frames_dir, f'frame_{i:03d}.jpg')
        cv2.imwrite(jpg_path, frame)
    # Last 50 frames as moving (58-107)
    elif i >= 58:
        moving_idx = i - 58  # Maps 58-107 to 0-49
        jpg_path = os.path.join(moving_frames_dir, f'frame_{moving_idx:03d}.jpg')
        cv2.imwrite(jpg_path, frame)

print(f"Organized 50 static frames to {static_frames_dir}")
print(f"Organized 50 moving frames to {moving_frames_dir}")
