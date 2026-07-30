import os
import re
import cv2
import numpy as np

# Root folder
root_dir = "../data_logs/data_1743866797_423192"

# Regex to extract timestamp
folder_pattern = re.compile(r"\((\d+),")

# Target object ID
target_id =49
mask_filename = "mask.png"  # Change if needed

# Collect and sort folders
folders = []
for name in os.listdir(root_dir):
    match = folder_pattern.match(name)
    if match:
        timestamp = int(match.group(1))
        full_path = os.path.join(root_dir, name)
        if os.path.isdir(full_path):
            folders.append((timestamp, full_path))

folders.sort(key=lambda x: x[0])

# Accumulated mask
accumulated_mask = None

for _, folder in folders:
    mask_path = os.path.join(folder, mask_filename)
    if os.path.isfile(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if accumulated_mask is None:
            accumulated_mask = np.zeros_like(mask, dtype=np.uint16)
        
        object_mask = (mask == target_id).astype(np.uint16)
        accumulated_mask += object_mask

# Visualize using OpenCV
if accumulated_mask is not None and np.any(accumulated_mask):
    norm_mask = cv2.normalize(accumulated_mask, None, 0, 255, cv2.NORM_MINMAX)
    norm_mask = norm_mask.astype(np.uint8)

    heatmap = cv2.applyColorMap(norm_mask, cv2.COLORMAP_HOT)

    window_name = f"Accumulated Mask for Object {target_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, heatmap)

    # Wait loop that exits cleanly on close or key
    while True:
        key = cv2.waitKey(100)
        if key != -1 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
else:
    print("No valid masks found with object ID 39.")
