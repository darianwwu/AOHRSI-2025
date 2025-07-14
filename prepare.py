# prepare.py
# Prepares the training data by converting the OSM mask to a binary mask,
# cropping the orthophoto to a multiple of tile size, and generating image and mask tiles.
# Inputs: Orthophoto GeoTIFF and OSM mask GeoTIFF.
# Outputs: Image and mask tiles (512x512) in the training_data directory.

# Sources:
# - OpenCV documentation: https://docs.opencv.org/4.x/
# - Stack Overflow: https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
# - Stack Overflow: https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
# - Stack Overflow: https://stackoverflow.com/questions/64692538/how-to-use-opencv-to-create-binary-mask-of-images

import os
import cv2
import numpy as np
from PIL import Image

 # Input paths
 # See README for details of input file formats
ortho_tif_path = "inputs/WarendorfOrtho.tif"
osm_tif_path = "inputs/WarendorfOSM.tif"
converted_ortho_path = "inputs/WarendorfOrthoConverted.png"
binary_mask_path = "inputs/WarendorfOSMBinary.png"

 # Output directories
tile_size = 512
image_tiles_dir = "training_data/tiles/images"
mask_tiles_dir = "training_data/tiles/masks"
os.makedirs(image_tiles_dir, exist_ok=True)
os.makedirs(mask_tiles_dir, exist_ok=True)

 # Convert OSM TIF to binary mask
print("Converting OSM mask to binary PNG...")

osm = cv2.imread(osm_tif_path, cv2.IMREAD_UNCHANGED)
if osm is None:
    raise ValueError(f"Could not load file: {osm_tif_path}")

print(f"Mask loaded: shape={osm.shape}, dtype={osm.dtype}, min={np.min(osm)}, max={np.max(osm)}")

binary_mask = np.where(osm > 0, 255, 0).astype("uint8")
cv2.imwrite(binary_mask_path, binary_mask)
print(f"Binary mask saved to: {binary_mask_path}")

 # Crop orthophoto & mask
def crop_to_multiple(img, tile_size):
    w, h = img.size
    return img.crop((0, 0, (w // tile_size) * tile_size, (h // tile_size) * tile_size))

ortho_img = Image.open(ortho_tif_path).convert("RGB")
ortho_img = crop_to_multiple(ortho_img, tile_size)
ortho_img.save(converted_ortho_path)
print(f"Orthophoto saved: {converted_ortho_path}")

mask_img = Image.open(binary_mask_path).convert("L")
mask_img = crop_to_multiple(mask_img, tile_size)
print("Mask synchronously cropped")

 # Generate tiles
width, height = ortho_img.size
tile_count = 0

for y in range(0, height, tile_size):
    for x in range(0, width, tile_size):
        img_tile = ortho_img.crop((x, y, x + tile_size, y + tile_size))
        mask_tile = mask_img.crop((x, y, x + tile_size, y + tile_size))
        tile_name = f"tile_{x}_{y}.png"
        img_tile.save(os.path.join(image_tiles_dir, tile_name))
        mask_tile.save(os.path.join(mask_tiles_dir, tile_name))
        tile_count += 1

print(f"{tile_count} tiles saved in:")
print(f"{image_tiles_dir}")
print(f"{mask_tiles_dir}")
