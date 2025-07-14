# predict.py
# Performs the segmentation prediction on a GeoTIFF image (of all sizes) using the trained model.
# Inputs: GeoTIFF image of the area to be segmented.
# Outputs: Predicted mask as a GeoTIFF file.

# Sources:
# - Rasterio documentation: https://rasterio.readthedocs.io/en/stable/
# - PyTorch: https://torchgeo.readthedocs.io/en/stable/tutorials/torchgeo.html
# - Segmentation Model GitHub: https://github.com/qubvel-org/segmentation_models.pytorch

import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

 # Input GeoTIFF
INPUT_TIF = "inputs/WarendorfOrtho.tif"
OUTPUT_TIF = "outputs/predicted_mask.tif"
MODEL_PATH = "models/roadsegmentation_model.pth"

 # Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

 # Parameters
TILE_SIZE = 512
transform = transforms.ToTensor()

 # Prediction on TIF tiles
with rasterio.open(INPUT_TIF) as src:
    width, height = src.width, src.height
    profile = src.profile
    profile.update(dtype='uint8', count=1)

    # Empty prediction mask
    prediction_mask = np.zeros((height, width), dtype=np.uint8)

    print("Starting prediction...")
    for y in tqdm(range(0, height, TILE_SIZE)):
        for x in range(0, width, TILE_SIZE):
            window = Window(x, y, TILE_SIZE, TILE_SIZE)

            # Crop window if at image edge
            win_w = min(TILE_SIZE, width - x)
            win_h = min(TILE_SIZE, height - y)

            # Read and normalize RGB
            img = src.read([1, 2, 3], window=window)  # Shape: [3, H, W]
            img = np.transpose(img, (1, 2, 0))  # → [H, W, 3]
            img = Image.fromarray(img.astype(np.uint8))
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            # Prediction
            with torch.no_grad():
                pred = model(img_tensor)
                pred = torch.sigmoid(pred).squeeze().cpu().numpy()
                pred_bin = (pred > 0.5).astype(np.uint8)

            # Insert mask
            prediction_mask[y:y+win_h, x:x+win_w] = pred_bin[:win_h, :win_w]

    # Save mask as GeoTIFF
    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        dst.write(prediction_mask, 1)

print(f"Prediction mask saved: {OUTPUT_TIF}")
