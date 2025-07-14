# evaluateonnx.py
# Additional script
# Evaluates the ONNX model (Deepness) on validation data.
# Calculates IoU and F1-score.
# Inputs: Validation image and mask tiles from the validation/tiles/ directory (can be created using prepare.py).
# Outputs: Evaluation metrics IoU and F1-score.

# Sources:
# - onnxruntime: https://onnxruntime.ai/docs/get-started/with-python.html
# - scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
# - Stack Overflow: https://stackoverflow.com/questions/75360420/running-a-pre-trained-onnx-model-image-recognition

import os
import glob
import numpy as np
import onnxruntime as ort
import cv2
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score

 # Paths
onnx_model_path = "models/road_segmentation_model_with_metadata_26_10_22.onnx"
image_dir = "validation/tiles/images"
mask_dir = "validation/tiles/masks"

 # Helper function: prepare image
def preprocess(img):
    img = cv2.resize(img, (512, 512))  # or as required by model input
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # → (1, C, H, W)
    return img

 # ONNX inference
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

 # Validation data
img_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

ious, f1s = [], []

for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
    img = cv2.imread(img_path)[:, :, ::-1]
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8).flatten()

    input_tensor = preprocess(img)
    pred = session.run(None, {input_name: input_tensor})[0]
    pred_mask = (pred[0, 0] > 0.5).astype(np.uint8).flatten()

    if mask.sum() > 0:
        ious.append(jaccard_score(mask, pred_mask))
        f1s.append(f1_score(mask, pred_mask))

 # Results
print(f"\nONNX model – IoU: {np.mean(ious):.4f}")
print(f"ONNX model – F1:  {np.mean(f1s):.4f}")
