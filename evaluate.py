# evaluate.py
# Evaluates the segmentation model on validation data.
# Inputs: Validation image and mask tiles from the validation/tiles/ directory (can be created using prepare.py).
# Outputs: Evaluation metrics such as IoU, F1-score, accuracy, etc.

# Sources:
# - StackOverflow: https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
# - scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
# - Segmentation Models: https://smp.readthedocs.io/en/latest/metrics.html

import os
import glob
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torchvision import transforms
from sklearn.metrics import (
    jaccard_score, f1_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, balanced_accuracy_score
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 # Dataset
class ValidationDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])[:, :, ::-1].copy()  # RGB
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        img_tensor = self.transform(img)
        mask_tensor = torch.tensor(mask).unsqueeze(0).float()

        return img_tensor, mask_tensor

# Load model
model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
model.load_state_dict(torch.load("models/roadsegmentation_model.pth", map_location=DEVICE))
model.eval()
model.to(DEVICE)

# Load validation data
val_dataset = ValidationDataset("validation/tiles/images", "validation/tiles/masks")
val_loader = DataLoader(val_dataset, batch_size=1)

# Collect metrics
ious, f1s, accs, precs, recs, specs, bals = [], [], [], [], [], [], []
TP_total, FP_total, TN_total, FN_total = 0, 0, 0, 0

print("Start Evaluation...")
with torch.no_grad():
    for imgs, masks in tqdm(val_loader):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(imgs)
        preds = torch.sigmoid(preds)
        preds_bin = (preds > 0.5).float()

        y_true = masks.cpu().numpy().flatten()
        y_pred = preds_bin.cpu().numpy().flatten()

        if y_true.sum() > 0:
            ious.append(jaccard_score(y_true, y_pred))
            f1s.append(f1_score(y_true, y_pred))
            accs.append(accuracy_score(y_true, y_pred))
            precs.append(precision_score(y_true, y_pred, zero_division=0))
            recs.append(recall_score(y_true, y_pred, zero_division=0))
            bals.append(balanced_accuracy_score(y_true, y_pred))

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            specs.append(tn / (tn + fp + 1e-6))  # specificity
            TP_total += tp
            FP_total += fp
            TN_total += tn
            FN_total += fn

 # Show results
print("\nEvaluation results:")
print(f"IoU (Jaccard Index):       {np.mean(ious):.4f}")
print(f"F1-Score (Dice Coeff.):    {np.mean(f1s):.4f}")
print(f"Accuracy:                  {np.mean(accs):.4f}")
print(f"Precision:                 {np.mean(precs):.4f}")
print(f"Recall (Sensitivity):      {np.mean(recs):.4f}")
print(f"Specificity:               {np.mean(specs):.4f}")
print(f"Balanced Accuracy:         {np.mean(bals):.4f}")
print(f"\n Total: TP={TP_total}, FP={FP_total}, TN={TN_total}, FN={FN_total}")
