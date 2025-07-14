# train.py
# Trains the segmentation model again
# Inputs: Image and mask tiles from the training_data directory (can be created using prepare.py).
# Outputs: Trained model saved as roadsegmentation_model_finetuned.pth in the models directory.

# Sources:
# - OpenCV documentation: https://docs.opencv.org/4.x/
# - Stack Overflow: https://stackoverflow.com/questions/68402165/how-to-create-a-custom-pytorch-dataset-with-multiple-labels-and-masks
# - Segmentation Model GitHub: https://github.com/qubvel-org/segmentation_models.pytorch
# - pyImageSearch: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import segmentation_models_pytorch as smp
from torchvision import transforms
from tqdm import tqdm

# Parameters
BATCH_SIZE = 4
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class RoadDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])[:, :, ::-1].copy()  # BGR → RGB
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        img = self.transform(img)
        mask = torch.tensor(mask).unsqueeze(0)  # [1, H, W]
        return img, mask

# Load and combine all datasets
datasets = []
base_dirs = ["training_data"]

for base in base_dirs:
    img_dir = os.path.join(base, "tiles", "images")
    mask_dir = os.path.join(base, "tiles", "masks")
    datasets.append(RoadDataset(img_dir, mask_dir))

# Combined dataset
combined_dataset = ConcatDataset(datasets)
loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize new model (U-Net with ResNet34 encoder)
model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1).to(DEVICE)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
print(" Start Training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for imgs, masks in tqdm(loader):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "models/roadsegmentation_model_finetuned.pth")
print("Training completed. Model saved as: roadsegmentation_model_finetuned.pth")
