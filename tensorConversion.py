import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

# 1. Load the dataset from your local folder
local_path = "./local_femnist_data"
dataset = load_from_disk(local_path)

# 2. Define the transformation to convert raw images to PyTorch Tensors
transform = Compose([ToTensor()])

# 3. Create a helper function to apply the transformation
def apply_transforms(batch):
    # Convert each image in the batch to a tensor (scales pixels to 0.0 - 1.0)
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch

# 4. Apply the transformation to the training data on the fly
train_data = dataset["train"].with_transform(apply_transforms)

# 5. Wrap it in a PyTorch DataLoader
# batch_size=32 is a standard starting point; shuffle=True mixes the order
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Verify the conversion by inspecting the first batch
first_batch = next(iter(train_loader))
images = first_batch["image"]
labels = first_batch["character"]

print(f"Image batch shape: {images.shape}") # Should be [32, 1, 28, 28]
print(f"Label batch shape: {labels.shape}") # Should be [32]