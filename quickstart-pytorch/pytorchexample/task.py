import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import random

# --- Data Poisining 
# Select random clients to become malicious
def get_malicious_ids(num_partitions, fraction, seed):
    count = int(num_partitions * fraction)
    rng = random.Random(seed)
    return set(rng.sample(range(num_partitions), count))


# --- 1. THE MODEL ---
class FEMNISTNet(nn.Module):
    def __init__(self):
        super(FEMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62) # 62 classes for FEMNIST

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# --- 2. DATA LOADING ---
def load_data(partition_id: int, num_partitions: int, batch_size: int = 32):
    """Loads a specific shard of the local FEMNIST dataset."""
    dataset = load_from_disk("/Users/nathanieldeleon/Desktop/Projects/HackR-Anomaly-26/local_femnist_data")
    transform = Compose([ToTensor()])

    def apply_transforms(batch):
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch

   # Split the main dataset into 90% train, 10% test
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    # Shard the 90% train set among the clients
    train_data = split_dataset["train"].shard(num_shards=num_partitions, index=partition_id)
    train_data = train_data.with_transform(apply_transforms)

    # Keep the 10% test set whole for the server's evaluation
    test_data = split_dataset["test"].with_transform(apply_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, test_loader

# --- 3. TRAINING LOOP ---
def train(net, trainloader, epochs, lr, device, flip_labels=False, num_classes=62):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["character"].to(device)

            if flip_labels:
                labels = (labels + 1) % num_classes

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

# --- 4. EVALUATION LOOP ---
def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy