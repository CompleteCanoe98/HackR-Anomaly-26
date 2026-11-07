import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

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
def load_data(partition_id: int, num_partitions: int):
    """Loads a specific shard of the local FEMNIST dataset."""
    dataset = load_from_disk("./local_femnist_data")
    transform = Compose([ToTensor()])

    def apply_transforms(batch):
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch

    # Split the training data among the clients
    train_data = dataset["train"].shard(num_shards=num_partitions, index=partition_id)
    train_data = train_data.with_transform(apply_transforms)
    
    # The test set remains centralized/whole for evaluation
    test_data = dataset["test"].with_transform(apply_transforms)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    return train_loader, test_loader

# --- 3. TRAINING LOOP ---
def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    
    for _ in range(epochs):
        for batch in trainloader:
            # Notice we use the "character" key as discovered earlier
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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