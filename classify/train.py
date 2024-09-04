import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.optim as optim
from dataset import RMBDataset
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os, glob

BATCH_SIZE = 16


train_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

valid_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

dic = "./classify/test"
split_dir = dic
print(split_dir)
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

print(type(train_data))

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)


model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def save_model(model, epoch, acc, path=dic):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, f'model_epoch_{epoch}_acc{acc}.pth'))


def train_and_validate_model(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        print(len(train_loader))
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        epoch_loss = running_loss / total_samples
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')

        if epoch % 5 == 0:
            accuracy = validate_model()
            save_model(model, epoch+1, accuracy)

def validate_model():
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / total_samples
    accuracy = 100 * correct / total
    print(f'Validation Loss: {average_loss:.4f}, Accuracy on validation set: {accuracy:.2f}%')
    return accuracy


train_and_validate_model(100)
