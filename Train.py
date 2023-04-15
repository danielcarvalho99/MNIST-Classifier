import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from Model import LeNet

# Locate device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU")
else:
    device = torch.device("cpu")
    print("CPU")

# MNIST dataset 
batch_size=64

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Train parameters
criterion = nn.CrossEntropyLoss()
model_adam = LeNet().to(device)
optimizer = torch.optim.Adam(model_adam.parameters(), lr=0.05)
n_steps = len(train_loader)
num_epochs = 10


def train(model):
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
        
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def test(model):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

train(model_adam)
test(model_adam)

torch.save(model_adam.state_dict(), "model_mnist.pth")