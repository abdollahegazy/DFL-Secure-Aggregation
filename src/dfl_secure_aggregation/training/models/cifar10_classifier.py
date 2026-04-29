from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .model import BaseTrainer


'''
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFAR10Trainer(BaseTrainer):
    def __init__(self, 
                 node_hash: int, 
                 device=None,
                 optimizer_cls = torch.optim.AdamW,
                 optimizer_kwargs = {'lr': 0.001},
                 loss_fn = nn.CrossEntropyLoss()):
        super().__init__(node_hash, device=device)
        self.model = Net().to(self.device)
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
        self.loss_fn = loss_fn

    def train(self, 
              dataset,
              epochs,
              batch_size,
              num_samples: int | None = None,
              shuffle=True):
        dloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.model.train()
        for _ in range(epochs):
            for x,y in dloader:
                inputs, labels = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, 
                 dataset,
                 batch_size:int = 256):
        dloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        loss = 0

        with torch.no_grad():
            for x, y in dloader:
                inputs, labels = x.to(self.device), y.to(self.device)

                # calculate outputs by running images through the network
                outputs = self.model(inputs)
                loss += self.loss_fn(outputs, labels).item()

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        loss = loss / len(dloader)
        return accuracy, loss
    
