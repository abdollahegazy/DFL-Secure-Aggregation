
from .basetrainer import BaseTrainer
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist():
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='data', train=True, transform=mnist_transform, download=True)
    return mnist_dataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    

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
        output = F.log_softmax(x, dim=1)
        return output
    
class MNISTTrainer(BaseTrainer):
    def __init__(self, node_hash:int, 
                 device,
                 optimizer_cls = torch.optim.AdamW,
                 optimizer_kwargs = {'lr': 0.001},
                 loss_fn = nn.CrossEntropyLoss()):
        super().__init__(node_hash, device, optimizer_cls, optimizer_kwargs, loss_fn)
        self.model = Net().to(self.device)
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)

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
        losses = []

        with torch.no_grad():
            for x, y in dloader:
                images, labels = x.to(self.device), y.to(self.device)
                outputs = self.model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # loss
                loss = self.loss_fn(outputs, labels)
                losses.append(loss.item())
        loss = sum(losses) / len(losses)
        accuracy = correct / total
        
        return accuracy, loss

