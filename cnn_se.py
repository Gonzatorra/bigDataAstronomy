import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


# Bloque Squeeze-and-Excitation (SE)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1)
        y = y.view(b, c)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

# Red convolucional con bloques SE
class CNNWithSE(nn.Module):
    def __init__(self, num_classes):  # 2 clases: estrellas y galaxias
        super(CNNWithSE, self).__init__()
        
        # Capas convolucionales
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Bloques SE
        self.se1 = SELayer(64)
        self.se2 = SELayer(128)
        self.se3 = SELayer(256)

        # Capa totalmente conectada
        self.fc = nn.Linear(256 * 8 * 8, num_classes)  # Después de 3 capas convolucionales y maxpooling

    def forward(self, x):
        # Pasar por las capas convolucionales
        x = F.relu(self.conv1(x))
        x = self.se1(x)  # Aplicar SE después de la primera capa
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = self.se2(x)  # Aplicar SE después de la segunda capa
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = self.se3(x)  # Aplicar SE después de la tercera capa
        x = F.max_pool2d(x, 2)

        # Aplanar las características y pasar a la capa FC
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_model(train_loader, test_loader, model, optimizer, criterion, epochs, device, writer):
    for epoch in range(epochs):
        model.train()  # Modo entrenamiento
        epoch_loss = 0.0

        # Entrenamiento
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader.dataset)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        # Evaluación en el conjunto de test
        model.eval()  # Modo evaluación
        epoch_loss_test = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_loss_test += loss.item()

        epoch_loss_test /= len(test_loader.dataset)
        writer.add_scalar("Loss/test", epoch_loss_test, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {epoch_loss_test:.4f}")