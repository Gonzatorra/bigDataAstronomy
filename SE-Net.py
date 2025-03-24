import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

# Bloque SE (Squeeze and Excitation)
class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

# Modelo de CNN con atención
class CNNWithSE(nn.Module):
    def __init__(self, num_classes=2, input_channels=3):
        super(CNNWithSE, self).__init__()
        # Capas convolucionales iniciales
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Capa de atención SE después de cada convolución
        self.se1 = SE_Block(64)
        self.se2 = SE_Block(128)
        self.se3 = SE_Block(256)

        # Capa de pooling
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling con kernel 2x2 y stride 2

        # Capa totalmente conectada (ajustada para entradas de 64x64)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # 256 canales y tamaño 8x8
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Aplicar las capas convolucionales, pooling y atención
        x = self.pool(F.relu(self.conv1(x)))
        x = self.se1(x)  # Aplicar la atención SE
        x = self.pool(F.relu(self.conv2(x)))
        x = self.se2(x)  # Aplicar la atención SE
        x = self.pool(F.relu(self.conv3(x)))
        x = self.se3(x)  # Aplicar la atención SE

        # Aplanar y pasar por las capas totalmente conectadas
        x = x.view(-1, 256 * 8 * 8)  # Ajustado para imágenes de 64x64
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Ajusta los valores según tu dataset
])

train_data = datasets.ImageFolder('/home/haizeagonzalez/bigData/imagenesRecortadasTrain', transform=transform)
val_data = datasets.ImageFolder('/home/haizeagonzalez/bigData/imagenesRecortadasTest', transform=transform)

# Crear DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)







# Crear el modelo
model = CNNWithSE(num_classes=2)  # 2 clases: estrella y galaxia

# Definir el optimizador
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Definir la función de pérdida (entropía cruzada para clasificación)
criterion = nn.CrossEntropyLoss()







# Función de entrenamiento
def train(model, train_loader, optimizer, criterion, device):
    model.train()  # Poner el modelo en modo entrenamiento
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Paso hacia adelante
        optimizer.zero_grad()  # Limpiar los gradientes
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Paso hacia atrás y optimización
        loss.backward()
        optimizer.step()

        # Estadísticas
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(train_loader), accuracy

# Función de validación
def validate(model, val_loader, criterion, device):
    model.eval()  # Poner el modelo en modo evaluación
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No calcular gradientes durante la validación
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Paso hacia adelante
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Estadísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(val_loader), accuracy










# Configuración del dispositivo (GPU o CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Número de épocas de entrenamiento
num_epochs = 10

# Ciclo de entrenamiento
for epoch in range(num_epochs):
    # Entrenar
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Validar
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
