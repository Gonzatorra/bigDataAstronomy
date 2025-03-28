import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

transform = transforms.Compose([
    transforms.ToTensor(),          # Convertir las imágenes a tensores
    transforms.Normalize(mean=[0.0645, 0.0423, 0.0293], std=[0.0810, 0.0660, 0.0439]) 
    #media y la desviación estándar de los píxeles en cada uno de los tres canales de color
    #(rojo, verde y azul) de las imágenes.
])

# Cargar los conjuntos de entrenamiento y prueba
train_dataset = datasets.ImageFolder(root=config.IMAGES_PATH+"/train", transform=transform)
test_dataset = datasets.ImageFolder(root=config.IMAGES_PATH+"/test", transform=transform)

# Crear los DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#En el conjunto de prueba, no es necesario mezclar las imágenes porque la evaluación no debe
# depender del orden. Así que deberías cambiarlo a shuffle=False para el test_loader

#Red neuronal CNN básica
class CNN(nn.Module):
    def __init__(self, num_classes=2):  # 2 clases: estrellas y galaxias
        super(CNN, self).__init__()
        
        # Capas convolucionales
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Capa totalmente conectada
        self.fc = nn.Linear(256 * 8 * 8, num_classes)  # Después de 3 capas convolucionales y maxpooling

    def forward(self, x):
        # Pasar por las capas convolucionales
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # Aplanar las características y pasar a la capa FC
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Crear el modelo
model = CNN(num_classes=2)  # 2 clases: estrellas y galaxias

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()  # 2 clases: por ejemplo, "estrella" y "galaxia"
model.to(device)



# Definir el optimizador
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Definir la función de pérdida (CrossEntropyLoss para clasificación)
criterion = nn.CrossEntropyLoss()


writer = SummaryWriter()
# Número de épocas para entrenar
epochs = 100

# Ciclo de entrenamiento
for epoch in range(epochs):
    model.train()  # Configurar el modelo en modo de entrenamiento
    epoch_loss = 0.0

    # Entrenamiento
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Mover datos a la GPU si está disponible

        # Poner a cero los gradientes de los optimizadores
        optimizer.zero_grad()

        # Pasar las imágenes por el modelo
        outputs = model(images)

        # Calcular la pérdida
        loss = criterion(outputs, labels)

        # Retropropagar el error
        loss.backward()

        # Actualizar los pesos del modelo
        optimizer.step()

        epoch_loss += loss.item()
    epoch_loss /= len(train_loader.dataset)
    writer.add_scalar("Loss/train", epoch_loss, epoch)


    # Evaluación en el conjunto de test
    model.eval()  # Configurar el modelo en modo de evaluación
    epoch_loss_test = 0.0
    with torch.no_grad():  # Desactivar los gradientes para la evaluación
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Pasar las imágenes por el modelo
            outputs = model(images)

            # Calcular la pérdida
            loss = criterion(outputs, labels)
            epoch_loss_test += loss.item()
    epoch_loss_test /= len(test_loader.dataset)
    writer.add_scalar("Loss/test", epoch_loss_test, epoch)

    if (epoch + 1) % 10 == 0:
    # Imprimir la pérdida media de entrenamiento y test
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {epoch_loss_test:.4f}")

writer.close()