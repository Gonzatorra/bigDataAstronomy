import torch
import torch.nn as nn
import torch.nn.functional as F
import checkpoint_utils
import set_seed


set_seed.set_seed(132)

#Red neuronal CNN básica
class CNN(nn.Module):
    def __init__(self, num_classes):
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

def train_model(train_loader, test_loader, model, optimizer, criterion, epochs, device, writer, checkpoint_path):
    start_epoch = 0
    #If there is a checkpoint
    try:
        start_epoch, _ = checkpoint_utils.load_checkpoint(model, optimizer, checkpoint_path)
    
    except FileNotFoundError:
        print("No checkpoint found. Training from zero.")

    
    for epoch in range(start_epoch, epochs):
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
            checkpoint_utils.save_checkpoint(epoch, model, optimizer, epoch_loss_test, checkpoint_path)
