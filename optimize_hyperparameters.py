import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from torch.utils.tensorboard import SummaryWriter
from cnn_se import CNNWithSE, train_model  # Asegúrate de importar correctamente

def optimize_hyperparameters(train_dataset, validation_dataset, device, checkpoint_path):
    param_grid = {
        "lr": [0.01, 0.001, 0.0001],
        "batch_size": [32, 64, 128],
        "epochs": [10, 20, 30]
    }

    best_valid_loss = float('inf')
    best_params = {}

    for params in ParameterGrid(param_grid):
        print(f"Training with hyperparameters: {params}")

        # Crear DataLoaders con el batch size seleccionado
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=params["batch_size"], shuffle=False)

        # Crear el modelo
        model = CNNWithSE(num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter()

        # Entrenar el modelo y obtener la pérdida en validación
        train_model(train_loader, validation_loader, model, optimizer, criterion, params["epochs"], device, writer, checkpoint_path)

        # Evaluar el modelo
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
        valid_loss /= len(validation_loader.dataset)

        writer.add_scalar("Loss/validation", valid_loss, params["epochs"])
        writer.close()

        print(f"Validation loss: {valid_loss:.4f}")

        # Guardar los mejores hiperparámetros
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_params = params

    print(f"Best hyperparameters: {best_params}")
    return best_params
