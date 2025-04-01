import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from cnn_se import CNNWithSE  # Asegúrate de que esta clase esté correctamente definida

# Función para realizar el entrenamiento y validación
def get_hyperparameters(train_loader, validation_loader, model, optimizer, criterion, epochs, device, writer):
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        # Entrenamiento
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Evaluar en validación
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

        validation_loss /= len(validation_loader)
        writer.add_scalar("Loss/validation", validation_loss, epoch)

    return validation_loss  # Optuna minimizará esto

# Función Objetivo para Optuna
def objective(trial, train_loader, validation_loader):
    # Definir espacio de búsqueda para los hiperparámetros
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)  #Different learning rates (range of values)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])  #Different batches size (specific values)
    epochs = trial.suggest_int("epochs", 10, 100)  #Different amount of epochs (range of values)

    model = CNNWithSE(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #Define the optimizer and the criterion
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter()

    # Llamar a la función de entrenamiento y validación
    validation_loss = get_hyperparameters(train_loader, validation_loader, model, optimizer, criterion, epochs, device, writer)

    # Finalizar el escritor
    writer.close()

    return validation_loss  # Optuna minimizará la pérdida en el conjunto de validación

# Crear estudio y encontrar los mejores hiperparámetros
def optimize_hyperparameters(train_loader, validation_loader):
    study = optuna.create_study(direction="minimize")  # Queremos minimizar la pérdida de validación
    study.optimize(lambda trial: objective(trial, train_loader, validation_loader), n_trials=15, n_jobs=4)  # Número de pruebas de hiperparámetros

    return study.best_params
