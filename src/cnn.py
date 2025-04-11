import torch
import torch.nn as nn
import torch.nn.functional as F
import checkpoint_utils
import set_seed


set_seed.set_seed(132)

#CNN basic neural network
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        #Fully connected layer
        self.fc = nn.Linear(256 * 8 * 8, num_classes)  #After 3 convolutional layer and maxpooling (see below)
        #256 because from the last convolutional layer we ger 256 channel
        #8*8 because by each max pooling the image is reduced the half so: 64/2/2/2 = 8
    def forward(self, x):
        #Go through all the convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        #Flat the characteristics and use it in de fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(train_loader, test_loader, model, optimizer, criterion, epochs, device, writer, checkpoint_path):
    start_epoch = 0
    #Load the checkpoint if there is any
    try:
        start_epoch, _ = checkpoint_utils.load_checkpoint(model, optimizer, checkpoint_path)
    
    except FileNotFoundError:
        print("No checkpoint found. Training from zero.")

    
    for epoch in range(start_epoch, epochs):
        model.train() 
        epoch_loss = 0.0

        #Training
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

        #Evaluation in the test set
        model.eval() 
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
