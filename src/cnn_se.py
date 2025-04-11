import torch
import torch.nn as nn
import torch.nn.functional as F
import checkpoint_utils as checkpoint_utils
import set_seed

set_seed.set_seed(132)


#New Squeeze-and-Excitation (SE) block
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        #Squeeze
        y = F.adaptive_avg_pool2d(x, 1)

        #Excitation
        y = y.view(b, c)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1) #Get a value netween 0 and 1

        #Recalibration
        return x * y

#CNN network with SE blocks
class CNNWithSE(nn.Module):
    def __init__(self, num_classes):  #2 classes: stars and galaxies
        super(CNNWithSE, self).__init__()
        
        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        #SE blocks
        self.se1 = SELayer(64)
        self.se2 = SELayer(128)
        self.se3 = SELayer(256)

        #Fully connected layer
        self.fc = nn.Linear(256 * 8 * 8, num_classes)  #After 3 convolutional and maxpooling (see below)

    def forward(self, x):
        #Go through all the convolutional layers
        x = F.relu(self.conv1(x))
        x = self.se1(x)  #Apply SE block to the first layer
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = self.se2(x)  #Apply SE block to the second layer
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = self.se3(x)  #Apply SE block to the third layer
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
        print(f"No checkpoint found. Training from zero.")
    
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
