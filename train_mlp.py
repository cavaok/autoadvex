import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from data import get_mnist_loaders
print('running train_mlp.py')

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader, _ = get_mnist_loaders()

image_dim = 28 * 28
train_epochs = 15


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


# Initialize model and optimizer
model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Convert labels to one-hot
            target = F.one_hot(labels, num_classes=10).float()

            optimizer.zero_grad()
            outputs = model(images)
            # Convert outputs to log probabilities
            log_probs = F.log_softmax(outputs, dim=1)

            loss = F.kl_div(log_probs, target, reduction='batchmean')
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0


def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')


# Train and test the model
train(train_epochs)
test()

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the trained model
torch.save(model.state_dict(), 'models/mlp.pth')
print("Model saved to models/mlp.pth")
