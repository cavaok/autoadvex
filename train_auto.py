import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import os
from helper import create_diffuse_one_hot
from data import get_mnist_loaders, get_fashion_mnist_loaders
import argparse

print('running train_auto.py')

parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--num_iters', type=int, default=2, help='Number of iterations of the autoencoder')
parser.add_argument('--sum_losses', type=str, default="True", help='If True sum losses if False take last')
parser.add_argument('--dataset', type=str, default="digit", help='Is dataset or fashion')

args = parser.parse_args()
sum_losses = args.sum_losses == "True"

# Get data loaders
if args.dataset == "digit":
    train_loader, test_loader, _ = get_mnist_loaders()
else:
    train_loader, test_loader, _ = get_fashion_mnist_loaders()


# Constants
image_dim = 28 * 28
num_classes = 10
input_dim = image_dim + num_classes

# Model definitions
encoder = nn.Sequential(
    nn.Linear(input_dim, 512),
    nn.ELU(),
    nn.Linear(512, 256),
    nn.ELU(),
    nn.Linear(256, 128),
    nn.ELU(),
    nn.Linear(128, 64),
    nn.ELU(),
    nn.Linear(64, 32)
)

decoder = nn.Sequential(
    nn.Linear(32, 64),
    nn.ELU(),
    nn.Linear(64, 128),
    nn.ELU(),
    nn.Linear(128, 256),
    nn.ELU(),
    nn.Linear(256, 512),
    nn.ELU(),
    nn.Linear(512, input_dim)
)


def autoencoder(x):
    encoded = encoder(x)
    decoded = decoder(encoded)
    return decoded


# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = encoder.to(device)
decoder = decoder.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)

# Constants
lambda_ = 0.5
num_epochs = 30
num_iterations = args.num_iters

# Training loop
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    train_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        diffuse_labels = create_diffuse_one_hot(labels).to(device)

        # Initial state
        initial_state = torch.cat((images, diffuse_labels), dim=1)
        current_state = initial_state
        targets = torch.cat((images, torch.eye(num_classes)[labels].to(device)), dim=1)

        iteration_losses = []

        # Iterations loop
        for iteration in range(num_iterations):
            # Get next state
            current_state = autoencoder(current_state)

            # Loss calc
            outputs_label_probs = F.softmax(current_state[:, image_dim:], dim=1)
            image_loss = nn.functional.mse_loss(current_state[:, :image_dim], targets[:, :image_dim])
            label_loss = nn.functional.kl_div(outputs_label_probs.log(), targets[:, image_dim:])

            # Store loss from this iteration
            iteration_loss = image_loss + lambda_ * label_loss
            iteration_losses.append(iteration_loss)

        # Choose final loss based on sum_losses flag
        if sum_losses:
            # Sum all losses
            total_batch_loss = sum(iteration_losses)
        else:
            # Take only the final loss
            total_batch_loss = iteration_losses[-1]

        # Backprop & optimization step
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        train_loss += total_batch_loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                  f"Loss: {total_batch_loss.item():.4f}")

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}')

# Evaluation loop
encoder.eval()
decoder.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.size(0), -1).to(device)
        diffuse_labels = create_diffuse_one_hot(labels).to(device)
        inputs = torch.cat((images, diffuse_labels), dim=1)
        targets = torch.cat((images, torch.eye(num_classes)[labels].to(device)), dim=1)

        outputs = autoencoder(inputs)
        outputs_label_probs = F.softmax(outputs[:, image_dim:], dim=1)
        image_loss = nn.functional.mse_loss(outputs[:, :image_dim], targets[:, :image_dim])
        label_loss = nn.functional.kl_div(outputs_label_probs.log(), targets[:, image_dim:])

        loss = image_loss + lambda_ * label_loss
        test_loss += loss.item()

        _, predicted = outputs[:, -10:].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.to(device)).sum().item()

print(f'Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100. * correct / total:.2f}%')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the trained models
torch.save(encoder.state_dict(), f'models/encoder_{args.num_iters}_{args.sum_losses}_{args.dataset}.pth')
torch.save(decoder.state_dict(), f'models/decoder_{args.num_iters}_{args.sum_losses}_{args.dataset}.pth')
print(f"Models saved to models/encoder_{args.num_iters}_{args.sum_losses}_{args.dataset}.pth and models/decoder_{args.num_iters}_{args.sum_losses}_{args.dataset}.pth")

