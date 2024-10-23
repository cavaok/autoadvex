import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from helper import create_diffuse_one_hot, visualize_input_output, visualize_adversarial, fifty_percent_two
from data import get_mnist_loaders


train_loader, test_loader, adversarial_loader = get_mnist_loaders()

image_dim = 28 * 28
num_classes = 10
input_dim = image_dim + num_classes

encoder = nn.Sequential(
    nn.Linear(input_dim, 512),
    nn.ELU(),
    nn.Linear(512, 256),
    nn.ELU(),
    nn.Linear(256, 128),
    nn.ELU(),
)

decoder = nn.Sequential(
    nn.Linear(128, 256),
    nn.ELU(),
    nn.Linear(256, 512),
    nn.ELU(),
    nn.Linear(512, input_dim)  # ,
    # nn.Sigmoid()
)


def autoencoder(x):
    encoded = encoder(x)
    decoded = decoder(encoded)
    return decoded


# AUTOENCODER TRAINING - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = encoder.to(device)
decoder = decoder.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)

num_epochs = 30

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    train_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        diffuse_labels = create_diffuse_one_hot(labels).to(device)
        inputs = torch.cat((images, diffuse_labels), dim=1)
        targets = torch.cat((images, torch.eye(num_classes)[labels].to(device)), dim=1)

        outputs = autoencoder(inputs)

        # turning into probability distribution before doing kld
        outputs_label_probs = F.softmax(outputs[:, image_dim:], dim=1)
        image_loss = nn.functional.mse_loss(outputs[:, :image_dim], targets[:, :image_dim])
        label_loss = nn.functional.kl_div(outputs_label_probs.log(), targets[:, image_dim:])

        loss = image_loss + 10 * label_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}')

    visualize_input_output(inputs, outputs)

# AUTOENCODER EVALUATION - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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

        loss = image_loss + 10 * label_loss
        test_loss += loss.item()

        _, predicted = outputs[:, -10:].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.to(device)).sum().item()

print(f'Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100. * correct / total:.2f}%')

# ADVERSARIAL EXAMPLE TRAINING - - - - - - - - - - - - - - - - - - - - - - - -
encoder.eval()
decoder.eval()

# Ensure no update in model params
for param in encoder.parameters():
    param.requires_grad = False
for param in decoder.parameters():
    param.requires_grad = False

# Grab information for only ONE image/label pair
for images, labels in adversarial_loader:
    image_part = images.view(images.size(0), -1).to(device).clone().detach()
    image_part.requires_grad_(True)
    label_part = create_diffuse_one_hot(labels).to(device)
    single_label = labels.item()
    break

# Pass image and label through autoencoder
concat_input = torch.cat((image_part, label_part), dim=1)
reconstructed = autoencoder(concat_input)

# Prepping reconstruction for visualization
first_image = image_part.clone().detach().view(28, 28).cpu().numpy()
first_label = label_part.clone().detach().cpu().numpy()
reconstructed_image_part = reconstructed[:, :image_dim].detach().view(28, 28).cpu().numpy()
reconstructed_label_part = reconstructed[:, image_dim:].detach().cpu().numpy()

# Visualize reconstruction
os.makedirs('adversarial_figures', exist_ok=True)
visualize_adversarial(first_image, 'Original Selected Image',
                      first_label, 'Diffuse Label',
                      reconstructed_image_part, 'Reconstructed Output Image',
                      reconstructed_label_part, 'Reconstructed Output Label',
                      'reconstruction.png', 'adversarial_figures')

# Saving a clone for training loop later
original = reconstructed.clone().detach()
original_image = original[:, :image_dim]

# Save for visualizing later
initial_image = original_image.clone().detach().view(28, 28).cpu().numpy()
initial_label = label_part.clone().detach().cpu().numpy()

# Setting up target label
target_label = fifty_percent_two(single_label, num_classes, device)

# Params
optimizer = optim.Adam([image_part], lr=0.01)
train_loops = 300

# Training loops
for loop in range(train_loops):
    current_input = torch.cat((image_part, label_part), dim=1)
    # Forward pass
    output = autoencoder(current_input)

    # turning into probability distribution before doing kld
    output_label_probs = F.softmax(output[:, image_dim:], dim=1)
    print(f"  Output probs: {output_label_probs.detach().cpu().numpy().round(3)}")
    label_loss = nn.functional.kl_div(output_label_probs.log(), target_label)  # reduction='sum')
    image_loss = nn.functional.mse_loss(image_part, original_image)

    loss = image_loss + 10 * label_loss

    # Prints the losses
    print(f"Adversarial Training Loop {loop + 1}/{train_loops}:")
    print(f"  Label Loss: {label_loss.item():.4f}")
    print(f"  Image Loss: {image_loss.item():.4f}")
    print(f"  Total Loss: {loss.item():.4f}")

    # Backprop and optim step
    optimizer.zero_grad()
    loss.backward()
    print(f"  Image grad max: {image_part.grad.abs().max().item() if image_part.grad is not None else 'None'}")

    optimizer.step()

    with torch.no_grad():
        image_part.data.clamp_(0, 1)


# Prepping final state for visualization
final_image = image_part.clone().detach().view(28, 28).cpu().numpy()
final_label = label_part.clone().detach().cpu().numpy()

# Visualize adversarial training results
visualize_adversarial(initial_image, 'First Guess Image',
                      initial_label, 'Diffuse Label',
                      final_image, 'Adversarial Image',
                      final_label, 'Diffuse Label',
                      'adversarial_training.png', 'adversarial_figures')

# Get final "test" by passing final state through autoencoder
concat_final = torch.cat((image_part, label_part), dim=1)
final_output = autoencoder(concat_final)

# Converting to numpy arrays
final_output_image = final_output[:, :image_dim].detach().view(28, 28).cpu().numpy()
final_output_label = final_output[:, image_dim:].detach().cpu().numpy()


# Visualize adversarial training results
visualize_adversarial(final_image, 'Adversarial Trained Image',
                      final_label, 'Diffuse Label',
                      final_output_image, 'Reconstructed Image',
                      final_output_label, 'Reconstructed Label Prediction',
                      'adversarial_testing.png', 'adversarial_figures')


