import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os

# Loading in MNIST data preparing datasets and loaders
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


def create_diffuse_one_hot(labels, num_classes=10, diffuse_value=0.1):
    diffuse_one_hot = np.full((labels.size(0), num_classes), diffuse_value)
    return torch.tensor(diffuse_one_hot, dtype=torch.float32)


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


def visualize_input_output(inputs, outputs, index=0, save_dir='figures'):
    # Create the figures directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    input_image = inputs[index, :image_dim].view(28, 28).cpu().numpy()
    input_label = inputs[index, image_dim:].cpu().numpy()
    output_image = outputs[index, :image_dim].view(28, 28).cpu().detach().numpy()
    output_label = outputs[index, image_dim:].cpu().detach().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(input_image, cmap='gray')
    axes[0, 0].set_title(f'Input Image')
    axes[0, 1].bar(range(10), input_label)
    axes[0, 1].set_title('Diffuse Prior Vector')
    axes[1, 0].imshow(output_image, cmap='gray')
    axes[1, 0].set_title(f'Output Image')
    axes[1, 1].bar(range(10), output_label)
    axes[1, 1].set_title('Output Label Vector')

    # Save the figure instead of showing it
    filename = f'visualization_{index}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)  # Close the figure to free up memory

    print(f"Visualization saved to {filepath}")


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

# Evaluate the model
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
        # turning into probability distribution before doing kld
        outputs_label_probs = F.softmax(outputs[:, image_dim:], dim=1)
        image_loss = nn.functional.mse_loss(outputs[:, :image_dim], targets[:, :image_dim])
        label_loss = nn.functional.kl_div(outputs_label_probs.log(), targets[:, image_dim:])

        loss = image_loss + 10 * label_loss
        test_loss += loss.item()

        _, predicted = outputs[:, -10:].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.to(device)).sum().item()

print(f'Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100. * correct / total:.2f}%')

# Adversarial Example Training
encoder.eval()
decoder.eval()

# Creating this loader so I can go grab 1 image
adversarial_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Ensure model parameters are not updated during adversarial training
for param in encoder.parameters():
    param.requires_grad = False
for param in decoder.parameters():
    param.requires_grad = False

for images, labels in adversarial_loader:
    # Keep original image separate and make it require gradients
    image_part = images.view(images.size(0), -1).to(device).clone().detach()
    image_part.requires_grad_(True)
    label_part = create_diffuse_one_hot(labels).to(device)
    single_label = labels.item()
    break  # only grab one

# Pass single image and label through autoencoder
concat_input = torch.cat((image_part, label_part), dim=1)
reconstructed = autoencoder(concat_input)

# Saving a clone in variable called original
original = reconstructed.clone().detach()
original_image = original[:, :image_dim]

# Save for visualization later
initial_image = original_image.clone().detach().view(28, 28).cpu().numpy()
initial_label = label_part.clone().detach().cpu().numpy()

# Setting target label
true_class = single_label
classes = list(range(10))
classes.remove(true_class)
random_class = np.random.choice(classes)
target_label = torch.zeros(1, num_classes, device=device)  # Initialize with zeros
target_label[0, true_class] = 0.5
target_label[0, random_class] = 0.5

# Params
optimizer = optim.Adam([image_part], lr=0.01)
train_loops = 300

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


# Visualize the training results - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create the adversarial_figures directory if it doesn't exist
os.makedirs('adversarial_figures', exist_ok=True)

# Saving final state and converting to numpy arrays
final_image = image_part.clone().detach().view(28, 28).cpu().numpy()
final_label = label_part.clone().detach().cpu().numpy()

# Create the visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# First guess (input)
axes[0, 0].imshow(initial_image, cmap='gray')
axes[0, 0].set_title('Input Image')
axes[0, 1].bar(range(10), initial_label[0])
axes[0, 1].set_title('Diffuse Prior Vector')

# Adversarial example
axes[1, 0].imshow(final_image, cmap='gray')
axes[1, 0].set_title('Adversarial Image')
axes[1, 1].bar(range(10), final_label[0])
axes[1, 1].set_title('Adversarial Label Vector')

# Save the figure
filepath = os.path.join('adversarial_figures', 'adversarial_training.png')
plt.savefig(filepath)
plt.close(fig)  # Close the figure to free up memory

print(f"Adversarial example training visualization saved to {filepath}")


# Visualization of final adversarial testing - - - - - - - - - - - - - - - - - - - - - -

# Gathering the "final" output of autoencoder with adversarial example
# input_adv.data[:, image_dim:] = diffuse_label  # re-append diffuse prior
concat_final = torch.cat((image_part, label_part), dim=1)
final_output = autoencoder(concat_final)
# final_label_probs = F.softmax(final_output[:, image_dim:], dim=1) might not need this

# Converting to numpy arrays
# adversarial_trained_image = final_image.detach().view(28, 28).cpu().numpy()  # same image as before
# adversarial_trained_label = final_label.detach().cpu().numpy()               # same label as before
final_output_image = final_output[:, :image_dim].detach().view(28, 28).cpu().numpy()      # final autoencoder output
final_output_label = final_output[:, image_dim:].detach().cpu().numpy()                   # final autoencoder output

# Create the visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# FGI -> Adversarial output (input_adv)
axes[0, 0].imshow(final_image, cmap='gray')
axes[0, 0].set_title('Adversarial Trained Image')
axes[0, 1].bar(range(10), final_label[0])
axes[0, 1].set_title('Adversarial Trained Label')

# AE -> autoencoder output
axes[1, 0].imshow(final_output_image, cmap='gray')
axes[1, 0].set_title('Final Output Image')
axes[1, 1].bar(range(10), final_output_label[0])
axes[1, 1].set_title('Final Output Label (w/ Softmax)')

# Save the figure
filepath = os.path.join('adversarial_figures', 'adversarial_testing.png')
plt.savefig(filepath)
plt.close(fig)  # Close the figure to free up memory

print(f"Adversarial example training visualization saved to {filepath}")

print("Target label was:", target_label.cpu().numpy().round(3))

