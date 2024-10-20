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
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU()
)

decoder = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, input_dim),
    nn.Sigmoid()
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
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

num_epochs = 15
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
        loss = criterion(outputs, targets)

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
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        _, predicted = outputs[:, -10:].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.to(device)).sum().item()

print(f'Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100. * correct / total:.2f}%')

# Adversarial Example Training
encoder.eval()
decoder.eval()

# Creating this loader so I can go grab 1 image
adversarial_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Ensure model parameters are not updated during adversarial training
for param in encoder.parameters():
    param.requires_grad = False
for param in decoder.parameters():
    param.requires_grad = False

# to make it less angry
image = None
diffuse_label = None
original_label = None
single_input = None
single_label = None

# Grab a single image and label
for images, labels in adversarial_loader:
    image = images.view(images.size(0), -1).to(device)
    diffuse_label = create_diffuse_one_hot(labels).to(device)
    single_input = torch.cat((image, diffuse_label), dim=1)
    single_label = labels.item()  # save for later
    break

# Pass single image and label through autoencoder
reconstructed = autoencoder(single_input)

# Saving a clone in variable called original
original = reconstructed.clone().detach()
original_label = original[:, image_dim:]
original_image = original[:, :image_dim]

# Setting target label
true_class = single_label
classes = list(range(10))
classes.remove(true_class)
random_class = np.random.choice(classes)
target_label = torch.zeros(1, num_classes, device=device)  # Initialize with zeros
target_label[0, true_class] = 0.5
target_label[0, random_class] = 0.5

# Preparing the "first guess" which consists of
#   original        -> Reconstruction of the single image and label we grabbed
#   diffuse_label   -> Diffuse Vector label from calling create_diffuse_one_hot() earlier
first_guess = original.clone().detach()
first_guess[:, image_dim:] = diffuse_label

# Cloning first_guess as the input to be modified for adversarial training
input_adv = first_guess.clone().detach()
input_adv.requires_grad_(True)  # making sure requires grad is TRUE

# Params
optimizer = optim.Adam([input_adv], lr=0.1)
train_loops = 300
output = None  # will use this later

for loop in range(train_loops):
    # Forward pass
    output = autoencoder(input_adv)

    # turning into probability distribution before doing kld
    output_label_probs = F.softmax(output[:, image_dim:], dim=1)

    # Loss calc
    label_loss = nn.functional.kl_div(output_label_probs.log(), target_label)
    image_loss = nn.functional.mse_loss(output[:, :image_dim], original_image)
    loss = label_loss * 50 + image_loss

    # Prints the losses
    print(f"Adversarial Training Loop {loop + 1}/{train_loops}:")
    print(f"  Label Loss: {label_loss.item():.4f}")
    print(f"  Image Loss: {image_loss.item():.4f}")
    print(f"  Total Loss: {loss.item():.4f}")

    # Backprop and optim step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Visualize the result - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create the adversarial_figures directory if it doesn't exist
os.makedirs('adversarial_figures', exist_ok=True)

# Converting to numpy arrays
#   first_guess_image -> using the first_guess that was cloned, saved, and unmodified to show the input
#   adversarial_image -> using input_adv as that is what was modified iteratively and results in our adv ex
first_guess_image = first_guess[:, :image_dim].detach().view(28, 28).cpu().numpy()
first_guess_label = first_guess[:, image_dim:].detach().cpu().numpy()
adversarial_image = input_adv[:, :image_dim].detach().view(28, 28).cpu().numpy()
adversarial_label = input_adv[:, image_dim:].detach().cpu().numpy()

# Create the visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# First guess (input)
axes[0, 0].imshow(first_guess_image, cmap='gray')
axes[0, 0].set_title('Input Image')
axes[0, 1].bar(range(10), first_guess_label[0])
axes[0, 1].set_title('Diffuse Prior Vector')

# Adversarial example
axes[1, 0].imshow(adversarial_image, cmap='gray')
axes[1, 0].set_title('Adversarial Image')
axes[1, 1].bar(range(10), adversarial_label[0])
axes[1, 1].set_title('Adversarial Label Vector')

# Save the figure
filepath = os.path.join('adversarial_figures', 'adversarial_example.png')
plt.savefig(filepath)
plt.close(fig)  # Close the figure to free up memory

print(f"Adversarial example visualization saved to {filepath}")

print("Target label was:", target_label.cpu().numpy().round(3))

