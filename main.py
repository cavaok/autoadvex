import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import os
from helper import create_diffuse_one_hot, visualize_adversarial, set_equal_confusion
from data import get_mnist_loaders
import argparse
print('running main.py')

# Set up argument parsing at the top of the script
parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--num_confused', type=int, default=2, help='Number of classes with equal confusion')
parser.add_argument('--includes_true', type=str, default='True', help='Whether or not classes includes true class')
parser.add_argument('--num_adversarial_examples', type=int, default=1, help='How many adv exs it will save')

# Parse args immediately - these will be available throughout the script
args = parser.parse_args()
includes_true = args.includes_true == "True"

# Get the adversarial loader (we only need this one now)
_, _, adversarial_loader = get_mnist_loaders()

# Constants (must match training exactly)
image_dim = 28 * 28
num_classes = 10
input_dim = image_dim + num_classes
lambda_ = 0.5

# Model definitions (must match training exactly)
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


# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained models
encoder.load_state_dict(torch.load('models/encoder.pth'))
decoder.load_state_dict(torch.load('models/decoder.pth'))

# Move models to device
encoder = encoder.to(device)
decoder = decoder.to(device)

# Set models to eval mode
encoder.eval()
decoder.eval()

# Ensure no update in model params
for param in encoder.parameters():
    param.requires_grad = False
for param in decoder.parameters():
    param.requires_grad = False

# The rest of your adversarial example generation code remains exactly the same
all_data = list(adversarial_loader)

for i in range(args.num_adversarial_examples):
    # Get the i-th image, wrapping around if needed
    image_batch, label_batch = all_data[i % len(all_data)]

    # Select just the first image/label from this batch
    image_part = image_batch[0].view(1, -1).to(device).clone().detach()  # Add batch dimension with view(1, -1)
    image_part.requires_grad_(True)
    label_part = create_diffuse_one_hot(label_batch[0:1]).to(device)  # Already has batch dimension
    single_label = label_batch[0].item()

    # Pass image and label through autoencoder
    concat_input = torch.cat((image_part, label_part), dim=1)  # Now both tensors are 2D
    reconstructed = autoencoder(concat_input)
    reconstructed_label_probs = F.softmax(reconstructed[:, image_dim:], dim=1)

    # Prepping reconstruction for visualization
    first_image = image_part.clone().detach().view(28, 28).cpu().numpy()
    first_label = label_part.clone().detach().cpu().numpy()
    reconstructed_image_part = reconstructed[:, :image_dim].detach().view(28, 28).cpu().numpy()
    reconstructed_label_part = reconstructed_label_probs.detach().cpu().numpy()

    folder_name = f'adversarial_figures_{args.num_confused}_{str(includes_true)}'
    os.makedirs(folder_name, exist_ok=True)
    visualize_adversarial(first_image, 'Original Selected Image',
                          first_label, 'Diffuse Label',
                          reconstructed_image_part, 'Reconstructed Output Image',
                          reconstructed_label_part, 'Reconstructed Output Label',
                          f'reconstruction_{i}.png', folder_name)
    print(f'reconstruction_{i}.png saved to ' + folder_name)

    # Saving a clone for training loop later
    original = reconstructed.clone().detach()
    original_image = original[:, :image_dim]

    # Setting up target label
    target_label = set_equal_confusion(single_label, num_classes, args.num_confused, device, includes_true)

    # Params
    optimizer = optim.Adam([image_part], lr=0.01)
    train_loops = 300

    # Training loops
    for loop in range(train_loops):
        # Ensure image_part maintains batch dimension throughout the loop
        current_input = torch.cat((image_part.view(1, -1), label_part), dim=1)  # Add view(1, -1) here too
        # Forward pass
        output = autoencoder(current_input)

        # turning into probability distribution before doing kld
        output_label_probs = F.softmax(output[:, image_dim:], dim=1)
        print(f"  Output probs: {output_label_probs.detach().cpu().numpy().round(3)}")
        label_loss = nn.functional.kl_div(output_label_probs.log(), target_label)
        image_loss = nn.functional.mse_loss(image_part.view(1, -1), original_image)  # Add view(1, -1) here

        loss = image_loss + lambda_ * label_loss

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

    # Get final "test" by passing final state through autoencoder
    concat_final = torch.cat((image_part.view(1, -1), label_part), dim=1)  # Add view(1, -1) here
    final_output = autoencoder(concat_final)
    final_label_probs = F.softmax(final_output[:, image_dim:], dim=1)

    # Converting to numpy arrays
    final_output_image = final_output[:, :image_dim].detach().view(28, 28).cpu().numpy()
    final_output_label = final_label_probs.detach().cpu().numpy()

    # Visualize adversarial training results
    visualize_adversarial(final_image, 'Adversarial Trained Image',
                          final_label, 'Diffuse Label',
                          final_output_image, 'Reconstructed Image',
                          final_output_label, 'Reconstructed Label Prediction',
                          f'adversarial_{i}.png', folder_name)
    print(f'adversarial_{i}.png saved to ' + folder_name)

