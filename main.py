import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from helper import create_diffuse_one_hot, visualize_adversarial_comparison, set_equal_confusion
from data import get_mnist_loaders
import argparse
import wandb

print('running main.py')

# Set up argument parsing at the top of the script
parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--num_confused', type=int, default=2, help='Number of classes with equal confusion')
parser.add_argument('--includes_true', type=str, default='True', help='Whether or not classes includes true class')
parser.add_argument('--num_adversarial_examples', type=int, default=1, help='How many adv exs it will save')
parser.add_argument('--wandb_project', type=str, default='autoadvex', help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default='cavaokcava', help='WandB entity/username')
parser.add_argument('--notes', type=str, default='', help='Notes about the experimental condition')

# Parse args
args = parser.parse_args()
includes_true = args.includes_true == "True"

run_name = f"{args.notes}_{args.num_confused}_confused_{args.includes_true}"
wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    name=run_name,
    config={
        "num_confused": args.num_confused,
        "includes_true": includes_true,
        "notes": args.notes,
        "num_examples": args.num_adversarial_examples
    }
)

# Get the adversarial loader
_, _, adversarial_loader = get_mnist_loaders()

# Constants (must match training exactly)
image_dim = 28 * 28
num_classes = 10
input_dim = image_dim + num_classes
lambda_ = 0.5

# Autoencoder Model definitions (must match training exactly)
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


# MLP Model definition (must match training exactly)
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


# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained autoencoder
encoder.load_state_dict(torch.load('models/encoder.pth'))
decoder.load_state_dict(torch.load('models/decoder.pth'))

# Load the trained MLP
mlp = MLP().to(device)
mlp.load_state_dict(torch.load('models/mlp.pth'))

# Move models to device
encoder = encoder.to(device)
decoder = decoder.to(device)

# Set all models to eval mode
encoder.eval()
decoder.eval()
mlp.eval()

# Ensure no update in model params
for param in encoder.parameters():
    param.requires_grad = False
for param in decoder.parameters():
    param.requires_grad = False
for param in mlp.parameters():
    param.requires_grad = False


all_data = list(adversarial_loader)


# ADVERSARIAL TRAINING =================================================================================
for i in range(args.num_adversarial_examples):
    # MUTUAL SET UP FOR AUTOENCODER AND MLP - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    image_batch, label_batch = all_data[i % len(all_data)]  # get ith image

    # AUTOENCODER: image & label set up
    image_part = image_batch[0].view(1, -1).to(device).clone().detach()
    image_part.requires_grad_(True)
    label_part = create_diffuse_one_hot(label_batch[0:1]).to(device)
    single_label = label_batch[0].item()

    # MLP: image set up & grab label prediction for visualization later
    mlp_image = image_part.clone().detach().requires_grad_(True)
    mlp_IMAGE_D = image_part.clone().detach().view(28, 28).cpu().numpy()  # NEED (IMAGE D)
    mlp_label_d = mlp(mlp_image)
    mlp_label_d_probs = F.softmax(mlp_label_d, dim=1)
    mlp_label_d = mlp_label_d_probs.detach().cpu().numpy()  # NEED (BAR CHART d)

    # AUTOENCODER: grab label prediction for visualization later
    auto_concat_input = torch.cat((image_part, label_part), dim=1)
    auto_IMAGE_A_label_a = autoencoder(auto_concat_input)
    auto_label_a_probs = F.softmax(auto_IMAGE_A_label_a[:, image_dim:], dim=1)
    first_image = image_part.clone().detach().view(28, 28).cpu().numpy()  # NEED (IMAGE A)
    # first_label = label_part.clone().detach().cpu().numpy()
    # reconstructed_image_part = auto_IMAGE_A_label_a[:, :image_dim].detach().view(28, 28).cpu().numpy()
    auto_label_a = auto_label_a_probs.detach().cpu().numpy()  # NEED (BAR CHART a)

    # AUTOENCODER: save a clone of output for training loop later
    original = auto_IMAGE_A_label_a.clone().detach()
    original_image = original[:, :image_dim]

    # MLP & AUTOENCODER: set up same target label
    target_label = set_equal_confusion(single_label, num_classes, args.num_confused, device, includes_true)
    mlp_target_label = target_label

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # AUTOENCODER ADVERSARIAL TRAINING LOOP - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    optimizer = optim.Adam([image_part], lr=0.01)
    train_loops = 300

    for loop in range(train_loops):
        current_input = torch.cat((image_part.view(1, -1), label_part), dim=1)

        output = autoencoder(current_input)

        output_label_probs = F.softmax(output[:, image_dim:], dim=1)
        print(f"  Output probs: {output_label_probs.detach().cpu().numpy().round(3)}")

        label_loss = nn.functional.kl_div(output_label_probs.log(), target_label)
        image_loss = nn.functional.mse_loss(image_part.view(1, -1), original_image)

        loss = image_loss + lambda_ * label_loss

        print(f"Adversarial Training Loop {loop + 1}/{train_loops}:")
        print(f"  Label Loss: {label_loss.item():.4f}")
        print(f"  Image Loss: {image_loss.item():.4f}")
        print(f"  Total Loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        print(f"  Image grad max: {image_part.grad.abs().max().item() if image_part.grad is not None else 'None'}")
        optimizer.step()

        with torch.no_grad():
            image_part.data.clamp_(0, 1)

    # Grab final image & label for visualization
    final_image = image_part.clone().detach().view(28, 28).cpu().numpy()  # NEED (IMAGE C)
    # final_label = label_part.clone().detach().cpu().numpy()
    # Grab final label by passing through autoencoder
    concat_final = torch.cat((image_part.view(1, -1), label_part), dim=1)  # Add view(1, -1) here
    final_output = autoencoder(concat_final)
    final_label_probs = F.softmax(final_output[:, image_dim:], dim=1)
    # final_output_image = final_output[:, :image_dim].detach().view(28, 28).cpu().numpy()
    final_output_label = final_label_probs.detach().cpu().numpy()  # NEED (BAR CHART c)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # MLP ADVERSARIAL TRAINING LOOP - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    mlp_optimizer = optim.Adam([mlp_image], lr=0.01)

    train_loops = 300
    for loop in range(train_loops):
        output = mlp(mlp_image)
        probs = F.softmax(output, dim=1)

        mlp_label_loss = F.kl_div(probs.log(), mlp_target_label)
        mlp_image_loss = F.mse_loss(mlp_image, image_part)
        mlp_loss = mlp_image_loss + lambda_ * mlp_label_loss

        mlp_optimizer.zero_grad()
        mlp_loss.backward()

        if loop % 50 == 0:
            print(f"\nMLP Step {loop + 1}/{train_loops}:")
            print(f"  Current probs: {probs.detach().cpu().numpy().round(3)}")
            print(f"  Target probs: {mlp_target_label.cpu().numpy().round(3)}")
            print(f"  Label Loss: {mlp_label_loss.item():.4f}")
            print(f"  Image Loss: {mlp_image_loss.item():.4f}")
            print(f"  Total Loss: {mlp_loss.item():.4f}")
            print(f"  Image grad max: {mlp_image.grad.abs().max().item()}")

        mlp_optimizer.step()

        with torch.no_grad():
            mlp_image.data.clamp_(0, 1)

    # Get final MLP predictions
    with torch.no_grad():
        # original_mlp_output = mlp(image_part)
        mlp_label_f = mlp(mlp_image)
        mlp_IMAGE_F = mlp_image.clone().detach().view(28, 28).cpu().numpy()  # NEED (IMAGE F)
        # original_mlp_probs = F.softmax(original_mlp_output, dim=1)
        mlp_label_f_probs = F.softmax(mlp_label_f, dim=1)
        mlp_label_f = mlp_label_f_probs.detach().cpu().numpy()  # NEED (BAR CHART F)

        '''
        print("\nFinal MLP Results:")
        print(f"Original predictions: {original_mlp_probs.cpu().numpy().round(3)}")
        print(f"Target distribution: {mlp_target_label.cpu().numpy().round(3)}")
        print(f"Adversarial predictions: {adversarial_mlp_probs.cpu().numpy().round(3)}")
        '''

    # Calculate distances for autoencoder (between A and C)
    auto_orig = torch.tensor(first_image).flatten()  # A
    auto_pert = torch.tensor(final_image).flatten()  # C
    auto_distances = {
        'Euclidean': float(torch.norm(auto_orig - auto_pert).cpu()),
        'MSE': float(F.mse_loss(auto_orig, auto_pert).cpu()),
    }

    # Calculate distances for MLP (between D and F)
    mlp_orig = torch.tensor(mlp_IMAGE_D).flatten()  # D
    mlp_pert = torch.tensor(mlp_IMAGE_F).flatten()  # F
    mlp_distances = {
        'Euclidean': float(torch.norm(mlp_orig - mlp_pert).cpu()),
        'MSE': float(F.mse_loss(mlp_orig, mlp_pert).cpu()),
    }

    # Collect data after adversarial training
    visualization_data = {
        'images': {
            'A': first_image,
            'C': final_image,
            'D': mlp_IMAGE_D,
            'F': mlp_IMAGE_F
        },
        'probabilities': {
            'a': auto_label_a,
            'c': final_output_label,
            'd': mlp_label_d,
            'f': mlp_label_f
        },
        'distances': {
            'auto': auto_distances,
            'mlp': mlp_distances
        }
    }

    fig = visualize_adversarial_comparison(
        images=visualization_data['images'],
        probabilities=visualization_data['probabilities'],
        distances=visualization_data['distances'],
        return_fig=True
    )

    wandb.log({
        f"example_{i}/comparison": wandb.Image(
            fig,
            caption=f"Confused {args.num_confused} classes {'with' if includes_true else 'without'} true class - {args.notes}"
        ),
        f"example_{i}/auto_euclidean": visualization_data['distances']['auto']['Euclidean'],
        f"example_{i}/auto_mse": visualization_data['distances']['auto']['MSE'],
        f"example_{i}/mlp_euclidean": visualization_data['distances']['mlp']['Euclidean'],
        f"example_{i}/mlp_mse": visualization_data['distances']['mlp']['MSE']
    })

wandb.finish()
