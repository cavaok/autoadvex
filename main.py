import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from helper import create_diffuse_one_hot, set_equal_confusion
from data import get_mnist_loaders, get_fashion_mnist_loaders
from supabase_logger import log_experiment_result
import argparse

print('running main.py')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argparse setup
parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--encoder_path', type=str, default='models/encoder_1_True_digit.pth', help='Model path')
parser.add_argument('--decoder_path', type=str, default='models/decoder_1_True_digit.pth', help='Model path')
parser.add_argument('--mlp_path', type=str, default='models/mlp.pth', help='Model path')
parser.add_argument('--num_confused', type=int, default=2, help='Number of classes with equal confusion')
parser.add_argument('--includes_true', type=str, default='True', help='Whether or not classes includes true class')
parser.add_argument('--num_adversarial_examples', type=int, default=1, help='How many adv exs it will save')
parser.add_argument('--digit_number', type=int, default=0, help='Which number do you want to train advex for')
parser.add_argument('--notes', type=str, default='None', help='Notes about the experimental condition')
parser.add_argument('--dataset', type=str, default="digit", help='Is dataset or fashion')

# Parse args
args = parser.parse_args()
includes_true = args.includes_true == "True"

if args.dataset == "digit":
    _, _, adversarial_loader = get_mnist_loaders()
else:
    _, _, adversarial_loader = get_fashion_mnist_loaders()

# Constants (MUST MATCH TRAINING!!!!)
image_dim = 28 * 28
num_classes = 10
input_dim = image_dim + num_classes
lambda_ = 0.5

# Autoencoder Model (MUST MATCH TRAINING!!!!!)
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


# MLP Model (MUST MATCH TRAINING !!!!!)
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


# Load pre-trained Autoencoder & MLP
encoder.load_state_dict(torch.load(args.encoder_path))
decoder.load_state_dict(torch.load(args.decoder_path))
mlp = MLP().to(device)
mlp.load_state_dict(torch.load(args.mlp_path))

# Move models to device & set to eval mode
encoder = encoder.to(device)
decoder = decoder.to(device)
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

# Get all data from adversarial loader
all_data = list(adversarial_loader)

# Separate into number groupings
if args.digit_number == 0:
    all_data = [data for data in all_data if data[1] == 0]
elif args.digit_number == 1:
    all_data = [data for data in all_data if data[1] == 1]
elif args.digit_number == 2:
    all_data = [data for data in all_data if data[1] == 2]
elif args.digit_number == 3:
    all_data = [data for data in all_data if data[1] == 3]
elif args.digit_number == 4:
    all_data = [data for data in all_data if data[1] == 4]
elif args.digit_number == 5:
    all_data = [data for data in all_data if data[1] == 5]
elif args.digit_number == 6:
    all_data = [data for data in all_data if data[1] == 6]
elif args.digit_number == 7:
    all_data = [data for data in all_data if data[1] == 7]
elif args.digit_number == 8:
    all_data = [data for data in all_data if data[1] == 8]
elif args.digit_number == 9:
    all_data = [data for data in all_data if data[1] == 9]
else:
    print("Using a mix of all digits")


# ADVERSARIAL TRAINING -----------------------------------------------------------------------------------
for i in range(args.num_adversarial_examples):
    # Mutual setup
    image_batch, label_batch = all_data[i % len(all_data)]  # get ith image & label pair
    single_label = label_batch[0].item()
    image_part = image_batch[0].view(1, -1).to(device).clone().detach()
    target_label = set_equal_confusion(single_label, num_classes, args.num_confused, device, includes_true)
    original_y = torch.zeros(1, num_classes, device=device)  # have to do this to log it
    original_y[0, single_label] = 1

    logging_values = {
        "dataset": args.dataset,
        "autoencoder_notes": args.notes,
        "original_x": image_part.clone().detach(),
        "original_y": original_y,
        "digit_number": args.digit_number,
        "target_distribution": target_label.clone().detach(),
        "num_confused": args.num_confused,
        "includes_true": includes_true,
    }

    # Autoencoder specific setup
    image_part.requires_grad_(True)
    label_part = create_diffuse_one_hot(label_batch[0:1]).to(device)

    # MLP specific setup
    mlp_image = image_part.clone().detach().requires_grad_(True)

    # MLP prediction on original image (for logging)
    mlp_prediction = mlp(mlp_image)
    mlp_prediction_label = F.softmax(mlp_prediction, dim=1)
    logging_values["mlp_prediction_label"] = mlp_prediction_label.clone().detach()
    #   WANT TO LOG: mlp_prediction_label as mlp_prediction_label

    # Autoencoder prediction on original image (for logging)
    auto_concat_input = torch.cat((image_part, label_part), dim=1)  # necessary for pass through
    auto_prediction = autoencoder(auto_concat_input)
    auto_prediction_label = F.softmax(auto_prediction[:, image_dim:], dim=1)
    logging_values["auto_prediction_label"] = auto_prediction_label.clone().detach()

    # Save clone of autoencoder output for training loop later
    original = auto_prediction.clone().detach()
    original_image = original[:, :image_dim]

    # Save MLP target distribution (don't think this is necessary anymore but leaving for now)
    mlp_target_label = target_label

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

    # Save adversarial example and prediction (for logging)
    autoadvex_concat = torch.cat((image_part.view(1, -1), label_part), dim=1)
    autoadvex_prediction = autoencoder(autoadvex_concat)
    autoadvex_prediction_label = F.softmax(autoadvex_prediction[:, image_dim:], dim=1)

    # Calculate label divergence from target distribution (for logging)
    autoadvex_label_divergence = F.kl_div(autoadvex_prediction_label.log(), target_label, reduction='sum')

    # Calculating MSE and Frobenius norm distances (for logging)
    autoadvex_mse = F.mse_loss(image_part.view(1, -1), original_image.view(1, -1))
    autoadvex_frob = torch.norm(image_part.view(1, -1) - original_image.view(1, -1), p='fro')

    logging_values.update({
        "autoadvex_x_hat": image_part.clone().detach(),
        "autoadvex_y_hat": autoadvex_prediction_label.clone().detach(),
        "autoadvex_label_kld": autoadvex_label_divergence,
        "autoadvex_mse": autoadvex_mse.item(),
        "autoadvex_frob": autoadvex_frob.item(),
    })

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

    # Save adversarial example and prediction (for logging)
    with torch.no_grad():
        mlpadvex_prediction = mlp(mlp_image)
        mlpadvex_prediction_label = F.softmax(mlpadvex_prediction, dim=1)

    # Calculating label divergence from target label (for logging)
    mlpadvex_label_divergence = F.kl_div(mlpadvex_prediction_label.log(), target_label, reduction='sum')

    # Calculating MSE and Frobenius norm distances (for logging)
    mlpadvex_mse = F.mse_loss(mlp_image.view(1, -1), original_image.view(1, -1))
    mlpadvex_frob = torch.norm(mlp_image.view(1, -1) - original_image.view(1, -1), p='fro')

    logging_values.update({
        "mlpadvex_x_hat": mlp_image.clone().detach(),
        "mlpadvex_y_hat": mlpadvex_prediction_label.clone().detach(),
        "mlpadvex_label_kld": mlpadvex_label_divergence,
        "mlpadvex_mse": mlpadvex_mse.item(),
        "mlpadvex_frob": mlpadvex_frob.item(),
    })

    # Final logging
    try:
        success = log_experiment_result(**logging_values)
        if success:
            print(f"Successfully logged results for iteration {i}")
        else:
            print(f"Failed to log results for iteration {i}")
    except Exception as e:
        print(f"Error logging results for iteration {i}: {str(e)}")




