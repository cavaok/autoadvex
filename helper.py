import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def create_diffuse_one_hot(labels, num_classes=10, diffuse_value=0.1):
    diffuse_one_hot = np.full((labels.size(0), num_classes), diffuse_value)
    return torch.tensor(diffuse_one_hot, dtype=torch.float32)


def visualize_input_output(inputs, outputs, index=0, save_dir='figures', image_dim=28*28):
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


def visualize_adversarial(top_image, string1, top_label, string2, bottom_image, string3, bottom_label, string4,
                          title, foldername):
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # original image
    axes[0, 0].imshow(top_image, cmap='gray')
    axes[0, 0].set_title(string1)
    axes[0, 1].bar(range(10), top_label[0])
    axes[0, 1].set_title(string2)

    # autoencoder output
    axes[1, 0].imshow(bottom_image, cmap='gray')
    axes[1, 0].set_title(string3)
    axes[1, 1].bar(range(10), bottom_label[0])
    axes[1, 1].set_title(string4)

    # Save the figure
    filepath = os.path.join(foldername, title)
    plt.savefig(filepath)
    plt.close(fig)  # Close the figure to free up memory

    print(f"Visualization saved to {filepath}")


def fifty_percent_two(single_label, num_classes, device):
    true_class = single_label
    classes = list(range(10))
    classes.remove(true_class)
    random_class = np.random.choice(classes)
    target_label = torch.zeros(1, num_classes, device=device)  # Initialize with zeros
    target_label[0, true_class] = 0.5
    target_label[0, random_class] = 0.5
    return target_label

