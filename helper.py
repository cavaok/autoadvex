import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def create_diffuse_one_hot(labels, num_classes=10, diffuse_value=0.1):
    diffuse_one_hot = np.full((labels.size(0), num_classes), diffuse_value)
    return torch.tensor(diffuse_one_hot, dtype=torch.float32)


def visualize_adversarial(top_image, string1, top_label, string2, bottom_image, string3, bottom_label, string4,
                          title, folder_name):
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
    filepath = os.path.join(folder_name, title)
    plt.savefig(filepath)
    plt.close(fig)  # Close the figure to free up memory

    print(f"Visualization saved to {filepath}")


def set_equal_confusion(single_label, num_classes, num_confused, device, includes_true=True):
    true_class = single_label
    classes = list(range(10))
    classes.remove(true_class)
    target_label = torch.zeros(1, num_classes, device=device)  # Initialize with zeros
    if includes_true:
        target_label[0, true_class] = 1 / num_confused
        for i in (num_confused - 1):
            random_class = np.random.choice(classes)
            classes.remove(random_class)
            target_label[0, random_class] = 1 / num_confused
    else:
        for i in num_confused:
            random_class = np.random.choice(classes)
            classes.remove(random_class)
            target_label[0, random_class] = 1 / num_confused
    return target_label

