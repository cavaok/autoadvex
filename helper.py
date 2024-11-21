import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wandb
import os


def create_diffuse_one_hot(labels, num_classes=10, diffuse_value=0.1):
    diffuse_one_hot = np.full((labels.size(0), num_classes), diffuse_value)
    return torch.tensor(diffuse_one_hot, dtype=torch.float32)


def set_equal_confusion(single_label, num_classes, num_confused, device, includes_true):
    true_class = single_label
    classes = list(range(10))
    classes.remove(true_class)
    target_label = torch.zeros(1, num_classes, device=device)
    loops = num_confused  # assert this case and redefine if the case
    if includes_true:
        target_label[0, true_class] = 1 / num_confused
        loops = num_confused - 1
    for i in range(loops):
        random_class = np.random.choice(classes)
        classes.remove(random_class)
        target_label[0, random_class] = 1 / num_confused

    return target_label


def visualize_adversarial_comparison(images, probabilities, distances, save_path=None, return_fig=False):
    # Create figure w/ portrait orient
    fig = plt.figure(figsize=(10, 13.33))

    # Create grid layout
    gs = gridspec.GridSpec(4, 3, height_ratios=[2, 1, 2, 1], hspace=0.3, wspace=0.3)

    # Helper function for MNIST images
    def plot_mnist(ax, image):
        ax.imshow(image, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    # Helper function for probability distributions
    def plot_probs(ax, probs):
        ax.bar(range(10), probs[0])  # [0] because probs are 2D arrays with shape (1, 10)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(10))

    # Helper function for distance table (placeholder)
    def plot_distance_table(ax, distances):
        metrics = list(distances.keys())  # ['Euclidean', 'MSE']
        values = [f"{distances[m]:.4f}" for m in metrics]  # Format to 4 decimal places
        table = ax.table(
            cellText=[[v] for v in values],
            rowLabels=metrics,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.axis('off')

    # Top row - Autoencoder
    # Original image (A)
    ax1 = plt.subplot(gs[0, 0])
    plot_mnist(ax1, images['A'])

    # Update the autoencoder distance table call
    ax2 = plt.subplot(gs[0, 1])
    plot_distance_table(ax2, distances['auto'])

    # Perturbed image (C)
    ax3 = plt.subplot(gs[0, 2])
    plot_mnist(ax3, images['C'])

    # Probability distributions for autoencoder
    ax4 = plt.subplot(gs[1, 0])
    plot_probs(ax4, probabilities['a'])
    ax5 = plt.subplot(gs[1, 2])
    plot_probs(ax5, probabilities['c'])

    # Bottom row - MLP
    # Original image (D)
    ax6 = plt.subplot(gs[2, 0])
    plot_mnist(ax6, images['D'])

    # Update the MLP distance table call
    ax7 = plt.subplot(gs[2, 1])
    plot_distance_table(ax7, distances['mlp'])

    # Perturbed image (F)
    ax8 = plt.subplot(gs[2, 2])
    plot_mnist(ax8, images['F'])

    # Probability distributions for MLP
    ax9 = plt.subplot(gs[3, 0])
    plot_probs(ax9, probabilities['d'])
    ax10 = plt.subplot(gs[3, 2])
    plot_probs(ax10, probabilities['f'])

    # Add titles
    plt.suptitle("ITERATIVE DENOISING AUTOENCODER", y=0.95)
    fig.text(0.5, 0.5, "MULTILAYER PERCEPTRON", ha='center')

    # Adjust layout
    plt.tight_layout()

    if return_fig:
        return fig
    elif save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()



