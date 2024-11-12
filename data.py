from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def get_mnist_loaders():
    # Loading in MNIST data preparing datasets and loaders
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    adversarial_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_loader, test_loader, adversarial_loader


def get_fashion_mnist_loaders():
    """
    Returns:
        tuple: (train_loader, test_loader, adversarial_loader)
        Each loader contains fashion items labeled 0-9:
        0: T-shirt/top
        1: Trouser
        2: Pullover
        3: Dress
        4: Coat
        5: Sandal
        6: Shirt
        7: Sneaker
        8: Bag
        9: Ankle boot
    """
    # Loading in Fashion-MNIST data preparing datasets and loaders
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    adversarial_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_loader, test_loader, adversarial_loader




