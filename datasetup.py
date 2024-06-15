import os

import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKER = os.cpu_count()


def create_dataloader(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int=NUM_WORKER
        ) -> tuple[DataLoader, DataLoader, list]:
    
    """creates train dataloader and test DataLoader

    Takes train dir and test dir as input and then convert them
    into Pytorch Datasets and then into Pytorch DataLoaders

    Args:
        train_dir: takes a train directory path as a string
        test_dir: takes a test directory path as a string
        transform: it is a torchvision.transforms method 
        batch_size: batch size to use in dataloader
        num_workers: number of worker for dataloader

    Returns:
        This function will return a tuple consists of
        trainloader testloader and class_names
        (where class_names is a list of target class names)
        
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)

    """

    train_data = datasets.ImageFolder(train_dir, transform=transform)

    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes

    trainloader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    testloader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    return trainloader, testloader, class_names

