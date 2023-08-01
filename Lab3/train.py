import torch
from torch import nn
from torch import optim
from torch import argmax as tensor_max
from torch.backends import mps as mpsback
from torch import device, cuda, mps, no_grad
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, ArgumentTypeError, Namespace

from dataloader import LeukemiaLoader
from model import ResNet18, ResNet50, ResNet152


def train(
    epochs: int,
    batch_size: int,
    optimizer: optim,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    train_device: device,
    train_dataset: LeukemiaLoader,
    test_dataset: LeukemiaLoader
) -> None:
    """
    :param epochs: number of epochs
    :param batch_size: batch size
    :param optimizer: optimizer
    :param learning_rate: learning rate
    :param momentum: momentum for SGD
    :param weight_decay: weight decay for optimier
    :param train_device: train device (cuda, mps, cpu)
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    """

    models = {
        'ResNet18': ResNet18(2).to(train_device),
        'ResNet50': ResNet50(2).to(train_device),
        'ResNet152': ResNet152(2).to(train_device),
    }

    accuracy = {
        'train': {key: [0 for _ in range(epochs)] for key in models.keys()},
        'test': {key: [0 for _ in range(epochs)] for key in models.keys()}
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for model_name, model in models.items():
        if train_device.type == 'mps':
            mps.empty_cache()
        elif train_device.type == 'cuda':
            cuda.empty_cache()

        model_optim = optimizer(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

        for epoch in tqdm(range(epochs)):
            model.train()
            train_acc, test_acc = 0.0, 0.0

            for data, label in train_loader:
                inputs = data.to(train_device)
                labels = label.to(train_device).long()

                prediction = model(inputs)
                loss = nn.CrossEntropyLoss()(prediction, labels)

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

                train_acc += (tensor_max(prediction, dim=1) == labels).sum().item()

            accuracy['train'][model_name][epoch] = train_acc * \
                100 / len(train_loader.dataset)

            model.eval()
            with no_grad():
                for data, label in test_loader:
                    inputs = data.to(train_device)
                    labels = label.to(train_device).long()

                    prediction = model(inputs)

                    test_acc += (tensor_max(prediction, dim=1) == labels).sum().item()

                accuracy['test'][model_name][epoch] = test_acc * \
                    100 / len(test_loader.dataset)

    longest = len(max(accuracy['train'].keys(), key=len)) + 6
    for train_or_test in accuracy.keys():
        for model_name in accuracy[train_or_test]:
            spaces = ''.join(
                [' ' for _ in range(longest - len(f'{model_name}_{train_or_test}'))])
            print(
                f'{model_name}_{train_or_test}: {spaces}{max(accuracy[train_or_test][model_name]):.2f} %')

    show_result(epochs, accuracy)
    save_model(Path('./storage'), models, accuracy)


def show_result(
    epoch: int,
    accuracy: dict
) -> None:
    plt.figure(0)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy curve')

    for train_or_test in accuracy.keys():
        for model_name in accuracy[train_or_test]:
            plt.plot(
                range(epoch),
                accuracy[train_or_test][model_name],
                label=f'{model_name}_{train_or_test}'
            )

    plt.legend(loc='lower right')
    plt.show()


def save_model(root_path: Path, models: Dict[str, nn.Module], accuracy: dict):
    folder_name = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    folder_path = root_path.joinpath(folder_name)
    folder_path.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        max_acc = max(accuracy['test'][model_name])
        torch.save(model, folder_path.joinpath(f'{model_name}-{max_acc}.pt'))


def check_optimizer_type(value: str) -> optim:
    OPTIM_STR_DICT = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adamw': optim.Adam,
        'adamax': optim.Adamax
    }

    try:
        return OPTIM_STR_DICT[value]
    except:
        raise ArgumentTypeError(f'Does not suppot optimizer {value}.')


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('-o', '--optimizer', default='sgd',
                        type=check_optimizer_type, help='Optimizer')
    parser.add_argument('-r', '--learning_rate', default=0.1,
                        type=float, help='Learning rate')
    parser.add_argument('-m', '--momentum', default=0.9,
                        type=float, help='Momentum for SGD')
    parser.add_argument('-wd', '--weight_decay', default=1e-4,
                        type=float, help='Weight dacay for optimizer')
    return parser.parse_args()


def main():
    args = parse_argument()
    epochs = args.epochs
    batch_size = args.batch_size
    optimizer = args.optimizer
    learning_rate = args.learning_rate
    momentum = args.momentum
    weight_decay = args.weight_decay

    if cuda.is_available():
        train_device_name = 'cuda'
        train_device = device('cuda')
    elif mpsback.is_available():
        train_device_name = 'mps'
        train_device = device('mps')
    else:
        train_device_name = 'cpu'
        train_device = device('cpu')

    print(f'Epochs:           {epochs}')
    print(f'Batch size:       {batch_size}')
    print(f'Learning rate:    {learning_rate}')
    print(f'Momentum:         {momentum}')
    print(f'Weight decay:     {weight_decay}')
    print(f'Train device:     {train_device_name}')

    train(
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        train_device=train_device,
        train_dataset=LeukemiaLoader('train'),
        test_dataset=LeukemiaLoader('valid')
    )


if __name__ == '__main__':
    main()
