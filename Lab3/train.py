import torch
from torch import nn
from torch import optim
from torch import argmax as tensor_max
from torch.backends import mps as mpsback
from torch import device, cuda, mps, no_grad
from torch.utils.data import DataLoader

import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError, Namespace

from dataloader import LeukemiaLoader
from model import ResNet, ResNet18, ResNet50, ResNet152


def train(
    model: ResNet,
    model_name: str,
    epochs: int,
    batch_size: int,
    optimizer: optim,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    train_device: device,
    train_dataset: LeukemiaLoader,
    test_dataset: LeukemiaLoader,
    save_folder: Path,
) -> None:
    """
    :param model: ResNet model
    :param model_name: model name
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

    accuracy = {
        'train': [0 for _ in range(epochs)],
        'test': [0 for _ in range(epochs)]
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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

    for epoch in range(epochs):
        print(f'Epoch {epoch}:')

        model.train()
        train_acc, test_acc = 0.0, 0.0

        for data, label in tqdm(train_loader):
            inputs = data.to(train_device)
            labels = label.to(train_device).long()

            prediction = model(inputs)
            loss = nn.CrossEntropyLoss()(prediction, labels)

            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

            train_acc += (tensor_max(prediction, dim=1) == labels).sum().item()

        accuracy['train'][epoch] = train_acc * 100 / len(train_loader.dataset)

        model.eval()
        with no_grad():
            for data, label in tqdm(test_loader):
                inputs = data.to(train_device)
                labels = label.to(train_device).long()

                prediction = model(inputs)

                test_acc += (tensor_max(prediction, dim=1) == labels).sum().item()

            accuracy['test'][epoch] = test_acc * 100 / len(test_loader.dataset)

    print(f'Train: {max(accuracy["train"])}')
    print(f'Test:  {max(accuracy["test"])}')

    save_model(save_folder, model_name, model, accuracy)
    show_result(model_name, accuracy, save_folder, epochs)


def show_result(
    model_name: str,
    accuracy: dict,
    save_path: Path,
    epoch: int,
) -> None:
    plt.figure(0)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy curve')

    for train_or_test in accuracy.keys():
        plt.plot(
            range(epoch),
            accuracy[train_or_test],
            label=f'{model_name}_{train_or_test}'
        )

    plt.legend(loc='lower right')
    plt.savefig(save_path.joinpath(f'accuracy.png'))


def save_model(save_path: Path, model_name: str, model: nn.Module, accuracy: dict):
    max_acc = max(accuracy['test'])
    save_path.mkdir(parents=True, exist_ok=True)

    # save model
    torch.save(model, save_path.joinpath(f'{model_name}-{max_acc}.pt'))
    # save accuracy
    with open(save_path.joinpath(f'{model_name}-{max_acc}.json'), 'w+', encoding='utf-8') as f:
        json.dump(accuracy, f, indent=2)


def check_model_type(value: str) -> ResNet:
    if value == 'ResNet18' or value == 'ResNet50' or value == 'ResNet152':
        return value
    else:
        raise ArgumentTypeError(f'Does not support model {value}')


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
    parser.add_argument('-m', '--model', default='ResNet18',
                        type=check_model_type, help='Model')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('-o', '--optimizer', default='sgd',
                        type=check_optimizer_type, help='Optimizer')
    parser.add_argument('-r', '--learning_rate', default=0.1,
                        type=float, help='Learning rate')
    parser.add_argument('-mm', '--momentum', default=0.9,
                        type=float, help='Momentum for SGD')
    parser.add_argument('-wd', '--weight_decay', default=1e-4,
                        type=float, help='Weight dacay for optimizer')
    return parser.parse_args()


def main():
    args = parse_argument()
    model = args.model
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

    print(f'Model:            {model}')
    print(f'Epochs:           {epochs}')
    print(f'Batch size:       {batch_size}')
    print(f'Learning rate:    {learning_rate}')
    print(f'Momentum:         {momentum}')
    print(f'Weight decay:     {weight_decay}')
    print(f'Train device:     {train_device_name}')

    model_name = model
    model = {
        'ResNet18': ResNet18(2).to(train_device),
        'ResNet50': ResNet50(2).to(train_device),
        'ResNet152': ResNet152(2).to(train_device),
    }[model_name]

    train(
        model=model,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        train_device=train_device,
        train_dataset=LeukemiaLoader('train'),
        test_dataset=LeukemiaLoader('valid'),
        save_folder=Path('./storage/latest')
    )


if __name__ == '__main__':
    main()
