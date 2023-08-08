import torch
from torch import nn
from torch import optim
from torch import argmax as tensor_max
from torch.backends import mps as mpsback
from torch import device, cuda, mps, no_grad
from torch.utils.data import DataLoader

import json
import matplotlib.pyplot as plt
from math import ceil
from tqdm import tqdm
from pathlib import Path
from pandas import DataFrame
from datetime import datetime
from argparse import ArgumentParser, ArgumentTypeError, Namespace

from eval import eval
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

    cur_best_acc = 0
    best_acc_weights = None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if model_name == 'ResNet18':
        predict_dataset = LeukemiaLoader('test18')
    elif model_name == 'ResNet50':
        predict_dataset = LeukemiaLoader('test50')
    else:
        predict_dataset = LeukemiaLoader('test152')

    if train_device.type == 'mps':
        mps.empty_cache()
    elif train_device.type == 'cuda':
        cuda.empty_cache()

    try:
        model_optim = optimizer(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    except:
        model_optim = optimizer(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    start_time = datetime.now()

    for epoch in range(epochs):
        print(f'\nEpoch {epoch}:')

        model.train()
        train_acc, test_acc, train_loss = 0.0, 0.0, 0.0

        for data, label in tqdm(train_loader):
            inputs = data.to(train_device)
            labels = label.to(train_device).long()

            prediction = model(inputs)
            loss = nn.CrossEntropyLoss()(prediction, labels)
            train_loss += loss.item()

            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

            train_acc += (tensor_max(prediction, dim=1) == labels).sum().item()

        train_loss /= ceil(len(train_dataset) / batch_size)
        accuracy['train'][epoch] = train_acc * 100 / len(train_loader.dataset)

        model.eval()
        with no_grad():
            for data, label in tqdm(test_loader):
                inputs = data.to(train_device)
                labels = label.to(train_device).long()

                prediction = model(inputs)

                test_acc += (tensor_max(prediction, dim=1) == labels).sum().item()

            accuracy['test'][epoch] = test_acc * 100 / len(test_loader.dataset)

        print(f'Loss:  {train_loss}')
        print(f'Train: {accuracy["train"][epoch]}')
        print(f'Test:  {accuracy["test"][epoch]}')

        if accuracy['test'][epoch] >= cur_best_acc:
            cur_best_acc = accuracy['test'][epoch]

            # weights
            best_acc_weights = model.state_dict()

            # save result
            _, test_predict_list = eval(
                model, test_dataset, predict_dataset, batch_size, train_device)

            result = DataFrame({
                'ID': predict_dataset.img_path_list,
                'label': test_predict_list
            })

            if model_name == 'ResNet18':
                result.to_csv('./312553004_resnet18.csv', index=False)
            elif model_name == 'ResNet50':
                result.to_csv('./312553004_resnet50.csv', index=False)
            else:
                result.to_csv('./312553004_resnet152.csv', index=False)

    end_time = datetime.now()

    print(f'Time:  {end_time - start_time}')
    print(f'Train: {max(accuracy["train"])}')
    print(f'Test:  {max(accuracy["test"])}')

    show_result(model_name, accuracy, save_folder, epochs)
    save_model(save_folder, model_name, best_acc_weights, accuracy)


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


def save_model(save_path: Path, model_name: str, model_weights: dict, accuracy: dict):
    max_acc = round(max(accuracy['test']))
    save_path.mkdir(parents=True, exist_ok=True)

    # save model
    torch.save(model_weights, save_path.joinpath(f'{model_name}-{max_acc}.pt'))
    # save accuracy
    with open(save_path.joinpath(f'{model_name}-{max_acc}.json'), 'w+', encoding='utf-8') as f:
        json.dump(accuracy, f, indent=2)


def check_model_type(value: str) -> ResNet:
    global MODEL_STR_DICT

    MODEL_STR_DICT = {
        'ResNet18': ResNet18(2),
        'ResNet50': ResNet50(2),
        'ResNet152': ResNet152(2),
    }

    if value in MODEL_STR_DICT.keys():
        return value
    else:
        raise ArgumentTypeError(f'Does not support model {value}')


def check_optimizer_type(value: str) -> optim:
    global OPTIM_STR_DICT

    OPTIM_STR_DICT = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adamw': optim.Adam,
        'adamax': optim.Adamax
    }

    if value in OPTIM_STR_DICT.keys():
        return value
    else:
        raise ArgumentTypeError(f'Does not suppot optimizer {value}.')


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', default='ResNet18',
                        type=check_model_type, help='Model')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('-o', '--optimizer', default='adam',
                        type=check_optimizer_type, help='Optimizer')
    parser.add_argument('-r', '--learning_rate', default=0.001,
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
    print(f'Optimizer:        {optimizer}')
    print(f'Learning rate:    {learning_rate}')
    print(f'Momentum:         {momentum}')
    print(f'Weight decay:     {weight_decay}')
    print(f'Train device:     {train_device_name}')

    model_name = model
    model = MODEL_STR_DICT[model_name].to(train_device)
    optimizer = OPTIM_STR_DICT[optimizer]

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
