import torch
from torch import nn
from torch import optim
from torch import max as tensor_max
from torch import Tensor, device, cuda, no_grad
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace, ArgumentTypeError

from model import EEGNet, DeepConvNet
from dataloader import read_bci_data


def show_result(
    model_type: str,
    epoch: int,
    accuracy: dict
) -> None:
    longest = len(max(accuracy['train'].keys(), key=len)) + 6

    plt.figure(0)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    if model_type == 'EEG':
        plt.title('Activation function comparison (EEGNet)')
    else:
        plt.title('Activation function comparison (DeepConvNet)')

    for train_or_test in accuracy.keys():
        for model_name in accuracy[train_or_test]:
            plt.plot(
                range(epoch),
                accuracy[train_or_test][model_name],
                label=f'{model_name}_{train_or_test}'
            )
            spaces = ''.join(
                [' ' for _ in range(longest - len(f'{model_name}_{train_or_test}'))])
            print(
                f'{model_name}_{train_or_test}: {spaces}{max(accuracy[train_or_test][model_name]):.2f} %')

    plt.legend(loc='lower right')
    plt.show()


def train(
    model_type: str,
    dropout: float,
    epoch: int,
    train_device: str,
    learning_rate: float,
    optimizer: optim,
    loss_function: nn.modules.loss,
    num_of_linear: int,
    batch_size: int,
    train_dataset: TensorDataset,
    test_dataset: TensorDataset
) -> None:
    if model_type == 'EEG':
        models = {
            'ELU': EEGNet(nn.ELU, dropout).to(train_device),
            'ReLU': EEGNet(nn.ReLU, dropout).to(train_device),
            'LeakyReLU': EEGNet(nn.LeakyReLU, dropout).to(train_device)
        }
    else:
        models = {
            'ELU': DeepConvNet(nn.ELU, dropout, num_of_linear).to(train_device),
            'ReLU': DeepConvNet(nn.ReLU, dropout, num_of_linear).to(train_device),
            'LeakyReLU': DeepConvNet(nn.LeakyReLU, dropout, num_of_linear).to(train_device)
        }

    accuracy = {
        'train': {key: [0 for _ in range(epoch)] for key in models.keys()},
        'test': {key: [0 for _ in range(epoch)] for key in models.keys()}
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, len(test_dataset))

    for model_name, model in models.items():
        cuda.empty_cache()
        model_optim = optimizer(model.parameters(), lr=learning_rate)

        # Train model
        for cur_epoch in tqdm(range(epoch)):
            model.train()
            for data, label in train_loader:
                inputs = data.to(train_device)
                labels = label.to(train_device).long()

                prediction = model.forward(inputs=inputs)

                model_optim.zero_grad()
                loss = loss_function(prediction, labels)
                loss.backward()
                model_optim.step()

                accuracy['train'][model_name][cur_epoch] += (
                    tensor_max(prediction, 1)[1] == labels).sum().item()

            accuracy['train'][model_name][cur_epoch] *= 100.0
            accuracy['train'][model_name][cur_epoch] /= len(train_dataset)

            # Test model
            model.eval()
            with no_grad():
                for data, label in test_loader:
                    inputs = data.to(train_device)
                    labels = label.to(train_device).long()

                    prediction = model.forward(inputs=inputs)

                    accuracy['test'][model_name][cur_epoch] += (
                        tensor_max(prediction, 1)[1] == labels).sum().item()

                accuracy['test'][model_name][cur_epoch] *= 100
                accuracy['test'][model_name][cur_epoch] /= len(test_dataset)

        print()

    show_result(model_type, epoch, accuracy)


def check_model_type(value: str) -> str:
    if value != 'EEG' and value != 'Deep':
        raise ArgumentTypeError('Only support EEGNet or DeepConvNet')
    return value


def check_optimizer_type(value: str) -> optim:
    OPTIM_STR_DICT = {
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
    parser.add_argument('-m', '--model', default='EEG',
                        type=check_model_type, help='EEGNet or DeepConvNet')
    parser.add_argument('-e', '--epochs', default=300, type=int, help='Number of epochs')
    parser.add_argument('-r', '--learning_rate', default=1e-3,
                        type=float, help='Learning rate')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('-o', '--optimizer', default='adam',
                        type=check_optimizer_type, help='Optimizer')
    parser.add_argument('-d', '--dropout', default=0.25,
                        type=float, help='Dropout probability')
    parser.add_argument('-l', '--linear', default=1, type=int,
                        help='Extra linear layers in DeepConvNet (default is 1)')
    return parser.parse_args()


def main():
    args = parse_argument()
    model = args.model
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    optimizer = args.optimizer
    dropout = args.dropout
    num_of_linear = args.linear

    train_data, train_label, test_data, test_label = read_bci_data()
    train_dataset = TensorDataset(Tensor(train_data), Tensor(train_label))
    test_dataset = TensorDataset(Tensor(test_data), Tensor(test_label))

    train_device = device('cuda' if cuda.is_available() else 'cpu')

    print(f'Model:            {model}')
    print(f'Epochs:           {epochs}')
    print(f'Learning rate:    {learning_rate}')
    print(f'Batch size:       {batch_size}')
    print(f'Dropout:          {dropout}')
    print(f'Number of linear: {num_of_linear}')
    print(f'Train device:     {"cuda" if cuda.is_available() else "cpu"}')

    train(
        model_type=model,
        dropout=dropout,
        epoch=epochs,
        train_device=train_device,
        learning_rate=learning_rate,
        optimizer=optimizer,
        loss_function=nn.CrossEntropyLoss(),
        num_of_linear=num_of_linear,
        batch_size=batch_size,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )


if __name__ == '__main__':
    main()
