import torch
from torch import nn
from torch import argmax as tenser_max
from torch import Tensor, cuda, device, no_grad
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from argparse import ArgumentParser, Namespace, ArgumentError, ArgumentTypeError

from dataloader import read_bci_data


def eval(model: nn.Module, dataset: TensorDataset, batch_size: int, train_device: device) -> float:
    acc = 0
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    with no_grad():
        for data, label in tqdm(loader):
            inputs = data.to(train_device)
            labels = label.to(train_device).long()

            prediction = model(inputs)

            acc += (tenser_max(prediction, dim=1) == labels).sum().item()

    return acc * 100 / len(loader.dataset)


def check_device_type(value: str) -> str:
    value = ''.join(value.split())

    if 'cuda' in value:
        return value
    else:
        raise ArgumentTypeError('Device must contain cuda.')


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-c', '--use_cpu', default=False,
                        action='store_true', help='Only use cpu')
    parser.add_argument('-d', '--use_device', default='cuda',
                        type=check_device_type, help='Specific gpu')
    parser.add_argument('-b', '--batch_size', default=1080, type=int, help='Batch size')
    parser.add_argument('-m', '--model_path', required=True, help='The model data path')
    return parser.parse_args()


def main():
    args = parse_argument()
    use_cpu = args.use_cpu
    use_device = args.use_device
    batch_size = args.batch_size
    model_path = args.model_path

    if use_cpu:
        train_device = device('cpu')
    else:
        if cuda.is_available():
            train_device = device(use_device)
        else:
            raise ArgumentError('Your device does not support cuda.')

    model = torch.load(model_path, map_location=train_device)

    _, _, test_data, test_label = read_bci_data()
    test_dataset = TensorDataset(Tensor(test_data), Tensor(test_label))

    acc = eval(model, test_dataset, batch_size, train_device)
    print(f'Model accuracy: {acc}')


if __name__ == '__main__':
    main()
