import torch
from torch import nn
from torch import argmax as tenser_max
from torch import device, no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from argparse import ArgumentParser, Namespace, ArgumentError, ArgumentTypeError

from dataloader import LeukemiaLoader
from model import ResNet18, ResNet50, ResNet152


def eval(
    model: nn.Module,
    valid_dataset: LeukemiaLoader,
    test_dataset: LeukemiaLoader,
    batch_size: int,
    train_device: device
) -> DataFrame:
    test_predict_list = []
    valid_predict_list = []
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    model.eval()
    with no_grad():
        print('Get valid dataset prediction')
        for data, _ in tqdm(valid_loader):
            inputs = data.to(train_device)
            prediction = model(inputs)
            valid_predict_list += tenser_max(prediction, dim=1).tolist()

        print('Get test dataset prediction')
        for data, _ in tqdm(test_loader):
            inputs = data.to(train_device)
            prediction = model(inputs)
            test_predict_list += tenser_max(prediction, dim=1).tolist()

    return valid_predict_list, test_predict_list


def check_device_type(value: str) -> str:
    value = ''.join(value.split())

    if 'cuda' in value or 'mps' in value:
        return value
    else:
        raise ArgumentTypeError('Device must contain cuda or mps.')


def check_model_type(value: str) -> str:
    if value == 'test18' or value == 'test50' or value == 'test152':
        return value
    else:
        raise ArgumentTypeError(f'Does not support architecture {value}')


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-c', '--use_cpu', default=False,
                        action='store_true', help='Only use cpu')
    parser.add_argument('-d', '--use_device', default='cuda',
                        type=check_device_type, help='Specific gpu')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('-m', '--model_path', required=True, help='The model data path')
    parser.add_argument('-a', '--architecture', default='test18',
                        type=check_model_type, help='The architecture for test')
    return parser.parse_args()


def main():
    args = parse_argument()
    use_cpu = args.use_cpu
    use_device = args.use_device
    batch_size = args.batch_size
    model_path = args.model_path
    architecture = args.architecture

    if use_cpu:
        train_device = device('cpu')
    else:
        try:
            train_device = device(use_device)
        except:
            raise ArgumentError(f'Your device does not support {use_device}.')

    if architecture == 'test18':
        model = ResNet18(2).to(train_device)
    elif architecture == 'test50':
        model = ResNet50(2).to(train_device)
    else:
        model = ResNet152(2).to(train_device)

    model.load_state_dict(torch.load(model_path))

    valid_dataset = LeukemiaLoader('valid')
    test_dataset = LeukemiaLoader(architecture)
    valid_predict_list, test_predict_list = eval(
        model, valid_dataset, test_dataset, batch_size, train_device)

    result = DataFrame({
        'ID': test_dataset.img_path_list,
        'label': test_predict_list
    })

    ground_truth = valid_dataset.label_list
    cm = confusion_matrix(y_true=ground_truth,
                          y_pred=valid_predict_list, normalize='true')
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                           False, True]).plot(cmap=plt.cm.Blues)
    plt.title(f'Normalized confusion matrix')

    if architecture == 'test18':
        plt.savefig('./ResNet18_confusion_matrix.png')
        result.to_csv('./312553004_resnet18.csv', index=False)
    elif architecture == 'test50':
        plt.savefig('./ResNet50_confusion_matrix.png')
        result.to_csv('./312553004_resnet50.csv', index=False)
    else:
        plt.savefig('./ResNet152_confusion_matrix.png')
        result.to_csv('./312553004_resnet152.csv', index=False)


if __name__ == '__main__':
    main()
