import json
from pathlib import Path
from matplotlib import pyplot as plt
from argparse import ArgumentParser, Namespace


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


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-f', '--folder', default='./storage/best/',
                        type=str, help='The folder contains three models')
    return parser.parse_args()


def main() -> None:
    args = parse_argument()
    folder_path = Path(args.folder)

    accuracy = {
        'train': {
            'ResNet18': [],
            'ResNet50': [],
            'ResNet152': [],
        },
        'test': {
            'ResNet18': [],
            'ResNet50': [],
            'ResNet152': [],
        },
    }

    # autopep8: off
    with open(folder_path.joinpath('ResNet18.json'), 'r', encoding='utf-8') as f:
        accuracy['train']['ResNet18'], accuracy['test']['ResNet18'] = json.load(f).values()

    with open(folder_path.joinpath('ResNet50.json'), 'r', encoding='utf-8') as f:
        accuracy['train']['ResNet50'], accuracy['test']['ResNet50'] = json.load(f).values()

    with open(folder_path.joinpath('ResNet152.json'), 'r', encoding='utf-8') as f:
        accuracy['train']['ResNet152'], accuracy['test']['ResNet152'] = json.load(f).values()
    # autopep8: on

    max_epochs = len(max(accuracy['train'].values(), key=len))
    for test_or_train in accuracy.keys():
        for model_name in accuracy[test_or_train].keys():
            acc_list = accuracy[test_or_train][model_name]
            acc_list += [0 for _ in range(max_epochs - len(acc_list))]

    show_result(max_epochs, accuracy)


if __name__ == '__main__':
    main()
