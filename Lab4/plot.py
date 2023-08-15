import pandas as pd
from matplotlib import pyplot as plt
from argparse import ArgumentParser, Namespace


def show_result(title: str, data_list: list) -> None:
    plt.figure(0)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.title(f'{title} curve')

    plt.plot(
        range(len(data_list)),
        data_list,
    )

    plt.legend(loc='lower right')
    plt.savefig('plot.png')


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', required=True, type=str)
    parser.add_argument('-c', '--column', required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_argument()
    file_path = args.file
    column = args.column

    df = pd.read_csv(file_path)
    data_list = df[column].to_list()[2:]

    show_result(column, data_list)


if __name__ == '__main__':
    main()
