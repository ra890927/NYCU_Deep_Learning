import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils import data
from torchvision import transforms
from typing import Tuple, List


class LeukemiaLoader(data.Dataset):
    def __init__(
        self,
        mode: str,
        root_path: str = './new_dataset'
    ) -> None:
        """
        :param mode: which mode dataset (train, valid, test)
        :param root_path: root path for dataset
        """

        super(LeukemiaLoader, self).__init__()

        self.mode = mode
        self.root_path = root_path
        self.img_path_list, self.label_list = self.__get_data()

        self.transform = transforms.RandomOrder([
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

        self.to_tensor = transforms.Compose([
            transforms.CenterCrop(350),
            transforms.Resize([350, 350]),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.img_path_list)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        label = self.label_list[index]
        img = Image.open(self.img_path_list[index])
        if self.mode == 'train':
            img = self.transform(img)
        img_data = self.to_tensor(img)
        return img_data, label

    def __get_data(self) -> Tuple[List[str]]:
        if self.mode == 'train':
            df = pd.read_csv(f'{self.root_path}/train.csv')
        elif self.mode == 'valid':
            df = pd.read_csv(f'{self.root_path}/valid.csv')
        elif self.mode == 'test18':
            df = pd.read_csv(f'{self.root_path}/resnet_18_test.csv')
        elif self.mode == 'test50':
            df = pd.read_csv(f'{self.root_path}/resnet_50_test.csv')
        elif self.mode == 'test152':
            df = pd.read_csv(f'{self.root_path}/resnet_152_test.csv')
        else:
            raise ValueError(f'Does not support {self.mode}')

        img_path_list = df['Path'].to_list()
        try:
            label_list = df['label'].to_list()
        except:
            label_list = [0 for _ in range(len(img_path_list))]

        return img_path_list, label_list
