import torch
from torch.utils import data
from torchvision import datasets, transforms

import json
import numpy as np
from PIL import Image
from sklearn import preprocessing

class iclevrLoader(data.Dataset):
    def __init__(self, root='./iclevr', mode='train') -> None:
        super().__init__()

        self.root = root
        self.mode = mode
        self.code = self.__get_code()

        self.transforms = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])

        if mode == 'train':
            self.img_path_list, self.label_list = self.__get_train_data()
        elif mode == 'test':
            self.label_list = self.__get_test_data()

    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, index):
        label = self.label_list[index]

        try:
            img = self.img_path_list[index]
            img = self.transforms(Image.open(img).convert('RGB'))
        except:
            img = torch.ones(1)

        return img, label

    def __get_code(self):
        with open(f'{self.root}/objects.json', 'r', encoding='utf-8') as json_file:
            code = json.load(json_file)
        return code
    
    def __get_train_data(self):
        with open(f'{self.root}/train.json', 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        lb = preprocessing.LabelBinarizer()
        lb.fit([i for i in range(24)])

        label_list = []
        img_path_list = []

        for img_path, labels_per_image in data.items():
            tmp = []
            img_path_list.append(f'{self.root}/{img_path}')
            for label in labels_per_image:
                tmp.append(np.array(lb.transform([self.code[label]])))
            label_list.append(np.sum(tmp, axis=0))

        return img_path_list, label_list
    
    def __get_test_data(self):
        with open(f'{self.root}/test.json', 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        lb = preprocessing.LabelBinarizer()
        lb.fit([i for i in range(24)])

        label_list = []
        for labels_per_image in data:
            tmp = []
            for label in labels_per_image:
                tmp.append(np.array(lb.transform([self.code[label]])))
            label_list.append(np.sum(tmp, axis=0))

        return label_list