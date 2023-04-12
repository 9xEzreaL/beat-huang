import os
import glob
import pandas as pd
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
# from utils.configuration import dataset1, dataset2, dataset3, blood_type, save_logs, split, total_num

label_mapping = {
    'blast': 0,
    'promyelo': 1,
    'myelo': 2,
    'meta': 3,
    'band': 4,
    'seg': 5
}

class BeatDataset(Dataset):
    def __init__(self, dataset1, dataset2, dataset3, blood_type, split, mode):

        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3

        self.split = split
        self.blood_type = blood_type

        self.data_list = list()

        self.mode = mode
        self.ensemble_all_data()


    def ensemble_all_data(self):
        for cell_type in self.blood_type:
            ds_dict = self.split[self.mode][cell_type]

            for ds_key, ds_num in ds_dict.items():
                if ds_num is not None:
                    self.data_list += sorted(glob.glob(os.path.join(getattr(self, ds_key), cell_type, '*')))[ds_num[0]: ds_num[1]]

    def transforms(self, mode):
        if mode == 'train':
            transform = A.Compose([
                A.Resize(width=420, height=420),
                A.RandomCrop(width=384, height=384),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(),
                A.Rotate(limit=45),
                A.CoarseDropout(max_holes=10)
                # A.RandomBrightnessContrast(p=0.2),
                ])
        else:
            transform = A.Compose([
                A.Resize(width=420, height=420),
                A.CenterCrop(width=384, height=384),
                ])
        return transform

    def __getitem__(self, index):
        label = label_mapping[self.data_list[index].split('/')[6]]
        id = self.data_list[index].split('/')[6]

        img = cv2.imread(self.data_list[index])
        transform = self.transforms(self.mode)
        transformed = transform(image=img)
        img = np.transpose(transformed["image"], (2,0,1))
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5

        return img, label, id

    def __len__(self):
        return len(self.data_list)

    def get_labels(self):
        return [label_mapping[x.split('/')[6]] for x in self.data_list]


if __name__ == "__main__":
    dataset1 = '/media/ExtHDD01/Dataset/WBC/WBC_dataset2'
    dataset2 = '/media/ExtHDD01/Dataset/WBC/WBC_dataset3'
    dataset3 = '/media/ExtHDD01/Dataset/WBC/WBC_dataset4'

    blood_type = ['blast', 'promyelo', 'myelo', 'meta', 'band', 'seg']

    total_num = {
        'dataset1': {'blast': 0, 'promyelo': 565, 'myelo': 1110, 'meta': 962, 'band': 1891, 'seg': 1510},
        'dataset2': {'blast': 2962, 'promyelo': 512, 'myelo': 992, 'meta': 658, 'band': 2095, 'seg': 2215},
        'dataset3': {'blast': 5279, 'promyelo': 352, 'myelo': 339, 'meta': 215, 'band': 815, 'seg': 2313}
    }

    split = {
        'train': {'blast': {'dataset1': None, 'dataset2': [0, 2632], 'dataset3': [0, 4693]},
                  'promyelo': {'dataset1': [0, 427], 'dataset2': [0, 427], 'dataset3': [0, 284]},
                  'myelo': {'dataset1': [0, 955], 'dataset2': [0, 854], 'dataset3': [0, 283]},
                  'meta': {'dataset1': [0, 815], 'dataset2': [0, 557], 'dataset3': [0, 180]},
                  'band': {'dataset1': [0, 1670], 'dataset2': [0, 1837], 'dataset3': [0, 709]},
                  'seg': {'dataset1': [0, 1335], 'dataset2': [0, 1958], 'dataset3': [0, 2027]}},

        'eval': {'blast': {'dataset1': None, 'dataset2': [2632, 2632 + 34], 'dataset3': [4693, 4693 + 66]},
                 'promyelo': {'dataset1': [427, 427 + 37], 'dataset2': [427, 427 + 34], 'dataset3': [284, 284 + 29]},
                 'myelo': {'dataset1': [955, 955 + 44], 'dataset2': [854, 854 + 39], 'dataset3': [283, 283 + 17]},
                 'meta': {'dataset1': [815, 815 + 51], 'dataset2': [557, 557 + 35], 'dataset3': [180, 180 + 14]},
                 'band': {'dataset1': [1670, 1670 + 32], 'dataset2': [1837, 1837 + 48], 'dataset3': [709, 709 + 20]},
                 'seg': {'dataset1': [1335, 1335 + 24], 'dataset2': [1958, 1958 + 35], 'dataset3': [2027, 2027 + 41]}},

        'test': {'blast': {'dataset1': None, 'dataset2': [2632 + 34, 2632 + 34 + 296],
                           'dataset3': [4693 + 66, 4693 + 66 + 520]},
                 'promyelo': {'dataset1': [427 + 37, 427 + 37 + 57], 'dataset2': [427 + 34, 427 + 34 + 51],
                              'dataset3': [284 + 29, 284 + 29 + 39]},
                 'myelo': {'dataset1': [955 + 44, 955 + 44 + 111], 'dataset2': [854 + 39, 854 + 39 + 99],
                           'dataset3': [283 + 17, 283 + 17 + 39]},
                 'meta': {'dataset1': [815 + 51, 815 + 51 + 96], 'dataset2': [557 + 35, 557 + 35 + 66],
                          'dataset3': [180 + 14, 180 + 14 + 21]},
                 'band': {'dataset1': [1670 + 32, 1670 + 32 + 189], 'dataset2': [1837 + 48, 1837 + 48 + 210],
                          'dataset3': [709 + 20, 709 + 20 + 86]},
                 'seg': {'dataset1': [1335 + 24, 1335 + 24 + 151], 'dataset2': [1958 + 35, 1958 + 35 + 222],
                         'dataset3': [2027 + 41, 2027 + 41 + 245]}}
    }

    save_logs = '/media/ExtHDD01/wbc_logs'
    data = BeatDataset(dataset1, dataset2, dataset3, blood_type, split, mode='test')
    loader = DataLoader(dataset=data, batch_size=8, shuffle=True, num_workers=2, drop_last=True)

    for img, label, id in loader:
        print(img.shape)
        print(label)
        print(id)
        assert 0
    # pass

