from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from numpy import nan

from carla_project.src.common import COLOR
from world_model.utils import preprocess_bev_state

N_CLASSES = len(COLOR)


class SequentialCarlaDataset(Dataset):
    def __init__(self, dataset_dir, sequence_length=1, skip_beginning=10):
        self.sequence_length = sequence_length
        self.skip_beginning = skip_beginning

        self.normalise_image = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ]
        )

        dataset_dir = Path(dataset_dir)
        measurements = list(sorted((dataset_dir / 'measurements').glob('*.json')))

        self.dataset_dir = dataset_dir
        self.frames = list()
        pd_measurements = pd.DataFrame([eval(x.read_text()) for x in measurements])

        self.labels = {}
        for key in ['steer', 'target_speed', 'speed', 'throttle', 'brake']:
            self.labels[key] = pd_measurements[key].values.astype(np.float32)
            self.labels[key][np.isnan(self.labels[key])] = 0.0

        self.labels['route_command'] = pd_measurements['command'].values

        for image_path in sorted((dataset_dir / 'rgb').glob('*.png')):
            frame = str(image_path.stem)

            assert (dataset_dir / 'rgb_left' / ('%s.png' % frame)).exists()
            assert (dataset_dir / 'rgb_right' / ('%s.png' % frame)).exists()
            assert (dataset_dir / 'topdown' / ('%s.png' % frame)).exists()

            self.frames.append(frame)

        self.frames = np.asarray(self.frames)

        self.frames = self.frames[skip_beginning:]
        for key, value in self.labels.items():
            self.labels[key] = value[skip_beginning:]

        assert len(self.frames) > 0, '%s has 0 frames.' % dataset_dir

    def __len__(self):
        return len(self.frames) - (self.sequence_length - 1)

    def __getitem__(self, index):
        path = self.dataset_dir

        data = {'image': [],
                'bev': [],
                'speed': [],
                'route_command': [],
                'action': [],
                'brake': [],
                }

        for i in range(index, index + self.sequence_length):
            frame = self.frames[i]

            rgb = Image.open(path / 'rgb' / ('%s.png' % frame))
            rgb = self.normalise_image(rgb)

            # TODO remove this cropping!!
            rgb = rgb[:, 16:]

            # rgb_left = Image.open(path / 'rgb_left' / ('%s.png' % frame))
            # rgb_left = transforms.functional.to_tensor(rgb_left)
            #
            # rgb_right = Image.open(path / 'rgb_right' / ('%s.png' % frame))
            # rgb_right = transforms.functional.to_tensor(rgb_right)

            image = torch.stack([rgb])#([rgb_left, rgb, rgb_right])

            topdown = Image.open(path / 'topdown' / ('%s.png' % frame))
            topdown = preprocess_bev_state(topdown)

            actions = torch.FloatTensor(np.stack([self.labels['steer'][i], self.labels['throttle'][i]], axis=-1))

            data['image'].append(image)
            data['bev'].append(topdown)
            data['speed'].append(torch.FloatTensor([self.labels['speed'][i]]))
            data['route_command'].append(torch.LongTensor([self.labels['route_command'][i]]))
            data['action'].append(actions)
            data['brake'].append(torch.LongTensor([self.labels['brake'][i]]))

        for key, value in data.items():
            data[key] = torch.stack(value)

        return data


def get_dataset_sequential(dataset_dir, is_train=True, batch_size=64, num_workers=4, debug_overfit=False, **kwargs):
    data = list()

    data_path = Path(dataset_dir) / 'train' if is_train else Path(dataset_dir) / 'val'

    episodes = list(sorted(data_path.glob('*')))

    for i, _dataset_dir in enumerate(episodes):
        data.append(SequentialCarlaDataset(_dataset_dir, **kwargs))
    data = torch.utils.data.ConcatDataset(data)
    if debug_overfit:
        print('DEBUG OVERFITTING!')
        data = torch.utils.data.Subset(data, torch.arange(batch_size))
    print(f'{len(data)} elements in {data_path}')

    shuffle = True if is_train else False

    return torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True
    )
