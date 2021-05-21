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

# Data has frame skip of 5.
GAP = 1
STEPS = 4
N_CLASSES = len(COLOR)


class SequentialCarlaDataset(Dataset):
    def __init__(self, dataset_dir, sequence_length=1, skip_beginning=10):
        self.sequence_length = sequence_length
        self.skip_beginning = skip_beginning

        dataset_dir = Path(dataset_dir)
        measurements = list(sorted((dataset_dir / 'measurements').glob('*.json')))

        self.dataset_dir = dataset_dir
        self.frames = list()
        pd_measurements = pd.DataFrame([eval(x.read_text()) for x in measurements])

        self.labels = np.stack([pd_measurements['steer'].values.astype(np.float32),
                                pd_measurements['target_speed'].values.astype(np.float32)],
                               axis=-1)
        self.labels[np.isnan(self.labels)] = 0.0

        self.route_commands = pd_measurements['command'].values

        for image_path in sorted((dataset_dir / 'rgb').glob('*.png')):
            frame = str(image_path.stem)

            assert (dataset_dir / 'rgb_left' / ('%s.png' % frame)).exists()
            assert (dataset_dir / 'rgb_right' / ('%s.png' % frame)).exists()
            assert (dataset_dir / 'topdown' / ('%s.png' % frame)).exists()
            assert int(frame) < len(self.labels)

            self.frames.append(frame)

        self.frames = np.asarray(self.frames)

        self.frames = self.frames[skip_beginning:]
        self.labels = self.labels[skip_beginning:]
        self.route_commands = self.route_commands[skip_beginning:]

        assert len(self.frames) > 0, '%s has 0 frames.' % dataset_dir

    def __len__(self):
        return len(self.frames) - GAP * STEPS - (self.sequence_length - 1)

    def __getitem__(self, index):
        path = self.dataset_dir

        data = {'image': [],
                'bev': [],
                'action': [],
                'route_command': [],
                }

        for i in range(index, index + self.sequence_length):
            frame = self.frames[i]

            rgb = Image.open(path / 'rgb' / ('%s.png' % frame))
            rgb = transforms.functional.to_tensor(rgb)

            rgb_left = Image.open(path / 'rgb_left' / ('%s.png' % frame))
            rgb_left = transforms.functional.to_tensor(rgb_left)

            rgb_right = Image.open(path / 'rgb_right' / ('%s.png' % frame))
            rgb_right = transforms.functional.to_tensor(rgb_right)

            image = torch.stack([rgb_left, rgb, rgb_right])

            topdown = Image.open(path / 'topdown' / ('%s.png' % frame))
            topdown = preprocess_bev_state(topdown)

            actions = torch.FloatTensor(self.labels[i])

            data['image'].append(image)
            data['bev'].append(topdown)
            data['action'].append(actions)
            data['route_command'].append(torch.LongTensor([self.route_commands[i]]))

        for key, value in data.items():
            data[key] = torch.stack(value)

        return data


def get_dataset_sequential(dataset_dir, is_train=True, batch_size=64, num_workers=4, **kwargs):
    data = list()

    data_path = Path(dataset_dir) / 'train' if is_train else Path(dataset_dir) / 'val'

    episodes = list(sorted(data_path.glob('*')))

    for i, _dataset_dir in enumerate(episodes):
        data.append(SequentialCarlaDataset(_dataset_dir, **kwargs))
    data = torch.utils.data.ConcatDataset(data)
    print(f'{len(data)} elements in {data_path}')

    shuffle = True if is_train else False

    return torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True
    )
