
import numpy as np
import os
from torch.utils.data import Dataset

class LatentImageDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.local_images = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        loaded = np.load(path)

        mean = loaded['mean'].astype(np.float32)
        logvar = loaded['logvar'].astype(np.float32)

        return mean, logvar
