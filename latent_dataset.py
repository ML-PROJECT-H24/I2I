
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

        try:
            loaded = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f'Error loading {path}: {e}')
            print('Skipping...')
            return self.__getitem__(np.random.randint(0, len(self.local_images))) # Get a random image
        
        mean = loaded['mean'].astype(np.float32)[0]
        logvar = loaded['logvar'].astype(np.float32)[0]

        return mean, logvar
