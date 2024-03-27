
import numpy as np
import os
from torch.utils.data import Dataset

class LatentImageDataset(Dataset):
    def __init__(self, data_dir, class_conditional=True):
        super().__init__()
        self.names = os.listdir(data_dir)
        self.local_images = [os.path.join(data_dir, x) for x in self.names]

        self.classes = None
        if class_conditional:
            class_names = [x.split('_')[0] for x in self.names]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
            self.classes = np.array(classes)
        
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

        out_dict = {}
        if self.classes is not None:
            out_dict['y'] = self.classes[idx]

        return mean, logvar, out_dict
