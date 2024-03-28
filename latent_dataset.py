
import numpy as np
import os
from torch.utils.data import Dataset

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npz"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class LatentImageDataset(Dataset):
    def __init__(self, data_dir, class_conditional=True):
        super().__init__()
        self.all_files = _list_image_files_recursively(data_dir)

        self.classes = None
        if class_conditional:
            class_names = [os.path.basename(x).split('_')[0] for x in self.all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
            self.classes = np.array(classes)

            print(f'Found {len(class_names)} samples with {len(sorted_classes)} classes')
        
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        path = self.all_files[idx]

        try:
            loaded = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f'Error loading {path}: {e}')
            print('Skipping...')
            return self.__getitem__(np.random.randint(0, len(self.all_files))) # Get a random image
        
        mean = loaded['mean'].astype(np.float32)[0]
        logvar = loaded['logvar'].astype(np.float32)[0]

        out_dict = {}
        if self.classes is not None:
            out_dict['y'] = self.classes[idx]

        return mean, logvar, out_dict
