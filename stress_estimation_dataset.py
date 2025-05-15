import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os


class STRESS_dataset(Dataset):
    """Dataset of emotion and shape coeficients of face images."""

    def __init__(self, root='/home/nikodim/big_data/Stress/outputs', subjects=range(1,60), min_frame=1, max_frame=3602, preload=True, transform=None):
        """
        Args:
            
        """
        self.root = root
        self.preload = preload
        self.tota_subjects = 59
        self.tota_tasks = 11
        self.transform = transform
        self.subjects = subjects
        self.index_map = []
        self.data = []

        for s in self.subjects:
            for t in range(1, self.tota_tasks + 1):
                # String with 3 zeros
                s = str(s).zfill(3)
                t = str(t).zfill(2)
                path = os.path.join(self.root, f"P{s}", f"tsk{t}_video")
                if not os.path.exists(path):
                    print(f"Skipping video {path} as it does not exist!")
                    continue
                else:
                    self.index_map.append((s, t))
                    if self.preload:
                        pose_shape_exp = np.load(os.path.join(path, "pose_shape_exp.npy"))
                        self.data.append(pose_shape_exp)
                    # dirs = os.listdir(path)
                    # dirs.sort()
                    # max_frame = int(dirs[-2].split('_')[0])
                    
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        s, t = self.index_map[idx]
        stress = 1 if int(t) in [3, 5, 7, 8, 10, 11] else 0
        if self.preload:
            data = self.data[idx]
        else:
            data = np.load(os.path.join(self.root, f"P{s}", f"tsk{t}_video", "pose_shape_exp.npy"))
        if self.transform:
            data = self.transform(data)
        return data, stress
        




if __name__ == "__main__":
    subjects = [i for i in range(1, 60) if i not in [1, 33, 49, 55]]
    s = STRESS_dataset(subjects=subjects, max_frame=3286)
    print(f"Number of subjects: {len(s.subjects)}")
    print(f"Number of tasks: {s.tota_tasks}")
    print(f"Number of samples: {len(s)}")
    print(f"Number of frames per subject: {s[0][0].shape[0]}")
    print(f"Number of samples per task: {len(s) // s.tota_tasks}")
    dt = DataLoader(s, batch_size=16, shuffle=True)
    dataiter = iter(dt)
    features, labels = next(dataiter)
    print(features.shape, labels.shape)