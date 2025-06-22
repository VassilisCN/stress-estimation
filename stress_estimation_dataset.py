import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os


class STRESS_dataset(Dataset):
    """Dataset of emotion and shape coeficients of face images."""

    def __init__(self, root='/home/nikodim/big_data/Stress/outputs', subjects=range(1,60), tasks=range(1, 12), sequence_pairs=[], min_frame=1, max_frame=3602, preload=True, multiclass=False, transform=None):
        """
        Args:
            
        """
        self.root = root
        self.preload = preload
        self.transform = transform
        self.subjects = subjects
        self.tasks = tasks
        self.index_map = sequence_pairs
        self.data = []
        self.multiclass = multiclass

        if self.index_map:
            # If sequence_pairs is provided, use it to create the index_map
            for s, t in self.index_map:
                s = str(s).zfill(3)
                t = str(t).zfill(2)
                path = os.path.join(self.root, f"P{s}", f"tsk{t}_video")
                if not os.path.exists(path):
                    print(f"Skipping video {path} as it does not exist!")
                    continue
                else:
                    if self.preload:
                        pose_shape_exp = np.load(os.path.join(path, "pose_shape_exp.npy"))
                        # pose = pose_shape_exp[:, :6]
                        exp = pose_shape_exp[:, 106:]
                        # pose_exp = np.concatenate([pose, exp], axis=1)
                        self.data.append(exp)
        else:
            for s in self.subjects:
                for t in self.tasks:
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
                            # pose = pose_shape_exp[:, :6]
                            exp = pose_shape_exp[:, 106:]
                            # pose_exp = np.concatenate([pose, exp], axis=1)
                            self.data.append(exp)
                        # dirs = os.listdir(path)
                        # dirs.sort()
                        # max_frame = int(dirs[-2].split('_')[0])
                    
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        s, t = self.index_map[idx]
        if self.multiclass:
            stress = 0 if int(t) in [1, 2, 4, 6, 9] else 0
            if int(t) == 3:
                stress = 1
            elif int(t) == 5:
                stress = 2
            elif int(t) == 7:
                stress = 3
            elif int(t) == 8:
                stress = 4
            elif int(t) == 10:
                stress = 5
            elif int(t) == 11:
                stress = 6
        else:
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