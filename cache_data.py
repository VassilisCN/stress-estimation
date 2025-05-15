
import os
import numpy as np


root='/home/nikodim/big_data/Stress/outputs'
tota_tasks = 11
subjects = [i for i in range(1, 60) if i not in [1, 33, 49, 55]]
min_frame = 1
max_frame = 3286

for s in subjects:
    for t in range(1, tota_tasks + 1):
        # String with 3 zeros
        s = str(s).zfill(3)
        t = str(t).zfill(2)
        path = os.path.join(root, f"P{s}", f"tsk{t}_video")
        if not os.path.exists(path):
            print(f"Skipping video {path} as it does not exist!")
            continue
        else:
            # dirs = os.listdir(path)
            # dirs.sort()
            # max_frame = int(dirs[-2].split('_')[0])
            print(s, t)
            data = None
            for i in range(min_frame, max_frame + 1):
                frame = str(i).zfill(6)
                p = os.path.join(path, f"{frame}_000")
                if not os.path.exists(p):
                    print(f"No frame detected in {p}.\n Some batches may have incosistent number of frames.")
                else:
                    pose = np.load(os.path.join(p, "pose.npy"))
                    shape = np.load(os.path.join(p, "shape.npy"))
                    exp = np.load(os.path.join(p, "exp.npy"))
                    if data is None:
                        data = np.expand_dims(np.concatenate((pose, shape, exp), axis=0), axis=0)
                    else:
                        tmp_d = np.expand_dims(np.concatenate((pose, shape, exp), axis=0), axis=0)
                        data = np.concatenate((data, tmp_d), axis=0)
            np.save(os.path.join(path, f"pose_shape_exp.npy"), data)