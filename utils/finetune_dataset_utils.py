import os, re, glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def default_id_from_name(name: str):
    # 예: input: IMG123_01.png → "IMG123", gt: IMG123.png → "IMG123"
    stem = Path(name).stem
    return stem.split('_')[0]

class RestoreFinetuneDataset(Dataset):
    def __init__(self, root, tasks=("rain",), split="train",
                 id_from_name=default_id_from_name,
                 transform_train=None, transform_eval=None,
                 mode="train"):  # "train" or "eval"
        self.root = Path(root)
        self.tasks = tasks if isinstance(tasks, (list, tuple)) else [tasks]
        self.split = split
        self.id_from_name = id_from_name
        self.mode = mode
        self.t_train = transform_train
        self.t_eval  = transform_eval

        self.samples = []  # (in_path, gt_path, task, id)
        for task in self.tasks:
            in_dir = self.root/task/split/"input"
            gt_dir = self.root/task/split/"gt"
            gt_map = {}  # id -> gt_path
            for gp in sorted(glob.glob(str(gt_dir/"*.*"))):
                gid = self.id_from_name(os.path.basename(gp))
                gt_map[gid] = gp
            # 각 input을 id로 묶어 GT 매칭
            for ip in sorted(glob.glob(str(in_dir/"*.*"))):
                iid = self.id_from_name(os.path.basename(ip))
                gp = gt_map.get(iid, None)
                if gp is not None:
                    self.samples.append((ip, gp, task, iid))
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found under {self.root} with tasks={self.tasks}, split={self.split}")

    def __len__(self): return len(self.samples)

    def _to_tensor(self, img):
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:  # gray → 3채널로
            arr = np.stack([arr]*3, axis=-1)
        return torch.from_numpy(arr).permute(2,0,1)

    def __getitem__(self, idx):
        in_path, gt_path, task, iid = self.samples[idx]
        x = Image.open(in_path).convert("RGB")
        y = Image.open(gt_path).convert("RGB")

        if self.mode == "train" and self.t_train:
            x, y = self.t_train(x, y)
        elif self.mode != "train" and self.t_eval:
            x, y = self.t_eval(x, y)

        x = self._to_tensor(x)
        y = self._to_tensor(y)
        meta = {"task": task, "id": iid, "in_path": in_path, "gt_path": gt_path}
        return x, y, meta
