import os, re, glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import re

_ID_PATTERNS = [
    re.compile(r'(?:F?GT)_(\d{3})_\d{4}$'),  # ...GT_034_0001 / ...FGT_034_0001
    re.compile(r'_(\d{3})_\d{4}$'),          # ..._034_0005 (입력)
]
def id_from_name_gt_rule(name: str) -> str:
    stem = Path(name).stem
    for pat in _ID_PATTERNS:
        m = pat.search(stem)
        if m:
            return m.group(1)
    raise ValueError(f"[id_from_name_gt_rule] 이름 규칙 미일치: {name}")

class RestoreFinetuneDataset(Dataset):
    def __init__(self, root, tasks=("rain",), split="train",
                 id_from_name=id_from_name_gt_rule,
                 transform_train=None, transform_eval=None,
                 mode="train"):  # "train" or "eval"
        self.root = Path(root)
        self.tasks = list(tasks) if isinstance(tasks, (list, tuple)) else [tasks]
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

        meta = {
            "task": task,
            "id": iid,
            "in_path": str(in_path),
            "gt_path": str(gt_path),
        }


        if self.mode == "train" and self.t_train:
            x1_pil, x2_pil, y1_pil = self.t_train(x, y)
            x1 = self._to_tensor(x1_pil)
            x2 = self._to_tensor(x2_pil)
            y1 = self._to_tensor(y1_pil)
            meta = {...}
            return (x1, x2), y1, meta
        
        x_t = self._to_tensor(x)
        y_t = self._to_tensor(y)
        if self.t_eval is not None and self.mode != "train":
            # 필요시 t_eval로 후처리(옵션)
            x_t, y_t = self.t_eval(x_t, y_t)
        return x_t, y_t, meta