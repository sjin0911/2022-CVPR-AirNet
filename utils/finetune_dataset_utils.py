import os, re, glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def id_from_name_gt_rule(name: str):
    """
    파일 이름에서 GT 매칭용 ID 추출.
    - GT: D-210725_O9125FGT_002_0001.jpg
    - 여기서 'FGT_' 뒤의 세자리(002)를 반환
    """
    stem = Path(name).stem
    if "FGT_" in stem:
        # GT 파일
        return stem.split("FGT_")[1][:3]   # '002'
    else:
        # Input 파일도 같은 자리에서 id를 뽑을 수 있다고 가정
        parts = stem.split('_')
        if len(parts) >= 2:
            return parts[-2]  # 예: ..._002_0005 → '002'
        else:
            raise ValueError(f"이름 규칙 확인 필요: {name}")

class RestoreFinetuneDataset(Dataset):
    def __init__(self, root, tasks=("rain",), split="train",
                 id_from_name=id_from_name_gt_rule,
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
            x1_pil, x2_pil, y1_pil = self.t_train(x, y)
            x1 = self._to_tensor(x1_pil)
            x2 = self._to_tensor(x2_pil)
            y1 = self._to_tensor(y1_pil)
            meta = {...}
            return (x1, x2), y1, meta
        elif self.mode != "train" and self.t_eval:
            # eval: full image
            x = self._to_tensor(x)
            y = self._to_tensor(y)
            return x, y, meta