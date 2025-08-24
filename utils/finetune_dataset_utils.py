import os, re, glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# 파일명 끝 "...(F?GT_)?NNN_####"에서 NNN 추출
_ID_PAT = re.compile(r'(?:F?GT_)?(?P<mid>\d{3})_(?P<seq>\d{4})$', re.IGNORECASE)

def id_from_name_gt_rule(name: str) -> str:
    stem = Path(name).stem
    m = _ID_PAT.search(stem)
    if not m:
        raise ValueError(f"[id_from_name_gt_rule] 이름 규칙 불일치: {name}")
    return m.group("mid")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def load_list(txt_path):
    with open(txt_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

class RestoreFinetuneDataset(Dataset):
    def __init__(self, root, tasks=("rain",), split="train",
                 id_from_name=id_from_name_gt_rule,
                 transform_train=None, transform_eval=None,
                 mode="train"):
        self.root = Path(root)
        self.tasks = list(tasks) if isinstance(tasks, (list, tuple)) else [tasks]
        self.split = split
        self.id_from_name = id_from_name
        self.mode = mode
        self.t_train = transform_train
        self.t_eval  = transform_eval

        self.samples = []  # (in_path, gt_path, task, id)

        for task in self.tasks:
            # txt 파일 경로 추정
            gt_txt = self.root / f"GT_{task}_{split}.txt"
            if not gt_txt.exists():
                raise FileNotFoundError(f"{gt_txt} not found!")

            gt_files = load_list(gt_txt)
            # txt에 기록된 파일명 → 실제 경로 구성
            in_dir = self.root / task / split / "input"
            gt_dir = self.root / task / split / "gt"

            for fname in gt_files:
                gp = gt_dir / fname
                ip = in_dir / fname.replace("GT_", "")  # 규칙: GT 없는 파일명이 input
                if gp.exists() and ip.exists():
                    iid = self.id_from_name(fname)
                    self.samples.append((str(ip), str(gp), task, iid))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples from txt under {self.root} with tasks={self.tasks}, split={self.split}")

    def __len__(self):
        return len(self.samples)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:  # gray -> 3채널
            arr = np.stack([arr]*3, axis=-1)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

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
            # t_train(x_pil, y_pil) -> (x1_pil, x2_pil, y1_pil) 가정
            x1_pil, x2_pil, y1_pil = self.t_train(x, y)
            x1 = self._to_tensor(x1_pil)
            x2 = self._to_tensor(x2_pil)
            y1 = self._to_tensor(y1_pil)
            return (x1, x2), y1, meta

        x_t = self._to_tensor(x)
        y_t = self._to_tensor(y)
        if self.t_eval is not None and self.mode != "train":
            x_t, y_t = self.t_eval(x_t, y_t)
        return x_t, y_t, meta
