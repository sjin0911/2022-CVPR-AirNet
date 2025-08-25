import os, subprocess, random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from PIL import Image
import wandb 

from utils.finetune_dataset_utils import RestoreFinetuneDataset
from net.model import AirNet
from option_finetune import options as opt

import os, glob, lmdb, pickle
import numpy as np
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset, DataLoader

#####################################
# 1. LMDB 생성 (처음 한 번만 실행)
#####################################
def create_lmdb(input_dir, gt_dir, lmdb_path, write_frequency=5000):
    if os.path.exists(lmdb_path):
        print(f"[LMDB] 이미 {lmdb_path} 있음, 생성 스킵")
        return

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    gt_files    = sorted(glob.glob(os.path.join(gt_dir, "*.jpg")))
    assert len(input_files) == len(gt_files), "input/gt 개수가 다름"

    map_size = 1 << 40  # 1TB까지 허용 (100GB 데이터 안전하게 커버)
    env = lmdb.open(lmdb_path, map_size=map_size)

    txn = env.begin(write=True)
    keys = []

    for idx, (inp, gt) in enumerate(zip(input_files, gt_files)):
        inp_img = read_image(inp).numpy()   # (C,H,W) numpy
        gt_img  = read_image(gt).numpy()
        k = f"{idx:08}".encode("ascii")

        txn.put(k, pickle.dumps((inp_img, gt_img)))
        keys.append(k)

        if (idx+1) % write_frequency == 0:
            txn.commit()
            txn = env.begin(write=True)
            print(f"[LMDB] {idx+1} / {len(input_files)} 저장 완료")

    txn.commit()
    with env.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"__len__", pickle.dumps(len(keys)))

    env.close()
    print(f"[LMDB] 생성 완료: {lmdb_path}, 총 {len(keys)}개 샘플")


#####################################
# 2. LMDB Dataset 정의
#####################################
class LmdbDataset(Dataset):
    def __init__(self, lmdb_path, transform=None, normalize=True):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys   = pickle.loads(txn.get(b"__keys__"))
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env.begin(write=False) as txn:
            inp_img, gt_img = pickle.loads(txn.get(key))

        inp = torch.from_numpy(inp_img).float()
        gt  = torch.from_numpy(gt_img).float()

        if self.normalize:
            inp, gt = inp/255., gt/255.

        if self.transform:
            inp = self.transform(inp)

        return inp, gt

# ----------------------
# Transforms
# ----------------------
def two_crops_transform(patch=256):
    def _tf(x_pil: Image.Image, y_pil: Image.Image):
        W, H = x_pil.size
        if W < patch or H < patch:
            x_pil = x_pil.resize((max(W, patch), max(H, patch)), Image.BICUBIC)
            y_pil = y_pil.resize((max(W, patch), max(H, patch)), Image.BICUBIC)
            W, H = x_pil.size

        i1 = random.randint(0, H - patch)
        j1 = random.randint(0, W - patch)
        x1 = x_pil.crop((j1, i1, j1+patch, i1+patch))
        y1 = y_pil.crop((j1, i1, j1+patch, i1+patch))

        i2 = random.randint(0, H - patch)
        j2 = random.randint(0, W - patch)
        x2 = x_pil.crop((j2, i2, j2+patch, i2+patch))
        return x1, x2, y1
    return _tf

# ----------------------
# Utils
# ----------------------
def to_wandb_image(tensor, caption=None):
    # tensor: (B,C,H,W) or (C,H,W)
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.detach().float().clamp(0,1).cpu()
    return wandb.Image(arr, caption=caption)

if __name__ == '__main__':
# ------------------ WandB init ------------------
    wandb.init(
        project="airnet-finetune",
        name="run1",
        entity = "sjin0911yoon",
        config={
            "lr": opt.lr,
            "batch_size": 16,
            "epochs": opt.epochs,
            "cuda": opt.cuda,
            "patch_size": getattr(opt, "patch_size", 256),
            "tasks": ["rain", "fog", "dust"]
        }
    )
    tasks=["rain", "fog", "dust"]
    base_dir="/content/local_data/fulldata"

    lmdb_datasets = {}

    for task in tasks:
        input_dir = os.path.join(base_dir, task, "train/input")
        gt_dir    = os.path.join(base_dir, task, "train/gt")
        lmdb_path = os.path.join(base_dir, f"{task}_train.lmdb")

        # LMDB 없으면 생성
        create_lmdb(input_dir, gt_dir, lmdb_path)

        # 불러오기
        lmdb_datasets[task] = LmdbDataset(lmdb_path)

    all_train_dataset = ConcatDataset([lmdb_datasets[t] for t in tasks])

    train_loader = DataLoader(
        all_train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    for x, y in train_loader:
        print(x.shape, y.shape)  # (B, C, H, W)
        break

    torch.cuda.set_device(opt.cuda)
    os.makedirs(opt.ckpt_path, exist_ok=True)

    # ------------------ Dataset / Loader ------------------
    train_tf = two_crops_transform(patch=getattr(opt, "patch_size", 256))
    root =  "/content/local_data/fulldata"

    # trainset = RestoreFinetuneDataset(
    #     root=root,  # TODO: 실제 데이터 루트로 교체
    #     tasks=["rain", "fog", "dust"],
    #     split="train",
    #     transform_train=train_tf,
    #     mode="train",
    # )
    # valset = RestoreFinetuneDataset(
    #     root=root,
    #     tasks=["rain","fog","dust"],
    #     split="val",
    #     mode="eval"
    # )

    # # test dataset
    # testset = RestoreFinetuneDataset(
    #     root=root,
    #     tasks=["rain","fog","dust"],
    #     split="test",
    #     mode="eval"
    # )
    
    # train_loader = DataLoader(trainset, batch_size=5, shuffle=True, num_workers=0)
    # val_loader   = DataLoader(valset,   batch_size=5, shuffle=False, num_workers=0)
    # test_loader  = DataLoader(testset,  batch_size=5, shuffle=False, num_workers=0)


    # ------------------ Model / Opt / Loss ------------------
    net = AirNet(opt).cuda()
    ckpt_path = os.path.join(opt.ckpt_path, 'All.pth')
    if os.path.isfile(ckpt_path):
        net.load_state_dict(torch.load(ckpt_path, map_location=torch.device(opt.cuda)))

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss().cuda()

    scaler = torch.cuda.amp.GradScaler()

    # (선택) 그래프/그라드 감시
    wandb.watch(net, log="all", log_freq=100)

    # ------------------ Train ------------------
    print('Start finetuning...')
    best_loss = float("inf")

    for epoch in range(opt.epochs):
        # ---------------- Train ----------------
        net.train()
        running_loss = 0.0
        for step, (x, y) in enumerate(tqdm(train_loader)):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred = net(x)
                loss = l1(pred, y)  # 예시: L1 loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        wandb.log({"epoch": epoch + 1, "train/loss": avg_loss})

        # 🔹 모든 epoch 저장
        ckpt_epoch = os.path.join(opt.ckpt_path, f"epoch_{epoch+1}.pth")
        torch.save(net.state_dict(), ckpt_epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_best = os.path.join(opt.ckpt_path, "best.pth")
            torch.save(net.state_dict(), ckpt_best)

    ckpt_final = os.path.join(opt.ckpt_path, "last.pth")
    torch.save(net.state_dict(), ckpt_final)

    wandb.run.summary["best_train_loss"] = best_loss