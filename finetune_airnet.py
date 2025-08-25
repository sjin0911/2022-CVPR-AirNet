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


    torch.cuda.set_device(opt.cuda)
    os.makedirs(opt.ckpt_path, exist_ok=True)

    # ------------------ Dataset / Loader ------------------
    train_tf = two_crops_transform(patch=getattr(opt, "patch_size", 256))
    root =  "/content/local_data/fulldata"

    trainset = RestoreFinetuneDataset(
        root=root,  # TODO: 실제 데이터 루트로 교체
        tasks=["rain", "fog", "dust"],
        split="train",
        transform_train=train_tf,
        mode="train",
    )
    valset = RestoreFinetuneDataset(
        root=root,
        tasks=["rain","fog","dust"],
        split="val",
        mode="eval"
    )

    # test dataset
    testset = RestoreFinetuneDataset(
        root=root,
        tasks=["rain","fog","dust"],
        split="test",
        mode="eval"
    )
    
    train_loader = DataLoader(trainset, batch_size=5, shuffle=True, num_workers=0)
    val_loader   = DataLoader(valset,   batch_size=5, shuffle=False, num_workers=0)
    test_loader  = DataLoader(testset,  batch_size=5, shuffle=False, num_workers=0)


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