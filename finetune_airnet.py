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
    base_dir="/content/local_data"


    torch.cuda.set_device(opt.cuda)
    os.makedirs(opt.ckpt_path, exist_ok=True)

    # ------------------ Dataset / Loader ------------------
    train_tf = two_crops_transform(patch=getattr(opt, "patch_size", 256))
    root =  "/content/local_data"

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

    
    scaler = torch.cuda.amp.GradScaler(enabled=True)  # 안 만들어져 있으면 한 줄 추가

    amp_ctx = torch.cuda.amp.autocast()

    # (선택) 그래프/그라드 감시
    wandb.watch(net, log="all", log_freq=100)

    # ------------------ Train ------------------
    print('Start finetuning...')
    best_loss = float("inf")

    
    for epoch in range(opt.epochs):
        net.train()
        running_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
            # batch 언패킹: ((x1,x2), y, meta) 또는 (x, y, meta)
            if isinstance(batch[0], (list, tuple)):
                x = batch[0][0]   # 첫 번째 crop만 사용
            else:
                x = batch[0]
            y = batch[1]
            # meta = batch[2]  # 필요하면 사용

            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                pred = net(x)
                loss = l1(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        wandb.log({"epoch": epoch + 1, "train/loss": avg_loss})

        # 체크포인트 저장
        os.makedirs(opt.ckpt_path, exist_ok=True)
        ckpt_epoch = os.path.join(opt.ckpt_path, f"epoch_{epoch+1}.pth")
        torch.save(net.state_dict(), ckpt_epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_best = os.path.join(opt.ckpt_path, "best.pth")
            torch.save(net.state_dict(), ckpt_best)

    # 마지막 저장 + 요약
    ckpt_final = os.path.join(opt.ckpt_path, "last.pth")
    torch.save(net.state_dict(), ckpt_final)
    wandb.run.summary["best_train_loss"] = best_loss