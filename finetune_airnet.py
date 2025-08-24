import os, subprocess, random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
            "batch_size": 5,
            "epochs": opt.epochs,
            "cuda": opt.cuda,
            "patch_size": getattr(opt, "patch_size", 256),
            "tasks": ["rain", "fog", "dust"]
        }
    )

    torch.cuda.set_device(opt.cuda)
    os.makedirs(opt.ckpt_path, exist_ok=True)

    # ------------------ Dataset / Loader ------------------
    train_tf = two_crops_transform(patch=getattr(opt, "patch_size", 256))
    finetuneset = RestoreFinetuneDataset(
        root="/content/drive/MyDrive/miniproject1/AirNet/Data/fulldata",  # TODO: 실제 데이터 루트로 교체
        tasks=["rain", "fog", "dust"],
        split="train",
        transform_train=train_tf,
        mode="train",
    )
    finetuneloader = DataLoader(
        finetuneset,
        batch_size=5,
        shuffle=True,
        num_workers=2,           
        pin_memory=True,
        persistent_workers=True,
    )

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
        net.train()
        running_loss = 0.0

        for step, ((x_q, x_k), y, meta) in enumerate(tqdm(finetuneloader)):
            x_q = x_q.cuda(non_blocking=True)
            x_k = x_k.cuda(non_blocking=True)
            y   = y.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                restored, output, target = net(x_query=x_q, x_key=x_k)
                contrast_loss = CE(output, target)
                l1_loss = l1(restored, y)
                loss = l1_loss + 0.1 * contrast_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # ---- step-wise log (간단 메트릭) ----
            if (step + 1) % 50 == 0 or (step == 0):
                # 샘플 이미지 로그 (부하 방지: 가끔만)
                log_imgs = {}
                try:
                    log_imgs["restored"] = to_wandb_image(restored)
                    log_imgs["target"]   = to_wandb_image(y)
                    log_imgs["query"]    = to_wandb_image(x_q)
                except Exception:
                    pass

                wandb.log({
                    "step": step + epoch * len(finetuneloader),
                    "train/loss": loss.item(),
                    "train/l1_loss": l1_loss.item(),
                    "train/contrast_loss": contrast_loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    **({k: v for k, v in log_imgs.items()} if log_imgs else {})
                })

        avg_loss = running_loss / len(finetuneloader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "epoch/train_loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(opt.ckpt_path, "best.pth")
            torch.save(net.state_dict(), best_path)
            print(f"  -> New best model saved! (loss {best_loss:.4f})")
            wandb.run.summary["best_loss"] = best_loss
            wandb.run.summary["best_ckpt"] = best_path

    print("Done.")
    wandb.finish()
