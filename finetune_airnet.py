import subprocess
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.finetune_dataset_utils import RestoreFinetuneDataset
from net.model import AirNet

from option_finetune import options as opt

import random
from PIL import Image

def two_crops_transform(patch=256):
    def _tf(x_pil: Image.Image, y_pil: Image.Image):
        W, H = x_pil.size
        if W < patch or H < patch:
            # 작은 이미지는 리사이즈(선호 크기: 최소 patch 이상)
            x_pil = x_pil.resize((max(W, patch), max(H, patch)), Image.BICUBIC)
            y_pil = y_pil.resize((max(W, patch), max(H, patch)), Image.BICUBIC)
            W, H = x_pil.size

        # 첫 뷰
        i1 = random.randint(0, H - patch)
        j1 = random.randint(0, W - patch)
        x1 = x_pil.crop((j1, i1, j1+patch, i1+patch))
        y1 = y_pil.crop((j1, i1, j1+patch, i1+patch))

        # 두 번째 뷰 (독립 크롭)
        i2 = random.randint(0, H - patch)
        j2 = random.randint(0, W - patch)
        x2 = x_pil.crop((j2, i2, j2+patch, i2+patch))

        # (원하면) 밝기/좌우반전 등 약한 증강 추가 가능
        return x1, x2, y1
    return _tf


if __name__ == '__main__':
    torch.cuda.set_device(opt.cuda)
    subprocess.check_output(['mkdir', '-p', opt.ckpt_path])

    best_loss = float("inf")
    
    train_tf = two_crops_transform(patch= opt.patch_size)
    finetuneset = RestoreFinetuneDataset(
        root="DATA_ROOT",
        tasks = ["rain", "fog", "dust"],
        split="train",
        transform_train=train_tf,
        mode = "train"
    )
    finetuneloader = DataLoader(
        finetuneset,
        batch_size = 5,
        shuffle =True,
        num_workders=2,
        pin_memory = True,
        persistent_workers=True,
    )


    opt.batch_size = 5
    ckpt_path = opt.ckpt_path + 'All.pth'

    # Network Construction
    torch.cuda.set_device(opt.cuda)
    net = AirNet(opt).cuda()
    net.load_state_dict(torch.load(ckpt_path, map_location=torch.device(opt.cuda)))

    # Optimizer and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss().cuda()

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler()

    # Start training
    print('Start finetuning...')
    for epoch in range(opt.epochs):
        running_loss = 0.0
        for (x_q, x_k), y, meta in tqdm(finetuneloader):
            x_q = x_q.cuda(non_blocking=True)
            x_k = x_k.cuda(non_blocking=True)
            y   = y.cuda(non_blocking=True)

            optimizer.zero_grad()

            # ===== AMP 영역 =====
            with torch.cuda.amp.autocast():
                restored, output, target = net(x_query=x_q, x_key=x_k)
                contrast_loss = CE(output, target)
                l1_loss = l1(restored, y)
                loss = l1_loss + 0.1 * contrast_loss
            # ====================

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(finetuneloader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), os.path.join(opt.ckpt_path, "best.pth"))
            print(f"  -> New best model saved! (loss {best_loss:.4f})")
