import subprocess
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset
from net.model import AirNet

from option_finetune import options as opt

if __name__ == '__main__':
    torch.cuda.set_device(opt.cuda)
    subprocess.check_output(['mkdir', '-p', opt.ckpt_path])

    finetuneset = TrainDataset(opt)
    finetuneloader = DataLoader(finetuneset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    # Network Construction
    net = AirNet(opt).cuda()
    ckpt = torch.load("ckpt/airnet_allinone.pth", map_location="cpu")
    net.load_state_dict(ckpt, strict=False)
    net.train()

    # Optimizer and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss().cuda()

    # Start training
    print('Start finetuning...')
    for epoch in range(opt.epochs):
        running_loss = 0.0
        for ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2) in tqdm(finetuneloader):
            degrad_patch_1, degrad_patch_2 = degrad_patch_1.cuda(), degrad_patch_2.cuda()
            clean_patch_1 = clean_patch_1.cuda()

            optimizer.zero_grad()

            restored, output, target = net(x_query=degrad_patch_1, x_key=degrad_patch_2)
            contrast_loss = CE(output, target)
            l1_loss = l1(restored, clean_patch_1)
            loss = l1_loss + 0.1 * contrast_loss

            # backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(finetuneloader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} (L1 {l1_loss.item():.4f}, CE {contrast_loss.item():.4f})")

        # Best loss 기준 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(opt.ckpt_path, "best.pth")
            torch.save(net.state_dict(), save_path)
            print(f"  -> New best model saved! (loss {best_loss:.4f})")