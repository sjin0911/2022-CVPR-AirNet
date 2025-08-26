import argparse
import subprocess
from tqdm import tqdm
import numpy as np
import math

import torch
from torch.utils.data import DataLoader
import torchvision

from utils.dataset_utils import TestSpecificDataset, PairMatchTestDataset
from utils.image_io import save_image_tensor
from pytorch_msssim import ssim

from net.model import AirNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--test_path', type=str, default="test/demo/", help='save path of test images')
    parser.add_argument('--output_path', type=str, default="output/demo/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint save path')
    opt = parser.parse_args()

    if opt.mode == 0:
        opt.batch_size = 3
        ckpt_path = opt.ckpt_path + 'Denoise.pth'
    elif opt.mode == 1:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'Derain.pth'
    elif opt.mode == 2:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'Dehaze.pth'
    elif opt.mode == 3:
        opt.batch_size = 5
        ckpt_path = opt.ckpt_path + 'All.pth'

    # construct the output dir
    subprocess.check_output(['mkdir', '-p', opt.output_path])

    np.random.seed(0)
    torch.manual_seed(0)

    # Make network
    torch.cuda.set_device(opt.cuda)
    net = AirNet(opt).cuda()
    net.eval()
    net.load_state_dict(torch.load(ckpt_path, map_location=torch.device(opt.cuda)))

    # test_set = TestSpecificDataset(opt)
    # testloader = DataLoader(test_set, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    ds = PairMatchTestDataset(
        input_root = "/content/2022-CVPR-AirNet/test/demo/Ensemble/input",
        gt_root = "/content/2022-CVPR-AirNet/test/demo/Ensemble/GT",
        input_list_txt = "/content/2022-CVPR-AirNet/test/demo/Ensemble/input_rain_ens.txt",
        base=16
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


    print('Start testing...')
    # with torch.no_grad():
    #     for ([clean_name], degrad_patch) in tqdm(testloader):
    #         degrad_patch = degrad_patch.cuda()

    #         restored = net(x_query=degrad_patch, x_key=degrad_patch)

    #         save_image_tensor(restored, opt.output_path + clean_name[0] + '.png')

    def psnr(x, y):
        mse = torch.mean((x - y) ** 2).item()
        if mse == 0: 
            return 99.0
        return 10 * math.log10(1.0 / mse)

    with torch.no_grad():
        for name, inp, gt in loader:
            inp = inp.cuda()
            pred = net(x_query = inp, x_key = inp)  # 네 모델 추론

            # 저장
            out_img = (pred.clamp(0,1) * 255).round().byte().cpu()[0]
            save_image_tensor(pred, opt.output_path + name[0] + '.png')

            # metric (GT 있을 때만)
            if gt[0] is not None:
                gt = gt.cuda()
                print(
                    name[0],
                    "PSNR:", round(psnr(pred, gt), 2),
                    "SSIM:", round(ssim(pred, gt, data_range=1.0, size_average=True).item(), 4)
                )