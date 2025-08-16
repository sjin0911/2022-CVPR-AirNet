# option_finetune.py
import argparse

parser = argparse.ArgumentParser()

# Device
parser.add_argument('--cuda', type=int, default=0)

# Training
parser.add_argument('--epochs', type=int, default=100, help='number of finetune epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for finetuning')

# Data
parser.add_argument('--de_type', type=list, default=['derain', 'dehaze', 'dust'],
                    help='degradation types for finetuning')
parser.add_argument('--patch_size', type=int, default=128, help='patch size of input')
parser.add_argument('--encoder_dim', type=int, default=256, help='dimensionality of encoder')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers')

# Path
parser.add_argument('--data_file_dir', type=str, default='data_dir/', help='where clean images are saved')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/', help='deraining dataset path')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/', help='dehazing dataset path')
parser.add_argument('--dust_dir', type=str, default='data/Train/Dust/', help='dust dataset path')
parser.add_argument('--output_path', type=str, default="output_finetune/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt_finetune/", help='checkpoint save path')

options = parser.parse_args()
options.batch_size = len(options.de_type)
