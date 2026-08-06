import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--patch", type=int, required=True,
                     help="Trains patch with given index (0-20)")
args = parser.parse_args()

patch_id = args.patch

from functions import *

print(f"[patch {patch_id}] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[patch {patch_id}] torch.cuda.device_count() = {torch.cuda.device_count()}")
print(f"[patch {patch_id}] device name = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

train_on_patch(patch_idx=patch_id, scan_var='rotpmdec', fid_cuts=True)

print('All done!')