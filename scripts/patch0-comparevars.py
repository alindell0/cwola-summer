import torch

from functions import *

scan_vars = ['rotpmdec', 'rotpmra']

for i in range(len(scan_vars)):
    scan_var = scan_vars[i]
    train_on_patch(patch_idx=0, scan_var=scan_var, fid_cuts=True)

  
  
