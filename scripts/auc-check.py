from functions import *
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score



df_rotpmdec = pd.read_hdf('../results/patch_0/rotpmdec/training/df_test.h5')
print('Rotated Proper Motion in Declination (rotpmdec) AUC and Average Precision:')

auc = roc_auc_score(df_rotpmdec['stream'].astype(int), df_rotpmdec['nn_score'])
print(f"AUC: {auc:.3f}")

## better for datasets with class imbalance, very rare signal
ap = average_precision_score(df_rotpmdec['stream'].astype(int), df_rotpmdec['nn_score'])
print(f"Average Precision: {ap:.3f}")



# df_rotpmra = pd.read_hdf('../results/patch_0/rotpmra/training/df_test.h5')
# print('Rotated Proper Motion in Right Ascension (rotpmra) AUC and Average Precision:')

# auc = roc_auc_score(df_rotpmra['stream'].astype(int), df_rotpmra['nn_score'])
# print(f"AUC: {auc:.3f}")

# ap = average_precision_score(df_rotpmra['stream'].astype(int), df_rotpmra['nn_score'])
# print(f"Average Precision: {ap:.3f}")
