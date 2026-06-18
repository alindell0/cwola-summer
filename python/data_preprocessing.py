import numpy as np
import pandas as pd
from astropy.table import Table
from tqdm import tqdm

# create numpy array of confirmed GD-1 stars from https://zenodo.org/records/1295543
tbl = Table.read('./../gaia-data/gd1/gd1-with-masks.fits')
tbl = tbl[tbl['pm_mask'] & tbl['gi_cmd_mask'] & tbl['stream_track_mask']]
np.save('./../gaia-data/gd1/gd1_stars.npy', tbl.as_array())


# functions from Dr. Pettee's repo https://github.com/hep-lbdl/GaiaCWoLa/tree/main
def angular_distance(angle1,angle2): # via_machinae function https://arxiv.org/abs/2104.12789
    # inputs are np arrays of [ra,dec]
    deltara=np.minimum(np.minimum(np.abs(angle1[:,0]-angle2[:,0]+360),
                                  np.abs(angle1[:,0]-angle2[:,0])),
                                  np.abs(angle1[:,0]-angle2[:,0]-360))
    deltadec=np.abs(angle1[:,1]-angle2[:,1])
    return np.sqrt(deltara**2+deltadec**2)
def FilterGD1(stars, gd1_stars):
    gd1stars=np.zeros(len(stars))
    for x in tqdm(gd1_stars):
        ra=x['ra']
        dec=x['dec']
        pmra=x['pmra']
        pmdec=x['pmdec']
        foundlist=angular_distance(np.dstack((stars[:,3],stars[:,2]))[0],np.array([[ra,dec]]))
        foundlist=np.sqrt(foundlist**2+(stars[:,0]-pmdec)**2+(stars[:,1]-pmra)**2)   
        foundlist=foundlist<.0001
        if len(np.argwhere(foundlist))>1:
            print(foundlist)
        if len(np.argwhere(foundlist))==1:
            gd1stars+=foundlist
    gd1stars=gd1stars.astype('bool')
    return gd1stars,stars[gd1stars]


# 21 GD-1 inclusive patches from doi.org/10.5281/zenodo.7897935
patch_list = [
     # b = 33.7 
     '../gaia-data/raw-data/gaiascan_l195.0_b33.7_ra128.4_dec28.8.npy',
     '../gaia-data/raw-data/gaiascan_l210.0_b33.7_ra132.6_dec16.9.npy',
     '../gaia-data/raw-data/gaiascan_l225.0_b33.7_ra138.1_dec5.7.npy', 
     # b = 41.8 
     '../gaia-data/raw-data/gaiascan_l187.5_b41.8_ra136.5_dec36.1.npy',
     '../gaia-data/raw-data/gaiascan_l202.5_b41.8_ra138.8_dec25.1.npy',
     '../gaia-data/raw-data/gaiascan_l217.5_b41.8_ra142.7_dec14.5.npy', 
     # b = 50.2 
     '../gaia-data/raw-data/gaiascan_l99.0_b50.2_ra224.7_dec60.6.npy',
     '../gaia-data/raw-data/gaiascan_l117.0_b50.2_ra202.4_dec66.5.npy',
     '../gaia-data/raw-data/gaiascan_l135.0_b50.2_ra174.3_dec65.1.npy',
     '../gaia-data/raw-data/gaiascan_l153.0_b50.2_ra156.2_dec57.5.npy',
     '../gaia-data/raw-data/gaiascan_l171.0_b50.2_ra148.6_dec47.0.npy',
     '../gaia-data/raw-data/gaiascan_l189.0_b50.2_ra146.9_dec35.6.npy',
     '../gaia-data/raw-data/gaiascan_l207.0_b50.2_ra148.6_dec24.2.npy',
     # b = 58.4 
     '../gaia-data/raw-data/gaiascan_l101.2_b58.4_ra212.7_dec55.2.npy',
     '../gaia-data/raw-data/gaiascan_l123.8_b58.4_ra192.0_dec58.7.npy',
     '../gaia-data/raw-data/gaiascan_l146.2_b58.4_ra171.8_dec54.7.npy',
     '../gaia-data/raw-data/gaiascan_l168.8_b58.4_ra160.5_dec45.5.npy',
     '../gaia-data/raw-data/gaiascan_l191.2_b58.4_ra156.9_dec34.1.npy',
     # b = 66.4 
     '../gaia-data/raw-data/gaiascan_l105.0_b66.4_ra203.7_dec49.1.npy',
     '../gaia-data/raw-data/gaiascan_l135.0_b66.4_ra185.4_dec50.0.npy',
     '../gaia-data/raw-data/gaiascan_l165.0_b66.4_ra171.4_dec43.0.npy',    
    ]

gd1 = np.load('./../gaia-data/gd1/gd1_stars.npy')

# clean file columns, cross-check as GD-1, save as hdf5 files
for patch_id in range(2):
    file = patch_list[patch_id]

    colnames = ["pmdec", "pmra", "dec", "ra", "b-r", "g", 
                "rotra", "rotdec", "rotpmra", "rotpmdec", "del1", "del2"]
    df_raw = pd.DataFrame(np.load(file), columns=colnames)
    df = df_raw.drop(columns=['del1','del2'])

    bool_stream, stream = FilterGD1(np.array(df), gd1)
    df['stream'] = bool_stream

    df.to_hdf(f'../gaia-data/cleaned-data/df_patch{patch_id}.h5', key=f'data', mode='w')
    print(f'Patch {patch_id} cleaned and saved.')
