
### imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




### loading .h5 file

def load_data(patch_idx):

    # load the .h5 data file into a dataframe, uses patch index (0-20)

    df = pd.read_hdf(f"../gaia-data/cleaned-data/gd1_patch{patch_idx}.h5")
    print(f"Data for Patch {patch_idx}")
    return df



def plot_data(df):
    
    # create figure with 1 row and 3 columns for data subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
    fig.suptitle('Full Patch Data')
    color = 'Greys'

    # plot position coordinates
    binspos = np.linspace(-15,15,100), np.linspace(-15,15,100)
    h = axes[0].hist2d(df['rotra'], df['rotdec'], bins=binspos, cmap=color, cmin=0, vmax=250)
    axes[0].set_xlabel('Rotated Right Ascension ϕ', fontsize = 10)
    axes[0].set_ylabel('Rotated Declination λ', fontsize = 10)
    c = fig.colorbar(h[3], ax=axes[0])
    c.ax.set_title('Counts', fontsize=8)


    # plot proper motion coordinates
    binspm = (np.linspace(-20,20,100), np.linspace(-20,20,100))
    h = axes[1].hist2d(df['rotpmra'], df['rotpmdec'], bins=binspm, cmap=color, cmin=0)
    axes[1].set_xlabel('Rotated Right Ascension Proper Motion μ_ϕcosλ', fontsize = 10)
    axes[1].set_ylabel('Rotated Declination Proper Motion μ_λ', fontsize = 10)  
    c = fig.colorbar(h[3], ax=axes[1])
    c.ax.set_title('Counts', fontsize=8)

    # plot photometric figures
    binsfeat = (np.linspace(0,3,100), np.linspace(10,21,100))
    h = axes[2].hist2d(df['b-r'], df['g'], bins=binsfeat, cmap=color, cmin=0)
    axes[2].set_xlabel('Color b-r', fontsize = 10)
    axes[2].set_ylabel('Magnitude g', fontsize = 10)
    c = fig.colorbar(h[3], ax=axes[2])
    c.ax.set_title('Counts', fontsize=8)

    plt.show()





### cleaning up and converting raw .npy data to .h5, functions from Dr. Pettee's repo https://github.com/hep-lbdl/GaiaCWoLa/tree/main

def angular_distance(angle1,angle2): # via_machinae function https://arxiv.org/abs/2104.12789
    # inputs are np arrays of [ra,dec]
    deltara=np.minimum(np.minimum(np.abs(angle1[:,0]-angle2[:,0]+360),
                                  np.abs(angle1[:,0]-angle2[:,0])),
                                  np.abs(angle1[:,0]-angle2[:,0]-360))
    deltadec=np.abs(angle1[:,1]-angle2[:,1])
    return np.sqrt(deltara**2+deltadec**2)
# function that cross checks GD-1 stars with raw patch data, returns boolean array

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