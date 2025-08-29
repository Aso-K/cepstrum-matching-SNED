"""
Cepstrum Matching of SNED
1. Generating simulated cepstrum from diffraction patterns
---------------------------------------------------------
This script accompanies the manuscript under review.
It demonstrates cepstrum analysis of convergent beam electron diffraction (CBED)
patterns using PNG images.

Disclaimer:
This code is provided as-is, without warranty of any kind. See DISCLAIMER.md.

Notes:
- The simulated electron diffraction patterns in the manuscript
  were originally calculated using the software *Recipro*.
  https://yseto.net/en/software/recipro
- Colormaps from cmocean are used (cmo.deep_r, cmo.thermal); if unavailable,
  replace with standard matplotlib colormaps.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rescale
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects


def scale_bar(pix_size, ax=None, unit='nm$^{-1}$', bar_length=1, px=0.97, py=0.95, 
              fs=22, c='w'):
    
    if ax == None:
        ax = plt.gca()
    
    _, x_size = ax.get_xlim()
    y_size, _ = ax.get_ylim()
    hline = patches.Rectangle((x_size*px-bar_length/pix_size, y_size*py), 
                              bar_length/pix_size, y_size*0.015,
                              alpha=1, facecolor=c,
                               path_effects=[patheffects.withStroke(linewidth=3, foreground='k', capstyle="round")]
                              )
    hline_ax = ax.add_patch(hline)
    hline_ax.set_zorder(100)
    
    # ax.text(px*x_size, (py-0.005)*y_size, 
    #             '%i %s'%(bar_length, unit),
    #             va='bottom', ha='right', c='w', fontsize=fs,
    #             path_effects=[patheffects.withStroke(linewidth=3, foreground='k', capstyle="round")])
    return

def scale_bar_cepstrum(pix_size2, ax=None, unit='nm', bar_length=0.5, px=0.97, py=0.95, 
              fs=22, c='w'):
    
    if ax == None:
        ax = plt.gca()
    # pix_size2 = 1/pix_size/264
    # print(pix_size2)
    _, x_size = ax.get_xlim()
    y_size, _ = ax.get_ylim()
    hline = patches.Rectangle((x_size*px-bar_length/pix_size2, y_size*py), 
                              bar_length/pix_size2, y_size*0.015,
                              alpha=1, facecolor=c,
                               path_effects=[patheffects.withStroke(linewidth=3, foreground='k', capstyle="round")]
                              )
    hline_ax = ax.add_patch(hline)
    hline_ax.set_zorder(100)
    
    # ax.text(px*x_size, (py-0.005)*y_size, 
    #             '%i %s'%(bar_length, unit),
    #             va='bottom', ha='right', c='w', fontsize=fs,
    #             path_effects=[patheffects.withStroke(linewidth=3, foreground='k', capstyle="round")])
    return


def create_rotational_symmetric_hann_window(shape):
    if len(shape) != 2:
        raise ValueError("Shape must be 2-dimensional")
    
    rows, cols = shape
    y = np.linspace(-1, 1, rows)[:, np.newaxis]
    x = np.linspace(-1, 1, cols)[np.newaxis, :]
    
    distance = np.sqrt(x**2 + y**2)
    
    normalized_distance = np.clip(distance, 0, 1)
    
    hann_window = 0.5 * (1 - np.cos(np.pi * (1 - normalized_distance)))
    
    return hann_window

cmap = 'cmo.deep_r'
cmap_cbed = 'cmo.thermal'
    
folder2 = ''

file2 = 'LiCoO2 O3 (I) 10-110nm.png'

file2 = 'Co3O4 10-110nm.png'
# file2 = 'CoO (NaCl) 10-110nm.png'


# file2 = 'SrTiO3 [-110].png'


savepath = folder2 + file2.split('.png')[0]


oversample_size = 512
crop_hw_cepst = 200 # 180
tol = 10

# oversample_size = 264
# crop_hw_cepst = 100
# tol = 3


hann2D = create_rotational_symmetric_hann_window((264,264))

pix_size = 0.07974688*2
pix_size_cepst = 1/(pix_size)/oversample_size
ind = 0
# file   = files[ind]


bg = 1e-15

img_raw = rgb2gray(imread(folder2+file2)[:,:,:3])
if img_raw.shape[1] != img_raw.shape[0]:
    print(img_raw.shape[1] != img_raw.shape[0])
    crop_hw_adjust = (img_raw.shape[1] - img_raw.shape[0])//2
    img_raw = img_raw[:,crop_hw_adjust:-crop_hw_adjust]
    print(img_raw.shape[1] != img_raw.shape[0])

#%%

if file2 == 'SrTiO3 [1-10].png':
    img_raw = np.flipud(img_raw)


# img_raw = rgb2gray(imread(folder+file)[:,:,:3])
img = rescale(img_raw, 0.5, anti_aliasing=True, )
img_ = img.copy() + bg
img_hw = int(np.round(img.shape[0]/2))

noise = ((np.random.normal(size = img.shape))+3)/200
noise[noise<0] = 0


img2 = np.zeros((oversample_size, oversample_size)) + bg

# log_img2 = np.log(img2 + bg)
log_img = np.log(img + bg)


wmean = np.average(log_img, weights=hann2D)
img3 = (log_img - wmean)*hann2D
img2[:264,:264] = img3


cepstrum = ((np.abs(np.fft.fftshift(np.fft.fft2(img2 + bg)))))
# cepstrum_bg = gaussian_filter(cepstrum_, 50)
#  = cepstrum_ - cepstrum_bg

plt.figure()
plt.imshow(cepstrum)
crop_hh = 12
img_for_contrast = img_.copy()
img_for_contrast[img_hw-crop_hh:img_hw+crop_hh,img_hw-crop_hh:img_hw+crop_hh] = 0


# vmin_img = np.percentile(img_for_contrast[img_for_contrast!=0], 10)
# vmax_img = np.percentile(img_for_contrast[img_for_contrast!=0], 0)

crop_hw_img = 90

crop_cepst = (cepstrum[crop_hw_cepst:oversample_size-crop_hw_cepst+1, 
                              crop_hw_cepst:oversample_size-crop_hw_cepst+1])
crop_cepst_ = crop_cepst.copy()

crop_cepst_size = crop_cepst.shape[0]
crop_cepst_[crop_cepst_size//2-tol:crop_cepst_size//2+tol, 
            crop_cepst_size//2-tol:crop_cepst_size//2+tol] = np.nan
crop_cepst__ = crop_cepst_[~np.isnan(crop_cepst_)]

plt.figure()
plt.imshow(crop_cepst_)

plt.figure()
# plt.hist(crop_cepst_.flatten(), bins=500)
plt.hist(crop_cepst__, bins=500)
plt.yscale('log')


vmin_cepst = np.percentile(crop_cepst__, 80)
vmax_cepst = np.percentile(crop_cepst__, 99.5)

vmin_img = np.percentile(img_for_contrast[img_for_contrast!=0], 1)
vmax_img = np.percentile(img_for_contrast[img_for_contrast!=0], 99.9)


### show CBED
plt.figure()
plt.imshow(img, cmap=cmap_cbed)
scale_bar(pix_size, bar_length=5)
plt.axis('off')


### show log(CBED)
plt.figure()
plt.imshow(log_img, cmap=cmap_cbed)
scale_bar(pix_size, bar_length=5)
plt.axis('off')

### show hann2D
plt.figure()
plt.imshow(hann2D, 
           # norm=LogNorm(vmin_img,vmax_img),
           # vmin=vmin_img,vmax=vmax_img,
           cmap=cmap_cbed)
# scale_bar(pix_size, bar_length=5)
plt.axis('off')
plt.colorbar()

### show windowed CBED
plt.figure()
plt.imshow((img2), 
           # norm=LogNorm(vmin_img,vmax_img),
           # vmin=vmin_img,vmax=vmax_img,
           cmap=cmap_cbed)
scale_bar(pix_size, bar_length=5)
plt.axis('off')

### show padded CBED
plt.figure()
plt.imshow((img3), 
           # norm=LogNorm(vmin_img,vmax_img),
           # vmin=vmin_img,vmax=vmax_img,
           cmap=cmap_cbed)
scale_bar(pix_size, bar_length=5)
plt.axis('off')


plt.figure()
plt.imshow(cepstrum, 
            vmin = vmin_cepst, vmax=vmax_cepst,
           cmap=cmap)
scale_bar_cepstrum(pix_size_cepst)
plt.axis('off')
# plt.colorbar()


plt.figure()
plt.imshow((img_), 
           # norm=LogNorm(vmin_img,vmax_img),
           vmin=vmin_img,vmax=vmax_img,
           cmap=cmap_cbed)
scale_bar(pix_size, bar_length=5)
plt.axis('off')

plt.figure()
plt.imshow((img_[crop_hw_img:264-crop_hw_img, 
                          crop_hw_img:264-crop_hw_img]), 
           # norm=LogNorm(vmin_img,vmax_img),
           vmin=vmin_img,vmax=vmax_img,
           cmap=cmap_cbed)
scale_bar(pix_size, bar_length=5)
plt.axis('off')

plt.figure()
# plt.scatter(crop_cepst_size//2,crop_cepst_size//2, zorder=1000, marker='x', c='w', s=500, lw=4, alpha=0.8)
# plt.scatter(crop_cepst_size//2,crop_cepst_size//2, zorder=1200, marker='x', c='w', s=170, lw=3)
# plt.scatter(52.5,17.5, zorder=1000, marker='^', s=1600, lw=2, facecolor='None', edgecolors='w', alpha=0.8)
# plt.scatter(72.3,20.7, zorder=1000, marker='o', s=1100, lw=2, facecolor='None', edgecolors='w', alpha=0.8)
plt.imshow(crop_cepst, 
            vmin = vmin_cepst, vmax=vmax_cepst,
           cmap=cmap)
scale_bar_cepstrum(pix_size_cepst)
plt.axis('off')
# plt.colorbar()

print(crop_cepst.shape)

#np.savetxt(folder2 + 'cepstrum_pixsize_nm-per-pixel.txt', [pix_size_cepst])
# np.save(savepath + '_crop-cepstrum.npy', crop_cepst)