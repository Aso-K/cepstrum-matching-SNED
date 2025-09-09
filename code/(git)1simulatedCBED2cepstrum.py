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

# ----------------------------------
# Helpers
# ----------------------------------
def scale_bar(pix_size, ax=None, unit='nm$^{-1}$', bar_length=1, px=0.97, py=0.95, fs=22, c='w'):
    """Draw a scale bar of length `bar_length` (in `unit`) onto an axis where
    1 pixel corresponds to `pix_size` (in `unit`/pixel)."""
    if ax is None:
        ax = plt.gca()
    _, x_size = ax.get_xlim()
    y_size, _ = ax.get_ylim()
    bar_px = bar_length / pix_size
    hline = patches.Rectangle(
        (x_size * px - bar_px, y_size * py),
        bar_px, y_size * 0.015,
        alpha=1, facecolor=c,
        path_effects=[patheffects.withStroke(linewidth=3, foreground='k', capstyle="round")]
    )
    ax.add_patch(hline).set_zorder(100)


def scale_bar_cepstrum(pix_size2, ax=None, unit='nm', bar_length=0.5, px=0.97, py=0.95, fs=22, c='w'):
    """Draw a scale bar for the cepstrum image where 1 pixel corresponds to `pix_size2` (in `unit`/pixel)."""
    if ax is None:
        ax = plt.gca()
    _, x_size = ax.get_xlim()
    y_size, _ = ax.get_ylim()
    bar_px = bar_length / pix_size2
    hline = patches.Rectangle(
        (x_size * px - bar_px, y_size * py),
        bar_px, y_size * 0.015,
        alpha=1, facecolor=c,
        path_effects=[patheffects.withStroke(linewidth=3, foreground='k', capstyle="round")]
    )
    ax.add_patch(hline).set_zorder(100)


def create_rotational_symmetric_hann_window(shape):
    """Create a rotationally symmetric Hann-like window on a unit disk."""
    if len(shape) != 2:
        raise ValueError("Shape must be 2-dimensional")
    rows, cols = shape
    y = np.linspace(-1, 1, rows)[:, np.newaxis]
    x = np.linspace(-1, 1, cols)[np.newaxis, :]
    distance = np.sqrt(x**2 + y**2)
    r = np.clip(distance, 0, 1)
    return 0.5 * (1 - np.cos(np.pi * (1 - r)))

# ----------------------------------
# Settings
# ----------------------------------
cmap = 'cmo.deep_r'
cmap_cbed = 'cmo.thermal'
plt.rcParams['image.cmap'] = cmap  # default colormap; individual plots set as needed

folder2 = ''
# Choose one pattern file:
file2 = 'Co3O4 10-110nm.png'
# file2 = 'LiCoO2 O3 (I) 10-110nm.png'
# file2 = 'CoO (NaCl) 10-110nm.png'
# file2 = 'SrTiO3 [-110].png'

savepath = folder2 + file2.split('.png')[0]

oversample_size = 512
crop_hw_cepst = 200
tol = 10

# Window size used in the original script (kept as-is)
hann2D = create_rotational_symmetric_hann_window((264, 264))

# Pixel size (reciprocal-nm per pixel) used for the CBED image scale bar (kept as-is)
pix_size = 0.07974688 * 2
# Cepstrum pixel size in nm per pixel (spatial domain of the cepstrum image)
pix_size_cepst = (1 / pix_size) / oversample_size

bg = 1e-15  # small positive offset to avoid log(0)

# ----------------------------------
# Load & pre-process image
# ----------------------------------
img_raw = rgb2gray(imread(folder2 + file2)[:, :, :3])

# center-crop to square if needed
if img_raw.shape[1] != img_raw.shape[0]:
    crop_hw_adjust = (img_raw.shape[1] - img_raw.shape[0]) // 2
    img_raw = img_raw[:, crop_hw_adjust:-crop_hw_adjust]

# specific orientation tweak (kept for reproducibility)
if file2 == 'SrTiO3 [1-10].png':
    img_raw = np.flipud(img_raw)

# downscale for noise reduction / anti-aliasing (kept)
img = rescale(img_raw, 0.5, anti_aliasing=True)
img_ = img.copy() + bg
img_hw = int(np.round(img.shape[0] / 2))

# ----------------------------------
# Cepstrum computation
# ----------------------------------
img2 = np.zeros((oversample_size, oversample_size)) + bg
log_img = np.log(img + bg)

# window & mean subtraction
wmean = np.average(log_img, weights=hann2D)
img3 = (log_img - wmean) * hann2D
img2[:264, :264] = img3  # pad into oversampled canvas

cepstrum = np.abs(np.fft.fftshift(np.fft.fft2(img2)))

# contrast helper for CBED display (mask center to expand range stats)
crop_hh = 12
img_for_contrast = img_.copy()
img_for_contrast[img_hw - crop_hh:img_hw + crop_hh, img_hw - crop_hh:img_hw + crop_hh] = 0

crop_hw_img = 90
crop_cepst = cepstrum[crop_hw_cepst:oversample_size - crop_hw_cepst + 1,
                      crop_hw_cepst:oversample_size - crop_hw_cepst + 1]
crop_cepst_ = crop_cepst.copy()

# discard central square for percentile estimation
cc = crop_cepst_.shape[0] // 2
crop_cepst_[cc - tol:cc + tol, cc - tol:cc + tol] = np.nan
crop_vals = crop_cepst_[~np.isnan(crop_cepst_)]

# robust display ranges
vmin_cepst = np.percentile(crop_vals, 80)
vmax_cepst = np.percentile(crop_vals, 99.5)
vmin_img = np.percentile(img_for_contrast[img_for_contrast != 0], 1)
vmax_img = np.percentile(img_for_contrast[img_for_contrast != 0], 99.9)

# ----------------------------------
# Visualization (kept, with minor cleanups)
# ----------------------------------
plt.figure()
plt.imshow(cepstrum)
plt.title('Cepstrum (raw)')
plt.colorbar()

plt.figure()
plt.imshow(crop_cepst)
plt.title('Cepstrum (crop, raw)')
plt.colorbar()

plt.figure()
plt.hist(crop_vals, bins=500)
plt.yscale('log')
plt.title('Cepstrum histogram (crop, center removed)')

# CBED
plt.figure()
plt.imshow(img, cmap=cmap_cbed, vmin=vmin_img, vmax=vmax_img)
scale_bar(pix_size, bar_length=5)
plt.axis('off')
plt.title('CBED')

# log(CBED)
plt.figure()
plt.imshow(log_img, cmap=cmap_cbed)
scale_bar(pix_size, bar_length=5)
plt.axis('off')
plt.title('log(CBED)')

# Hann window
plt.figure()
plt.imshow(hann2D, cmap=cmap_cbed)
plt.axis('off')
plt.colorbar()
plt.title('Hann window (rotationally symmetric)')

# windowed CBED (padded canvas)
plt.figure()
plt.imshow(img2, cmap=cmap_cbed)
scale_bar(pix_size, bar_length=5)
plt.axis('off')
plt.title('Windowed + padded CBED')

# windowed CBED (just the windowed block)
plt.figure()
plt.imshow(img3, cmap=cmap_cbed)
scale_bar(pix_size, bar_length=5)
plt.axis('off')
plt.title('Windowed CBED (block)')

# cepstrum with display range
plt.figure()
plt.imshow(cepstrum, vmin=vmin_cepst, vmax=vmax_cepst, cmap=cmap)
scale_bar_cepstrum(pix_size_cepst)
plt.axis('off')
plt.title('Cepstrum')

# CBED (contrast-focused crop)
plt.figure()
plt.imshow(img_[crop_hw_img:264 - crop_hw_img, crop_hw_img:264 - crop_hw_img],
           vmin=vmin_img, vmax=vmax_img, cmap=cmap_cbed)
scale_bar(pix_size, bar_length=5)
plt.axis('off')
plt.title('CBED (center-cropped for contrast)')

# cepstrum (cropped, display range)
plt.figure()
plt.imshow(crop_cepst, vmin=vmin_cepst, vmax=vmax_cepst, cmap=cmap)
scale_bar_cepstrum(pix_size_cepst)
plt.axis('off')
plt.title('Cepstrum (crop)')

print("crop_cepst shape:", crop_cepst.shape)

np.savetxt(folder2 + 'cepstrum_pixsize_nm-per-pixel.txt', [pix_size_cepst])
np.save(savepath + '_crop-cepstrum.npy', crop_cepst)
