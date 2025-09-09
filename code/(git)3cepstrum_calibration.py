"""
Cepstrum Matching of SNED
3. Calibration of experimental cepstrum
---------------------------------------------------------
This script accompanies the manuscript under review.

Disclaimer:
This code is provided as-is, without warranty of any kind. See DISCLAIMER.md.

Notes:
- Colormaps from cmocean are used (cmo.deep_r, cmo.thermal); if unavailable,
  replace with standard matplotlib colormaps.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import warp_polar
from skimage.registration import phase_cross_correlation
from skimage import transform

# ----------------------------------
# Settings
# ----------------------------------
cmap = 'cmo.deep_r'
plt.rcParams['image.cmap'] = cmap

folder = ''  # base folder for inputs/outputs
file = 'ewpc_tol10_bg1.0e+01_os512_crop97.npy'  # EWPC data (oversampled cepstrum stack)

# infer oversample size from filename (e.g., *_os512_*)
oversample_size = int(file.split('_os')[-1].split('_crop')[0])

pix_size = 0.07974688 * 2  # nm^{-1} per pixel in CBED
pix_size_cepst = 1 / pix_size / oversample_size  # nm per pixel in cepstrum domain

savefile = os.path.splitext(file)[0]
save_folder = os.path.join(folder, savefile)
savepath = os.path.join(save_folder, savefile)

path_ewpc_data = os.path.join(save_folder, file)
path_df_src = os.path.join(folder, 'df_img.npy')
path_df = os.path.join(save_folder, 'df_img.npy')

os.makedirs(save_folder, exist_ok=True)

# move EWPC data into the save folder (keep original behavior)
if not os.path.exists(path_ewpc_data) and os.path.exists(os.path.join(folder, file)):
    shutil.move(os.path.join(folder, file), path_ewpc_data)

# move DF image into the save folder (keep original behavior)
if not os.path.exists(path_df) and os.path.exists(path_df_src):
    shutil.move(path_df_src, path_df)

# ----------------------------------
# Load data
# ----------------------------------
ewpc_data = np.load(path_ewpc_data)  # shape: (H, W, Y, X)
df_img = np.load(path_df)

# Summaries for visualization
ewpc_sum = ewpc_data.sum(axis=(2, 3))  # sum over scan positions (Y, X)

def func_limit_xy(limit_in_xy):
    """Average EWPC over a rectangular region in scan coords (y_start, y_end, x_start, x_end)."""
    y0, y1, x0, x1 = limit_in_xy
    return ewpc_data[:, :, y0:y1, x0:x1].mean(axis=(2, 3))

# ----------------------------------
# Quick check at a specific scan position
# ----------------------------------
# Example position (row, col) = (y, x)
pos_x = [80, 170]  # (y, x)

ewpc_extract = ewpc_data[:, :, pos_x[0], pos_x[1]]

plt.figure(figsize=(8, 2))
plt.subplot(1, 3, 1)
plt.imshow(ewpc_sum,
           vmin=np.percentile(ewpc_sum, 10),
           vmax=np.percentile(ewpc_sum, 99),
           cmap=cmap)
plt.title('EWPC sum over scan')

plt.subplot(1, 3, 2)
plt.imshow(df_img, cmap='gray')
plt.plot(pos_x[1], pos_x[0], '+', c='magenta')
plt.title('DF image (marker)')

plt.subplot(1, 3, 3)
vmin_loc = np.percentile(ewpc_extract, 10)
vmax_loc = np.percentile(ewpc_extract, 99)
plt.imshow(ewpc_extract, cmap=cmap, vmin=vmin_loc, vmax=vmax_loc)
cbar = plt.colorbar(extend='both')
plt.title('EWPC at (y, x)')

plt.tight_layout()
plt.show()

# ----------------------------------
# Region selection & log-polar transform
# ----------------------------------
y_start, y_end = 130, 256
x_start, x_end = 0, 256

STO_mean = func_limit_xy((y_start, y_end, x_start, x_end))  # experimental average
sim_STO_path = ''  # path to simulated cepstrum (np.load)
STO_sim = np.load(sim_STO_path)  # shape: (H, W)

radius = STO_sim.shape[0] // 2
cut_radius_in = 45  # ignore central region for matching

plt.figure()
plt.imshow(STO_mean)
plt.title('EWPC mean (selected region)')
plt.show()

STO_exp_logpolar = warp_polar(STO_mean, radius=radius, scaling='log')
STO_sim_logpolar = warp_polar(STO_sim, radius=radius, scaling='log')

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(np.any(STO_exp_logpolar == 0, axis=0))
plt.title('Zero columns (exp log-polar)')

plt.subplot(3, 1, 2)
plt.plot(np.mean(STO_exp_logpolar, axis=0))
plt.title('Mean over angle (exp log-polar)')

plt.subplot(3, 1, 3)
plt.plot(np.std(STO_exp_logpolar, axis=0))
plt.title('Std over angle (exp log-polar)')
plt.tight_layout()
plt.show()

def plt_settings(title):
    plt.colorbar(label='Intensity (a.u.)')
    plt.xlabel('Radial distance (px)')
    plt.ylabel('Angle (deg)')
    plt.title(title)

plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.imshow(STO_exp_logpolar, aspect=0.2)
plt_settings('Experiment')

plt.subplot(2, 2, 2)
plt.imshow(STO_sim_logpolar, aspect=0.2)
plt_settings('Simulation')

# determine outer cut where exp data becomes zero after warp
cut_radius_out = int(np.min(np.where(np.any(STO_exp_logpolar == 0, axis=0))[0]))

mask_exp = np.zeros_like(STO_exp_logpolar, dtype=bool)
mask_sim = np.zeros_like(STO_sim_logpolar, dtype=bool)
mask_exp[:, cut_radius_in:cut_radius_out] = True
mask_sim[:, cut_radius_in:] = True

plt.subplot(2, 2, 3)
plt.imshow(STO_exp_logpolar * mask_exp, aspect=0.2)
plt_settings('Experiment (mask)')

plt.subplot(2, 2, 4)
plt.imshow(STO_sim_logpolar * mask_sim, aspect=0.2)
plt_settings('Simulation (mask)')

plt.tight_layout()
plt.savefig(savepath + '_STO_logpolar.png')
plt.show()

# ----------------------------------
# Registration in log-polar (scale + rotation)
# ----------------------------------
# Robust unpack for skimage versions that return (shift, error, diffphase)
_shift_tuple = phase_cross_correlation(
    STO_sim_logpolar,
    STO_exp_logpolar,
    upsample_factor=10,
    normalization=None,
    reference_mask=mask_sim,
    moving_mask=mask_exp
)
shiftr, shiftc = _shift_tuple[0]  # row (angle), col (log-radius)

klog = radius / np.log(radius)
shift_scale = 1 / (np.exp(shiftc / klog))  # exp/sim scale ratio in real units

# keep original angle mapping behavior
angle = np.minimum(shiftr, shiftr - 180)

print(f'Recovered value for cc rotation: {shiftr}')
print(f'Recovered value for scaling difference: {shift_scale}')

# ----------------------------------
# Save calibration results
# ----------------------------------
pd_data = pd.DataFrame(
    [y_start, y_end, x_start, x_end, angle, shift_scale, pix_size_cepst, pix_size_cepst / shift_scale],
    index=[
        'y_start(px)', 'y_end(px)', 'x_start(px)', 'x_end(px)',
        'estimated-rot-angle(deg)', 'ratio(exp/sim)',
        'pix_size_sim(nm/px)', 'pix_size_exp(nm/px)'
    ]
)
pd_data.to_csv(savepath + '_STO-calibration.csv')

# extents for visualization (nm)
extent_sim = np.array([
    -STO_sim.shape[1] / 2, STO_sim.shape[1] / 2,
     STO_sim.shape[0] / 2, -STO_sim.shape[0] / 2
]) * pix_size_cepst

extent_exp = np.array([
    -STO_mean.shape[1] / 2, STO_mean.shape[1] / 2,
     STO_mean.shape[0] / 2, -STO_mean.shape[0] / 2
]) * (pix_size_cepst / shift_scale)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(STO_mean, extent=extent_exp)
plt.xlim(-0.7, 0.7)
plt.ylim(0.7, -0.7)
plt.title('Experiment (extent)')

plt.subplot(1, 2, 2)
plt.imshow(STO_sim, extent=extent_sim)
plt.xlim(-0.7, 0.7)
plt.ylim(0.7, -0.7)
plt.title('Simulation (extent)')
plt.tight_layout()
plt.show()

# ----------------------------------
# Align simulation to experiment (scale + rotate around center)
# ----------------------------------
def transform_from_center(image, scale_factor=1.0, rotation_angle=0):
    center_y = int(np.ceil(image.shape[0] / 2))
    center_x = int(np.ceil(image.shape[1] / 2))
    rotation_rad = np.deg2rad(rotation_angle)

    tform = transform.AffineTransform(
        scale=(scale_factor, scale_factor),
        rotation=rotation_rad,
        translation=(
            -center_x * (scale_factor * np.cos(rotation_rad) - 1)
            + center_y * scale_factor * np.sin(rotation_rad),
            -center_y * (scale_factor * np.cos(rotation_rad) - 1)
            - center_x * scale_factor * np.sin(rotation_rad)
        )
    )
    return transform.warp(image, tform.inverse, mode='constant', cval=0)

def p_norm(img, pmin=1, pmax=99):
    """Percentile normalization with central disk masked out."""
    hw = 7
    center = int(np.ceil(img.shape[0] / 2))
    img_ = img.copy()
    img_[center - hw:center + hw + 1, center - hw:center + hw + 1] = np.nan
    v = img_[~np.isnan(img_)]
    vmin = np.percentile(v, pmin)
    vmax = np.percentile(v, pmax)
    return (img - vmin) / (vmax - vmin + 1e-12)

STO_sim_rescale = transform_from_center(STO_sim.astype('float64'), shift_scale, -angle)

crop_sim = (STO_sim.shape[0] - STO_mean.shape[0]) // 2
crop_sim_rescale = (STO_sim_rescale.shape[0] - STO_mean.shape[0]) // 2

def plt_settings2(title):
    plt.colorbar(label='Normalized intensity')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

STO_exp_crop = p_norm(STO_mean)
STO_sim_crop = p_norm(STO_sim[crop_sim:-crop_sim, crop_sim:-crop_sim])
STO_sim_crop_resize = p_norm(STO_sim_rescale[crop_sim_rescale:-crop_sim_rescale,
                                             crop_sim_rescale:-crop_sim_rescale])

plt.figure(figsize=(11, 6))
plt.subplot(2, 3, 1)
plt.imshow(STO_exp_crop)
plt_settings2('Experiment')

plt.subplot(2, 3, 2)
plt.imshow(STO_sim_crop)
plt_settings2('Simulation')

plt.subplot(2, 3, 3)
plt.imshow(STO_exp_crop - STO_sim_crop)
plt_settings2('Exp. - Sim.')

plt.subplot(2, 3, 4)
plt.imshow(STO_exp_crop)
plt_settings2('Experiment')

plt.subplot(2, 3, 5)
plt.imshow(STO_sim_crop_resize)
plt_settings2('Simulation (resize)')

plt.subplot(2, 3, 6)
plt.imshow(STO_exp_crop - STO_sim_crop_resize)
plt_settings2('Exp. - Sim. (resize)')

plt.tight_layout()
plt.savefig(savepath + '_calibration_fig.png')
plt.show()

# ----------------------------------
# Save aligned/cropped arrays
# ----------------------------------
np.save(savepath + '_STO_exp.npy', STO_exp_crop)
np.save(savepath + '_STO_sim_crop.npy', STO_sim_crop)
np.save(savepath + '_STO_sim_resize_crop.npy', STO_sim_crop_resize)
