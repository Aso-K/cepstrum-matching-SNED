"""
Cepstrum Matching of SNED
4. Cepstrum matching of experiment and simulation
---------------------------------------------------------
This script accompanies the manuscript under review.

Disclaimer:
This code is provided as-is, without warranty of any kind. See DISCLAIMER.md.

Notes:
- Colormaps from cmocean are used (cmo.deep_r, cmo.thermal); if unavailable,
  replace with standard matplotlib colormaps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp_polar, rescale
import cv2
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

# ----------------------------------
# Settings
# ----------------------------------
cmap = 'cmo.deep_r'
plt.rcParams['image.cmap'] = cmap

folder = ''  # base folder
subfolder = 'ewpc_tol10_bg1.0e+01_os512_crop97'
savefolder = os.path.join(folder, subfolder) + os.sep

path_ewpc_data = os.path.join(savefolder, subfolder + '.npy')
path_df = os.path.join(savefolder, 'df_img.npy')
path_calib_csv = os.path.join(savefolder, subfolder + '_STO-calibration.csv')

# ----------------------------------
# Load calibration (robust to column names)
# ----------------------------------
pd_data = pd.read_csv(path_calib_csv, index_col=0)
# robust extraction (works whether the column label is '0' or something else)
def _get_val(idx):
    series = pd_data.loc[idx]
    # take the first value regardless of column name
    return float(series.values[0])

shift_scale = _get_val('ratio(exp/sim)')
pix_size_cepst_sim = _get_val('pix_size_sim(nm/px)')
pix_size_cepst_exp = _get_val('pix_size_exp(nm/px)')

# ----------------------------------
# Load data
# ----------------------------------
ewpc_data = np.load(path_ewpc_data)  # (H, W, Y, X)
df_img = np.load(path_df)

# ----------------------------------
# Extents (not strictly required for CC; kept for visualization consistency)
# ----------------------------------
sim_size = 153
extent_sim = np.array([-sim_size/2, sim_size/2, sim_size/2, -sim_size/2]) * pix_size_cepst_sim

exp_size = int(subfolder.split('_crop')[-1])
extent_exp = np.array([-exp_size/2, exp_size/2, exp_size/2, -exp_size/2]) * pix_size_cepst_exp

# ----------------------------------
# Simulation inputs
# ----------------------------------
sim_files = [
    'LiCoO2 O3 (I) 10-110nm',
    'Co3O4 10-110nm',
    'CoO (NaCl) 10-110nm',
    'SrTiO3 [-110]',
]
sim_folder = ''  # folder where *_crop-cepstrum.npy are stored

sim_arrays_list = []
sim_paths = []

for title in sim_files:
    sim_path = os.path.join(sim_folder, title + '_crop-cepstrum.npy')
    sim_arr = np.load(sim_path)
    plt.figure(); plt.imshow(sim_arr); plt.title(title)
    sim_arrays_list.append(sim_arr)
    sim_paths.append(title)

    # mirror variant for LiCoO2 (as in original)
    if title == 'LiCoO2 O3 (I) 10-110nm':
        mirror_arr = np.fliplr(sim_arr)
        plt.figure(); plt.imshow(mirror_arr); plt.title(title + ' (mirror)')
        sim_arrays_list.append(mirror_arr)
        sim_paths.append(title + ' (mirror, fliplr)')

sim_arrays = np.asarray(sim_arrays_list)
np.savetxt(os.path.join(savefolder, 'CC_sim_paths_mirror.txt'), sim_paths, fmt='%s')

# ----------------------------------
# Cross-correlation in log-polar
# ----------------------------------
cut_inner_polar = 10  # inner cutoff (avoid center)

def polar_crosscorr(ewpc_exp_polar, ewpc_sim):
    """
    Compute angle-dependent normalized cross-correlation between
    experimental and simulated cepstra in log-polar domain, after
    scaling the simulated radius by `shift_scale`.
    Returns 11 summary statistics:
        [CCmax, angle_at_CCmax, CCmean, CCstd, CCmedian,
         CCmean_peak, CCstd_peak, CCmedian_peak,
         CCmean_bg, CCstd_bg, CCmedian_bg]
    """
    # log-polar transform of simulation, then rescale radius to match exp/sim ratio
    ewpc_sim_lp = warp_polar(ewpc_sim, radius=ewpc_sim.shape[1] // 2)
    ewpc_sim_lp = rescale(ewpc_sim_lp, (1, shift_scale), anti_aliasing=True)

    # radial extent match
    cut_outer_polar = min(ewpc_sim_lp.shape[1], ewpc_exp_polar.shape[1])
    sim_crop = ewpc_sim_lp[:, cut_inner_polar:cut_outer_polar]
    exp_crop = ewpc_exp_polar[:, cut_inner_polar:cut_outer_polar]

    # template match (OpenCV expects float32)
    # use up to 181 deg in template to avoid wrap overlap artifacts
    CCprofile = cv2.matchTemplate(
        sim_crop.astype(np.float32),
        exp_crop[:181, :].astype(np.float32),
        cv2.TM_CCOEFF_NORMED
    )[:, 0]

    x_profile = np.arange(len(CCprofile))

    # enforce 360-deg periodicity by tiling and interpolating
    CC_period = np.tile(CCprofile, 3)
    x_period = np.hstack((x_profile - 180, x_profile, x_profile + 180))

    x_fit = np.arange(stop=len(CCprofile), step=0.1)
    fit = interp1d(x_period, CC_period, kind='cubic')
    CC_fit = fit(x_fit)

    CCmax = float(np.max(CC_fit))
    angle_at_CCmax = float(np.mean(x_fit[CC_fit == CCmax]))
    CCmean = float(np.mean(CCprofile))
    CCstd = float(np.std(CCprofile))
    CCmedian = float(np.median(CCprofile))
    CC_interp_results = [CCmax, angle_at_CCmax, CCmean, CCstd, CCmedian]

    # Peak vs background statistics around the best angle (±x_range, periodic)
    x_range = 45
    cond_peak = (np.abs(x_profile - angle_at_CCmax) < x_range) | \
                (180 - np.abs(x_profile - angle_at_CCmax) < x_range)
    CCmean_peak = float(np.mean(CCprofile[cond_peak]))
    CCstd_peak = float(np.std(CCprofile[cond_peak]))
    CCmedian_peak = float(np.median(CCprofile[cond_peak]))
    CC_peak_results = [CCmean_peak, CCstd_peak, CCmedian_peak]

    cond_bg = ~cond_peak
    CCmean_bg = float(np.mean(CCprofile[cond_bg]))
    CCstd_bg = float(np.std(CCprofile[cond_bg]))
    CCmedian_bg = float(np.median(CCprofile[cond_bg]))
    CC_bg_results = [CCmean_bg, CCstd_bg, CCmedian_bg]

    return *CC_interp_results, *CC_peak_results, *CC_bg_results

# ----------------------------------
# Evaluate over a scan region
# ----------------------------------
y_start, y_end = 0, 140
x_start, x_end = 0, 256

yy = np.arange(y_start, y_end)
xx = np.arange(x_start, x_end)
X, Y = np.meshgrid(xx, yy)
pos_y_arr = Y.ravel()
pos_x_arr = X.ravel()

# ret_array shape: (n_sims, 11 stats, Y, X)
ret_array = np.zeros((sim_arrays.shape[0], 11, y_end - y_start, x_end - x_start), dtype=np.float32)

for j in tqdm(range(len(pos_y_arr))):
    pos_y = int(pos_y_arr[j])
    pos_x = int(pos_x_arr[j])

    # experimental slice and its log-polar image
    exp_slice = ewpc_data[:, :, pos_y, pos_x]
    exp_lp = warp_polar(exp_slice, radius=exp_slice.shape[1] // 2)

    for k in range(sim_arrays.shape[0]):
        ret_array[k, :, pos_y - y_start, pos_x - x_start] = polar_crosscorr(exp_lp, sim_arrays[k, :, :])

np.save(os.path.join(savefolder, 'CCresult_mirror.npy'), ret_array)

# ----------------------------------
# Quick visualization of summary maps & histograms (first simulation index)
# ----------------------------------
ind = 0

plt.figure(figsize=(8, 15))
for i in range(6):
    cm = 'twilight' if i == 1 else None
    plt.subplot(6, 2, 2 * i + 1)
    plt.imshow(ret_array[ind, i, :, :], cmap=cm, interpolation='none')
    plt.title(f'Stat {i}')
    plt.subplot(6, 2, 2 * i + 2)
    plt.hist(ret_array[ind, i, :, :].ravel(), bins=100)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 10))
for i in range(4):
    data = ret_array[ind, i + 6, :, :]
    plt.subplot(4, 2, 2 * i + 1)
    plt.imshow(data, interpolation='none')
    plt.title(f'Stat {i+6}')
    plt.subplot(4, 2, 2 * i + 2)
    plt.hist(data.ravel(), bins=100)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 10))
for i in range(4):
    data = ret_array[ind, i + 10, :, :]
    plt.subplot(4, 2, 2 * i + 1)
    plt.imshow(data, interpolation='none')
    plt.title(f'Stat {i+10}')
    plt.subplot(4, 2, 2 * i + 2)
    plt.hist(data.ravel(), bins=100)
plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(ret_array[ind, 0, :, :])
plt.title('CCmax map (sim index 0)')
plt.show()
