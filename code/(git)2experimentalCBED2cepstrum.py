"""
Cepstrum Matching of SNED
2. Converting experimental SNED data to cepstrum
---------------------------------------------------------
This script accompanies the manuscript under review.

Disclaimer:
This code is provided as-is, without warranty of any kind. See DISCLAIMER.md.

Notes:
- Colormaps from cmocean are used (cmo.deep_r, cmo.thermal); if unavailable,
  replace with standard matplotlib colormaps.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import rotate as rotate2

# ----------------------------------
# Settings
# ----------------------------------
cmap = 'cmo.deep_r'
plt.rcParams['image.cmap'] = cmap

#%%
folder = ''
file = ''

data4d = np.load(folder + file)['arr_0']  # shape: (ky, kx, y, x)

ky_shape, kx_shape, y_shape, x_shape = data4d.shape

def create_rotational_symmetric_hann_window(shape):
    """Create a rotationally symmetric Hann-like window on a unit disk."""
    if len(shape) != 2:
        raise ValueError("Shape must be 2-dimensional")
    rows, cols = shape
    y = np.linspace(-1, 1, rows)[:, np.newaxis]
    x = np.linspace(-1, 1, cols)[np.newaxis, :]
    r = np.clip(np.sqrt(x**2 + y**2), 0, 1)
    return 0.5 * (1 - np.cos(np.pi * (1 - r)))

# window is defined on (kx, ky) as in original
window_shape = (kx_shape, ky_shape)
hann2D = create_rotational_symmetric_hann_window(window_shape)
print('finished')

#%%
# Scan-grid indices
x2 = np.linspace(0, x_shape - 1, x_shape)
y2 = np.linspace(0, y_shape - 1, y_shape)
X2, Y2 = np.meshgrid(x2, y2)
x3 = X2.astype(int).ravel()
y3 = Y2.astype(int).ravel()

# Example pixel to inspect
y_sli = 180
x_sli = 132

data4d_sli = data4d[:, :, y_sli, x_sli].astype(np.float64)

df_img = np.load(folder + 'df_img.npy')

plt.figure()
plt.imshow(df_img, cmap='gray')
plt.scatter(x_sli, y_sli, marker='+', c='m')
plt.axis('off')

plt.figure()
plt.imshow(data4d_sli)

#%% CBED rotation & oversampling confirmation
angle = 51
hw_crop_ewpc = 48
tol_cbed = 100
bg_log = tol_cbed

oversample_size = 512
oversample_array = np.zeros((oversample_size, oversample_size))
# (re)use hann2D computed above

# floor at tol, rotate, log
data4d_sli_ = data4d_sli.copy()
data4d_sli_[data4d_sli_ < tol_cbed] = tol_cbed
data4d_sli_rot = rotate2(np.flipud(data4d_sli_), angle, reshape=False, order=1, cval=tol_cbed)

log_data4d_sli = np.log(data4d_sli_rot - tol_cbed + bg_log)

# weighted-mean subtraction & windowing
wmean = np.average(log_data4d_sli, weights=hann2D)
fft_target_ = (log_data4d_sli - wmean) * hann2D

# oversample (zero-pad)
fft_target = np.zeros_like(oversample_array)
fft_target[:ky_shape, :kx_shape] = fft_target_
ewpc = np.abs(np.fft.fftshift(np.fft.fft2(fft_target)))
ewpc_crop = ewpc[
    oversample_size // 2 - hw_crop_ewpc : oversample_size // 2 + hw_crop_ewpc + 1,
    oversample_size // 2 - hw_crop_ewpc : oversample_size // 2 + hw_crop_ewpc + 1,
]

plt.figure(figsize=(10, 5))
plt.subplot(231); plt.imshow(data4d_sli);      plt.axis('off'); plt.title('SNED');           plt.colorbar()
plt.subplot(232); plt.imshow(data4d_sli_rot);  plt.axis('off'); plt.title('Rotated');        plt.colorbar()
plt.subplot(233); plt.imshow(log_data4d_sli);  plt.axis('off'); plt.title('Log(SNED + bg)'); plt.colorbar()
plt.subplot(234); plt.imshow(fft_target_);     plt.axis('off'); plt.title('Windowed');       plt.colorbar()
plt.subplot(235); plt.imshow(ewpc_crop);       plt.axhline(32); plt.axhline(37); plt.axis('off'); plt.title('Cepstrum'); plt.colorbar()
plt.subplot(236); plt.plot(ewpc_crop[33:37, :].mean(axis=0)); plt.title('EWPC profile')

#%% Oversampling for entire dataset
def ewpc_oversample(savepath, hw_crop_ewpc, tol_cbed, oversample_size):
    """Compute oversampled cepstrum crops for each scan pixel and save."""
    ret_data = np.zeros((hw_crop_ewpc * 2 + 1, hw_crop_ewpc * 2 + 1, y_shape, x_shape))
    oversample_array = np.zeros((oversample_size, oversample_size))
    hann2D_local = create_rotational_symmetric_hann_window((kx_shape, ky_shape))

    for i in tqdm(range(len(x3))):
        xs = x3[i]; ys = y3[i]
        cbed = data4d[:, :, ys, xs].astype(np.float64)
        cbed[cbed < tol_cbed] = tol_cbed

        cbed_rot = rotate2(np.flipud(cbed), angle, reshape=False, order=1, cval=tol_cbed)
        log_cbed = np.log(cbed_rot)  # tol_cbedで下駄を履かせているのでlog(0)は回避

        fft_block = log_cbed * hann2D_local
        canvas = np.zeros_like(oversample_array)
        canvas[:ky_shape, :kx_shape] = fft_block

        ewpc_full = np.abs(np.fft.fftshift(np.fft.fft2(canvas)))
        crop = ewpc_full[
            oversample_size // 2 - hw_crop_ewpc : oversample_size // 2 + hw_crop_ewpc + 1,
            oversample_size // 2 - hw_crop_ewpc : oversample_size // 2 + hw_crop_ewpc + 1,
        ]
        ret_data[:, :, ys, xs] = crop

    np.save(savepath, ret_data)
    return ret_data

#%%
tol_cbed = 10
oversample_size = 512
hw_crop_ewpc = 48

path_ewpc_os_data = folder + 'ewpc_tol%i_os%i_crop%i_2.npy' % (
    tol_cbed, oversample_size, hw_crop_ewpc * 2 + 1
)

ewpc_os_data = ewpc_oversample(path_ewpc_os_data, hw_crop_ewpc, tol_cbed, oversample_size)
