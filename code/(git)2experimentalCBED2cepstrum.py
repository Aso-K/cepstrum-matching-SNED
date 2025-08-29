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
from cupyx.scipy.ndimage import rotate as npx_rotate

cmap = 'cmo.deep_r'
plt.rnparams['image.cmap'] = cmap


#%%
folder = ''
file = ''

data4d = np.load(folder+file)['arr_0']
# offset = tff.imread(folder + 'offsetMap.tif')

ky_shape = data4d.shape[0]
kx_shape = data4d.shape[1]
y_shape = data4d.shape[2]
x_shape = data4d.shape[3]

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

window_shape = (kx_shape, ky_shape)
hann2D = create_rotational_symmetric_hann_window(window_shape)
print('finished')


#%%
x2 = np.linspace(0,x_shape-1,x_shape)
y2 = np.linspace(0,y_shape-1,y_shape)
X2,Y2 = np.meshgrid(x2,y2)
x3 = X2.flatten().astype(int)
y3 = Y2.flatten().astype(int)


y_sli_np = 180
x_sli_np = 132

data4d_sli_np = data4d[:, :, y_sli_np, x_sli_np].astype(np.float64)

df_img = np.load(folder + 'df_img.npy')

plt.figure()
plt.imshow(df_img, cmap='gray')
plt.scatter(x_sli_np, y_sli_np, marker='+', c='m')
plt.axis('off')

plt.figure()
plt.imshow(data4d_sli_np)

#%% background estimation in cbed
pad_from_edge = 10
data4d_sli_np_bg_ = data4d_sli_np.copy()
data4d_sli_np_bg_[pad_from_edge:y_shape-pad_from_edge,
                  pad_from_edge:x_shape-pad_from_edge] = np.nan

print(np.nanmean(data4d_sli_np_bg_))
print(np.nanstd(data4d_sli_np_bg_))

data4d_sli_np_bg_nandrop = data4d_sli_np_bg_[~np.isnan(data4d_sli_np_bg_)]

plt.figure()
plt.hist(data4d_sli_np_bg_nandrop, bins=100)
plt.yscale('log')


#%% CBED rotation & oversampling confirmation


# bg = np.float64(1e-15)
# bg = np.float64(np.log(data4d_sli_np.mean()))

angle = 51

hw_crop_ewps=48
# tol_log = 5
tol_cbed = 100
bg_log = tol_cbed

oversample_size = 512
oversample_array = np.zeros((oversample_size, oversample_size))
window_shape = (kx_shape, ky_shape)
hann2D = create_rotational_symmetric_hann_window(window_shape)

hann2D_gpu = np.asarray(hann2D)

data4d_sli_ = np.asarray(data4d_sli_np)
data4d_sli__ = data4d_sli_.copy()
data4d_sli__[data4d_sli__ < tol_cbed] = tol_cbed
data4d_sli = npx_rotate(np.flipud(data4d_sli__), angle, reshape=False, order=1, cval=tol_cbed)

log_data4d_sli =  np.log(data4d_sli - tol_cbed + bg_log)

# log_data4d_sli = log_data4d_sli_.copy()
# log_data4d_sli[log_data4d_sli<tol_log] = tol_log
wmean = np.average(log_data4d_sli, weights=hann2D)

fft_target_ =(log_data4d_sli-wmean)*hann2D_gpu



fft_target = np.zeros_like(oversample_array.copy())
fft_target[:ky_shape,:kx_shape] = fft_target_
ewps = np.abs(np.fft.fftshift(np.fft.fft2(fft_target)))
ewps_crop = ewps[oversample_size//2-hw_crop_ewps:oversample_size//2+hw_crop_ewps+1,
                  oversample_size//2-hw_crop_ewps:oversample_size//2+hw_crop_ewps+1]
ewps_crop_np = np.asnumpy(ewps_crop)

plt.figure(figsize=(10,5))
plt.subplot(231)
plt.imshow(np.asnumpy(data4d_sli__))
plt.axis('off')
plt.title('SNED')
plt.colorbar()

plt.subplot(232)
plt.imshow(np.asnumpy(data4d_sli))
plt.axis('off')
plt.title('Log(SNED + bg)')
plt.colorbar()

plt.subplot(233)
plt.imshow(np.asnumpy(log_data4d_sli))
plt.axis('off')
plt.title('Log(SNED + bg), plus')
plt.colorbar()

plt.subplot(234)
plt.imshow(np.asnumpy(fft_target_))
plt.axis('off')
plt.title('Windowed')
plt.colorbar()

plt.subplot(235)
plt.imshow(ewps_crop_np)
plt.axhline(32)
plt.axhline(37)
plt.axis('off')
plt.title('Cepstrum')
plt.colorbar()

plt.subplot(236)
plt.plot(ewps_crop_np[33:37,:].mean(axis=0))
# plt.axis('off')
# plt.title('EWPC')
# plt.colorbar()



#%% for glant application

plt.figure()
plt.imshow(df_img, cmap='gray')
plt.scatter(x_sli_np, y_sli_np, marker='+', c='m')
plt.axis('off')

display_crop = 52

plt.figure()
plt.imshow(rotate2(np.flipud(data4d_sli_np), angle,reshape=False), cmap='cmo.thermal',
           vmin=np.percentile(data4d_sli_np,1),
           vmax=np.percentile(data4d_sli_np,99.8))
plt.xlim(132-display_crop, 132+display_crop)
plt.ylim(132-display_crop, 132+display_crop)
plt.axis('off')

crop_cepst_size = ewps_crop_np.shape[0]

plt.figure()
plt.imshow(ewps_crop_np, vmin=np.percentile(ewps_crop_np,50),
           vmax=np.percentile(ewps_crop_np,99.2))
plt.scatter(crop_cepst_size//2,crop_cepst_size//2, zorder=1000, marker='x', c='k', s=230, lw=6)
plt.scatter(crop_cepst_size//2,crop_cepst_size//2, zorder=1200, marker='x', c='w', s=170, lw=3)
plt.scatter(crop_cepst_size//2-7,crop_cepst_size//2-32, 
            zorder=1000, marker='^', s=1600, lw=2, 
            facecolor='None', edgecolors='w')
plt.scatter(crop_cepst_size//2+9,crop_cepst_size//2-30, 
            zorder=1000, marker='o', s=1100, lw=2, 
            facecolor='None', edgecolors='w')
plt.axis('off')


#%%

def ewps_gpu_oversample(savepath, hw_crop_ewps, tol_cbed, oversample_size):
    ret_data_gpu = np.zeros((hw_crop_ewps*2+1, hw_crop_ewps*2+1, y_shape, x_shape))

    x3_gpu = np.asarray(x3)
    y3_gpu = np.asarray(y3)
    # data4d_gpu = np.asarray(data4d)
    # ret_data_gpu = np.asarray(ret_data)
    
    
    oversample_array = np.zeros((oversample_size, oversample_size))
    
    window_shape = (kx_shape, ky_shape)
    hann2D = create_rotational_symmetric_hann_window(window_shape)
    
    hann2D_gpu = np.asarray(hann2D)
    # bg = np.float64(1e-15)
    # bg = np.float64(int(data4d.mean()))

    for i in tqdm(range(len(x3_gpu))):
        x_sli = x3_gpu[i]
        y_sli = y3_gpu[i]
        x_sli_np = x3[i]
        y_sli_np = y3[i]
        
        # slice data
        data4d_sli_np = data4d[:, :, y_sli_np, x_sli_np].astype(np.float64)
        
        # from np to np
        data4d_sli_ = np.asarray(data4d_sli_np)
        
        # remove background
        data4d_sli__ = data4d_sli_.copy()
        data4d_sli__[data4d_sli__ < tol_cbed] = tol_cbed
        
        # rotate
        data4d_sli = npx_rotate(np.flipud(data4d_sli__), 
                                angle, 
                                reshape=False, 
                                order=1, 
                                cval=tol_cbed)
        # log
        log_data4d_sli =  np.log(data4d_sli)
        
        # # A) weighted mean subtraction
        # wmean = np.average(log_data4d_sli, weights=hann2D)
        # fft_target_ = (log_data4d_sli - wmean)*hann2D_gpu
        
        # B) NO subtraction
        fft_target_ = log_data4d_sli*hann2D_gpu
        
        # oversampling
        fft_target = np.zeros_like(oversample_array.copy())
        fft_target[:ky_shape,:kx_shape] = fft_target_
        
        ewps = np.abs(np.fft.fftshift(np.fft.fft2(fft_target)))
        ewps_crop = ewps[oversample_size//2-hw_crop_ewps:oversample_size//2+hw_crop_ewps+1,
                         oversample_size//2-hw_crop_ewps:oversample_size//2+hw_crop_ewps+1]
        ret_data_gpu[:,:, y_sli, x_sli] = ewps_crop
    
    
    ret_data = np.asnumpy(ret_data_gpu)
    np.save(savepath, ret_data)
    return ret_data


#%%
tol_cbed = 10
oversample_size = 512
hw_crop_ewps=48


path_ewps_os_data = folder + 'ewps_tol%i_os%i_crop%i_2.npy'%(tol_cbed,
                                                           oversample_size,
                                                           hw_crop_ewps*2+1)
# bool_ewps_os_data = os.path.isfile(path_ewps_os_data)
bool_ewps_os_data = False

if bool_ewps_os_data:
    ewps_os_data = np.load(path_ewps_os_data)
else:
    ewps_os_data = ewps_gpu_oversample(path_ewps_os_data, 
                                       hw_crop_ewps, 
                                       tol_cbed, 
                                       oversample_size)


