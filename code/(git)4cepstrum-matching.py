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

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp_polar
import cv2
from skimage.transform import rescale
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm


cmap = 'cmo.deep_r'
plt.rcParams['image.cmap'] = cmap


folder = ''

subfolder = 'ewps_tol10_bg1.0e+01_os512_crop97'

savefolder = folder + subfolder + '/'

path_ewpc_data = savefolder + subfolder + '.npy'
path_df = savefolder + 'df_img.npy'

pd_data = pd.read_csv(savefolder + subfolder + '_STO-calibration.csv',
                      index_col=0)

shift_scale = pd_data.loc['ratio(exp/sim)','0']
pix_size_cepst_sim = pd_data.loc['pix_size_sim(nm/px)','0']
pix_size_cepst_exp = pd_data.loc['pix_size_exp(nm/px)','0']




#%%
ewpc_data = np.load(path_ewpc_data)
df_img = np.load(path_df)

#%%
sim_size = 153
extent_sim = np.array([-sim_size/2,
                        sim_size/2,
                        sim_size/2,
                       -sim_size/2])*pix_size_cepst_sim

exp_size = int(subfolder.split('_crop')[-1])
extent_exp = np.array([-exp_size/2,
                        exp_size/2,
                        exp_size/2,
                       -exp_size/2])*pix_size_cepst_exp


#%%
sim_arrays_list = list()
sim_files = ['LiCoO2 O3 (I) 10-110nm',
             'Co3O4 10-110nm',
             'CoO (NaCl) 10-110nm',
             'SrTiO3 [-110]'
             ]
sim_paths = list()
    
sim_folder = ''

for i in range(0,len(sim_files)):
    sim_path = sim_folder + sim_files[i] + '_crop-cepstrum.npy'
    title = sim_files[i]
    
    np_sim_cepst = np.load(sim_path)
    
    plt.figure()
    plt.imshow(np_sim_cepst)
    plt.title(title)
    
    sim_arrays_list.append(np_sim_cepst)
    sim_paths.append(title)
    if title == 'LiCoO2 O3 (I) 10-110nm':
        sim_arrays_list.append(np.fliplr(np.load(sim_path))) #mirror
        sim_paths.append(sim_path + '(mirror, fliplr)')
        
        plt.figure()
        plt.imshow(np.fliplr(np_sim_cepst))
        plt.title(sim_files[i]+' (mirror)')

sim_arrays = np.array(sim_arrays_list)
np.savetxt(savefolder+'CC_sim_paths_mirror.txt', sim_paths, fmt='%s')


#%%
cut_inner_polar = 10
crop_sim_CC_ = 10



def polar_crosscorr(ewpc_exp_polar, ewpc_sim):
    # def kde_estimate(CCprofile):
    #     kde_model = gaussian_kde(CCprofile, 
    #                              # bw_method='silverman'
    #                              )
    #     x_grid = np.linspace(CCprofile.min(), CCprofile.max(), 1000)
    #     y = kde_model(x_grid)
    #     mode = x_grid[y==y.max()][0]
    #     return mode
    
    ewpc_sim_polar = rescale(warp_polar(ewpc_sim, radius=ewpc_sim.shape[1]//2), (1,shift_scale))
    
    cut_outer_polar = np.minimum(ewpc_sim_polar.shape[1], 
                                 ewpc_exp_polar.shape[1])
    
    ewpc_sim_polar_crop = ewpc_sim_polar[:,cut_inner_polar:cut_outer_polar]
    ewpc_exp_polar_crop = ewpc_exp_polar[:,cut_inner_polar:cut_outer_polar]
    
    CCprofile = cv2.matchTemplate(ewpc_sim_polar_crop.astype(np.float32), 
                             ewpc_exp_polar_crop[:181,:].astype(np.float32),
                             cv2.TM_CCOEFF_NORMED)[:,0]
    
    x_profile = np.arange(len(CCprofile))

    CCprofile_period = np.tile(CCprofile, 3)
    x_profile_period = np.hstack((x_profile-180,x_profile,x_profile+180))
    
    x_profile_fit = np.arange(stop=len(CCprofile),step=0.1)
    fit = interp1d(x_profile_period, CCprofile_period, kind='cubic')
    CCprofile_fit = fit(x_profile_fit)
    
    CCmax = np.max(CCprofile_fit)
    angle_at_CCmax = np.mean(x_profile_fit[CCprofile_fit==CCmax])
    CCmean = np.mean(CCprofile)
    CCstd = np.std(CCprofile)
    # CCmin = np.min(CCprofile_fit)
    CCmedian = np.median(CCprofile)
    # CCmode = kde_estimate(CCprofile)
    CC_interp_results = [CCmax, angle_at_CCmax, CCmean, CCstd, CCmedian]
    
    x_range = 45
    cond_peak = ((np.abs(x_profile-angle_at_CCmax)<x_range)|
                 (180 - np.abs(x_profile-angle_at_CCmax)<x_range))
    CCmean_peak = np.mean(CCprofile[cond_peak])
    CCstd_peak = np.std(CCprofile[cond_peak])
    CCmedian_peak = np.median(CCprofile[cond_peak])
    # CCmode_peak = kde_estimate(CCprofile[cond_peak])
    CC_peak_results = [CCmean_peak, CCstd_peak, CCmedian_peak]
    
    cond_bg = ~cond_peak
    CCmean_bg = np.mean(CCprofile[cond_bg])
    CCstd_bg = np.std(CCprofile[cond_bg])
    CCmedian_bg = np.median(CCprofile[cond_bg])
    # CCmode_bg = kde_estimate(CCprofile[cond_bg])
    CC_bg_results = [CCmean_bg, CCstd_bg, CCmedian_bg]
    
    return *CC_interp_results, *CC_peak_results, *CC_bg_results


#%%
y_start = 0
y_end   = 140
x_start = 0
x_end   = 256

yy = np.arange(y_start, y_end)
xx = np.arange(x_start, x_end)
X,Y = np.meshgrid(xx,yy)
pos_y_arr = Y.flatten()
pos_x_arr = X.flatten()

ret_array = np.zeros((len(sim_arrays), 11, y_end-y_start, x_end-x_start))

for j in tqdm(range(0,len(pos_y_arr))):
    pos_y = pos_y_arr[j]
    pos_x = pos_x_arr[j]
    slice_ewpc_data = ewpc_data[:,:,pos_y,pos_x]
    ewpc_exp_polar = warp_polar(slice_ewpc_data, radius=slice_ewpc_data.shape[1]//2)
    for k in range(0,sim_arrays.shape[0]):
        ret_array[k,:,pos_y,pos_x] = polar_crosscorr(ewpc_exp_polar, sim_arrays[k,:,:])

np.save(savefolder+'CCresult_mirror.npy', ret_array)


#%%
ind = 0

plt.figure(figsize=(8,15))
for i in range(0,6):
    cmap = None
    if i == 1:
        cmap='twilight'
    plt.subplot(6,2,2*i+1)
    plt.imshow(ret_array[ind,i,:,:], cmap=cmap, interpolation='none')
    plt.subplot(6,2,2*i+2)
    h = plt.hist(ret_array[ind,i,:,:].flatten(), bins=100)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,10))
for i in range(0,4):
    cmap = None
    data = ret_array[ind,i+6,:,:]
    plt.subplot(4,2,2*i+1)
    plt.imshow(data, cmap=cmap, interpolation='none')
    plt.subplot(4,2,2*i+2)
    h = plt.hist(data.flatten(), bins=100)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,10))
for i in range(0,4):
    cmap = None
    data = ret_array[ind,i+10,:,:]
    plt.subplot(4,2,2*i+1)
    plt.imshow(data, cmap=cmap, interpolation='none')
    plt.subplot(4,2,2*i+2)
    h = plt.hist(data.flatten(), bins=100)
plt.tight_layout()
plt.show()


plt.figure()
plt.imshow(ret_array[ind,0,:,:]
           )
