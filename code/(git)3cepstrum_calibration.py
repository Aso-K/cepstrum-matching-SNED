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

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp_polar
from skimage.registration import phase_cross_correlation
import pandas as pd
from skimage import transform
import os
import shutil

cmap = 'cmo.deep_r'
plt.rcParams['image.cmap'] = cmap


folder = ''

file = 'ewps_tol10_bg1.0e+01_os512_crop97.npy'

oversample_size = int(file.split('_os')[-1].split('_crop')[0])

pix_size = 0.07974688*2
pix_size_cepst = 1/(pix_size)/oversample_size

savefile = file.split('.npy')[0]
save_folder = folder + savefile
savepath = save_folder + '/' + savefile

path_ewpc_data = save_folder+'/'+file

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
# move ewpc data
if not os.path.exists(path_ewpc_data):
    shutil.move(folder+file, path_ewpc_data)
    
path_df = save_folder + '/df_img.npy'
# move df img
if not os.path.exists(path_df):
    shutil.move(folder + '/df_img.npy', path_df)
    
    
#%%
ewpc_data = np.load(path_ewpc_data)



ewpc_img_sum = ewpc_data[64,64,:,:]/ewpc_data.sum(axis=(0,1))
ewpc_sum = ewpc_data.sum(axis=(2,3))

df_img = np.load(path_df)
ewpc_max = ewpc_data.max(axis=(2,3))
ewpc_std = np.std(ewpc_data,axis=(2,3))



def func_limit_xy(limit_in_xy):
    ewpc_lim_in_xy_ = ewpc_data[:,
                                :,
                                limit_in_xy[0]:limit_in_xy[1],
                                limit_in_xy[2]:limit_in_xy[3]
                                ].mean(axis=(2,3))
    return ewpc_lim_in_xy_


##%%
# pos_x = [54,170] # CoO
pos_x = [80,170]
# pos_x = [5,10]
# pos_x = [60,100]

ewpc_extract = ewpc_data[:,:,pos_x[0],pos_x[1],]

plt.figure(figsize=(8,2))
plt.subplot(131)
plt.imshow(ewpc_sum, 
           vmin=np.percentile(ewpc_sum, 10),
           vmax=np.percentile(ewpc_sum, 99),
           cmap=cmap)

plt.subplot(132)
plt.imshow(df_img, cmap='gray')
plt.plot(pos_x[1],pos_x[0],'+',c='magenta')

plt.subplot(133)
plt.imshow(ewpc_extract,
           cmap=cmap,
           vmin=np.percentile(ewpc_data[:,:,pos_x[0],pos_x[1],], 10),
           vmax=np.percentile(ewpc_data[:,:,pos_x[0],pos_x[1],], 99))
cbar = plt.colorbar(extend='both')
cbar.ax.yaxis.offsetText.set(va="bottom", ha="center")
cbar.ax.yaxis.OFFSETTEXTPAD = 3
# register_bottom_offset(cbar.ax.yaxis, bottom_offset)
# cbar.update_ticks()

plt.show()


#%%
y_start = 130
y_end   = 256
x_start = 0
x_end   = 256

STO_mean = func_limit_xy((y_start,y_end,x_start,x_end))
sim_STO_path = ''
STO_sim = np.load(sim_STO_path)

radius = STO_sim.shape[0]//2

cut_radius_in = 45


plt.figure()
plt.imshow(STO_mean)

STO_exp_logpolar = warp_polar(STO_mean, radius=radius, 
                              scaling='log')


STO_sim_logpolar = warp_polar(STO_sim, radius=radius, 
                              scaling='log')


plt.figure()
plt.subplot(311)
plt.plot(np.any(STO_exp_logpolar==0, axis=0))

plt.subplot(312)
plt.plot(np.mean(STO_exp_logpolar, axis=0))

plt.subplot(313)
plt.plot(np.std(STO_exp_logpolar, axis=0))

plt.show()

def plt_settings(title):
    plt.colorbar(label='Intensity (a.u.)')
    plt.xlabel('Radial distance (px)')
    plt.ylabel('Angle (deg)')
    plt.title(title)



plt.figure(figsize=(8,6))
plt.subplot(221)
plt.imshow(STO_exp_logpolar, aspect=0.2)
plt_settings('Experiment')

plt.subplot(222)
plt.imshow(STO_sim_logpolar, aspect=0.2)
plt_settings('Simulation')

cut_radius_out = np.min(np.where(np.any(STO_exp_logpolar==0, axis=0))[0])


mask_exp = np.zeros_like(STO_exp_logpolar, dtype=bool)
mask_sim = np.zeros_like(STO_exp_logpolar, dtype=bool)
mask_exp[:,cut_radius_in:cut_radius_out] = True
mask_sim[:,cut_radius_in:] = True


plt.subplot(223)
plt.imshow(STO_exp_logpolar*mask_exp,aspect=0.2)
plt_settings('Experiment (mask)')


plt.subplot(224)
plt.imshow(STO_sim_logpolar*mask_sim, aspect=0.2)
plt_settings('Simulation (mask)')

plt.tight_layout()
plt.savefig(savepath+'_STO_logpolar.png')

plt.show()



shifts = phase_cross_correlation(
    STO_sim_logpolar, STO_exp_logpolar, 
    upsample_factor=10, 
    normalization=None,
    reference_mask = mask_sim, moving_mask=mask_exp
)
shiftr, shiftc = shifts
klog = radius / np.log(radius)
shift_scale = 1 / (np.exp(shiftc / klog))

angle = np.minimum(shiftr, shiftr-180)

print(f'Recovered value for cc rotation: {shiftr}')
print()
print(f'Recovered value for scaling difference: {shift_scale}')


pd_data = pd.DataFrame([y_start, 
                        y_end, 
                        x_start, 
                        x_end,
                        angle,
                        shift_scale,
                        pix_size_cepst,
                        pix_size_cepst/shift_scale],
                       index=['y_start(px)',
                                'y_end(px)',
                                'x_start(px)',
                                'x_end(px)',
                                'estimated-rot-angle(deg)',
                                'ratio(exp/sim)',
                                'pix_size_sim(nm/px)',
                                'pix_size_exp(nm/px)'])
pd_data.to_csv(savepath+'_STO-calibration.csv')

extent_sim = np.array([-STO_sim.shape[1]/2,
                        STO_sim.shape[1]/2,
                        STO_sim.shape[0]/2,
                       -STO_sim.shape[0]/2])*pix_size_cepst

extent_exp = np.array([-STO_mean.shape[1]/2,
                        STO_mean.shape[1]/2,
                        STO_mean.shape[0]/2,
                       -STO_mean.shape[0]/2])*pix_size_cepst/shift_scale

plt.figure()
plt.subplot(121)
plt.imshow(STO_mean, extent=extent_exp)
plt.xlim(-0.7, 0.7)
plt.ylim( 0.7,-0.7)
plt.subplot(122)
plt.imshow(STO_sim, extent=extent_sim)
plt.xlim(-0.7, 0.7)
plt.ylim( 0.7,-0.7)
plt.show()

#%%
def transform_from_center(image, scale_factor=1.0, rotation_angle=0):

    center_y = int(np.ceil(image.shape[0]/2))
    center_x = int(np.ceil(image.shape[1]/2))
    
    rotation_rad = np.deg2rad(rotation_angle)
    
    transform_matrix = transform.AffineTransform(
        scale=(scale_factor, scale_factor),
        rotation=rotation_rad,
        translation=(
            -center_x * (scale_factor * np.cos(rotation_rad) - 1) 
            + center_y * scale_factor * np.sin(rotation_rad),
            -center_y * (scale_factor * np.cos(rotation_rad) - 1) 
            - center_x * scale_factor * np.sin(rotation_rad)
        )
    )
    
    result = transform.warp(
        image, 
        transform_matrix.inverse,
        mode='constant',  
        cval=0  
    )
    
    return result


def p_norm(img,pmin=1,pmax=99):
    hw = 7
    center = int(np.ceil(img.shape[0]/2))
    img_ = img.copy()
    img_[center-hw:center+hw+1,center-hw:center+hw+1] = np.nan
    img__ = img_[~np.isnan(img_)]
    vmin = np.percentile(img__, pmin)
    vmax = np.percentile(img__, pmax)
    return (img-vmin)/(vmax-vmin)

# STO_mean_rescale = rescale(STO_mean.astype('float64'), 1/shift_scale)
STO_sim_rescale = transform_from_center(STO_sim.astype('float64'), shift_scale, -angle)
crop_sim = (STO_sim.shape[0] - STO_mean.shape[0])//2
crop_sim_rescale = (STO_sim_rescale.shape[0] - STO_mean.shape[0])//2

def plt_settings2(title):
    plt.colorbar(label='Normalized intensity')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    
STO_exp_crop = p_norm(STO_mean)
STO_sim_crop = p_norm(STO_sim[crop_sim:-crop_sim,crop_sim:-crop_sim])
STO_sim_crop_resize = p_norm(STO_sim_rescale[crop_sim_rescale:-crop_sim_rescale,crop_sim_rescale:-crop_sim_rescale])

plt.figure(figsize=(11,6))
plt.subplot(231)
plt.imshow(STO_exp_crop)
plt_settings2('Experiment')

plt.subplot(232)
plt.imshow(STO_sim_crop)
plt_settings2('Simulation')

plt.subplot(233)
plt.imshow(STO_exp_crop - STO_sim_crop)
plt_settings2('Exp. - Sim.')

plt.subplot(234)
plt.imshow(STO_exp_crop)
plt_settings2('Experiment')


plt.subplot(235)
plt.imshow(STO_sim_crop_resize)
plt_settings2('Simulation (resize)')


plt.subplot(236)
plt.imshow(STO_exp_crop - STO_sim_crop_resize)
plt_settings2('Exp. - Sim.(resize)')


plt.tight_layout()
plt.savefig(savepath+'_calibration_fig.png')

plt.show()

np.save(savepath+'_STO_exp.npy', STO_exp_crop)
np.save(savepath+'_STO_sim_crop.npy', STO_sim_crop)
np.save(savepath+'_STO_sim_resize_crop.npy', STO_sim_crop_resize)


