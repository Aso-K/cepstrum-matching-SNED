"""
Cepstrum Matching of SNED
5. Show the matching result as maps
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
import mymodule as mym
import pandas as pd

cmap = 'cmo.deep_r'
plt.rcParams['image.cmap'] = cmap

pix_size = 0.37994*4

folder = ''

subfolder = 'ewps_tol10_bg1.0e+01_os512_crop97'

savefolder = folder + subfolder + '/'

path_ewpc_data = savefolder + subfolder + '.npy'
path_vADF = savefolder + 'df_img.npy'

pd_data = pd.read_csv(savefolder + subfolder + '_STO-calibration.csv',
                      index_col=0)
txt_sim_ = pd.read_table(folder+subfolder+'/CC_sim_paths_mirror.txt', encoding="shift-jis", header=None)
txt_sim = list()

for s in range(0,len(txt_sim_)):
    string = str(txt_sim_.iat[s,0])
    if '/' in string:
        string = string.split('/')[-1]
        print('remove non-neccesary part')
    txt_sim.append(string)

shift_scale = pd_data.loc['ratio(exp/sim)','0']
pix_size_cepst_sim = pd_data.loc['pix_size_sim(nm/px)','0']
pix_size_cepst_exp = pd_data.loc['pix_size_exp(nm/px)','0']

crosscorr_map = np.load(savefolder+'CCresult_mirror.npy')

vADF_image_ = np.load(path_vADF)
vADF_image = vADF_image_[:crosscorr_map.shape[2],:]

xy_A = [0.80, 0.62]
xy_B = [0.43,0.63]
xy_C = [0.07,0.5]
ABC = ['A', 'B', 'C']
xy_ABC = [xy_A, xy_B, xy_C]
c_ABC = ['w','w','w']
ew_ABC = [None,None,None]

xy_STO = [1,0.01]
xy_SRO = [1,0.15]
xy_LCO = [1,0.4]
xy_FIB = [1,0.75]
xy_Vac = [1,0.9]
materials = ['Substrate','Current collector', 'Cathode', 'Protection layers', 'Vacuum']
xy_materials = [xy_STO, xy_SRO, xy_LCO, xy_FIB, xy_Vac]
c_materials = ['w','w','w','w','w']
ew_materials = [None,None,None,None,None]



def image_settings(bool_ABC=True,
                   materials=materials,
                   scalebar_pos=(0.02,0.02), 
                   c_scale='w',
                   ew_scale=None, 
                   ew_ABC=ew_ABC, 
                   ew_materials=ew_materials, 
                   c_materials=c_materials,
                   c_ABC=c_ABC,
                   sep=1.5,
                   fontsize_scale=16):
    if c_scale=='k':
        ec_scale = 'w'
    if c_scale=='w':
        ec_scale = 'k'
    mym.scalebar2(pix_size, c=c_scale, bbox_to_anchor=scalebar_pos, 
                 max_length_ratio_to_width=0.15, sep=sep, ec=ec_scale, 
                 ew=ew_scale,fontsize=fontsize_scale)
    
    if bool_ABC:
        # Settings for A, B, C
        for i in range(len(ABC)):
            xy = xy_ABC[i]
            c  = c_ABC[i]
            if c=='k':
                ec = 'w'
            if c=='w':
                ec = 'k'
                
            d_x_arrow = -0.03
            d_y_arrow = 0.005
            d_y_text = 0.07
            
            if ew_ABC[i]==None:
                ow = 0
            else:
                ow = ew_ABC[i]
            
            mym.add_text_to_image(ABC[i],xy[0]+ow/500,xy[1]+ow/500, color=c, fontsize=16, 
                                  pad=0, ew=ew_ABC[i], ec=ec)
            

            
            mym.annotate_simple_curved_arrow(
                plt.gca(),
                " ",
                (xy[0]+d_x_arrow, xy[1]+d_y_arrow), 
                (xy[0]+d_x_arrow, xy[1]+d_y_text),
                xycoords='axes fraction',
                textcoords='axes fraction',
                mode='straight',
                # angleA=0,
                # angleB=90,
                rad=0.,
                head_length=3,
                head_width=4,
                tail_width=1.2,
                relpos=(0.,0.),
                outline_width=ow,
                facecolor=c,
                outline_color=ec,
                mutation_scale=2,
                ha='left',fontweight='normal',fontsize=16,
                shrinkA=0)
    
    # Settings for each material
    for j in range(len(materials)):
        xy = xy_materials[j]
        c  = c_materials[j]
        if c=='k':
            ec = 'w'
        if c=='w':
            ec = 'k'
        mym.add_text_to_image(materials[j],xy[0],xy[1], ec=ec, color=c,
                              fontsize=14, 
                              ha='right', va='bottom', pad=3,x_pad=5,
                              ew=ew_materials[j])
    plt.axis('off')
    
    
    
    return


# See expc_analysis_ ... _20250701_single-test
pos_x = [75,190] # LCO


plt.figure(figsize=(8,4))
plt.imshow(vADF_image, cmap='gray', 
           vmin=np.percentile(vADF_image, 5), 
           vmax=np.percentile(vADF_image, 99.95))
plt.scatter(pos_x[1], pos_x[0], marker='+', c='red', s=150, lw=3)
image_settings()
plt.show()
# plt.tight_layout()





#%%
ind = 8

pmin = 75
pmax = 99.9
vmin = np.percentile(crosscorr_map[ind,0,:,:],pmin)
vmax = np.percentile(crosscorr_map[ind,0,:,:],pmax)

def plt_settings(cbar_title, location='left', ticks=None):
    plt.colorbar(label=cbar_title, shrink=0.8, pad=0.02,
                 location=location, ticks=ticks)

if ind == 0:
    fig = plt.figure(figsize=(9,8))
    
    fig.subplots_adjust(
        # left=0.,    # Left margin of the figure
        # right=1,   # Right margin of the figure
        # top=1,     # Top margin of the figure
        # bottom=0,  # Bottom margin of the figure
        hspace=0.05,  # Vertical space between subplots
        # wspace=0.1   # Horizontal space between subplots
    )
    
    plt.subplot(211) 
    plt.imshow(crosscorr_map[ind,0,:,:], vmin=vmin, vmax=vmax)
    image_settings(
                    c_materials=['w','w','k','w','w'],
                   # ew_materials=[None, 3, None, None, None],
                   # c_scale='w'
                   )
    plt_settings(r'$\mathit{C}_{max}$ (a.u.)', location='right')
    mym.add_text_to_image('a', 0, 1, ha='left', va='top', weight='semibold', 
                          fontsize=24, pad=6)
    
    # --- 4th subplot (another imshow) ---
    plt.subplot(212)
    plt.imshow(crosscorr_map[ind,1,:,:], cmap='twilight', interpolation='none')
    image_settings(ew_scale=None,
                   c_scale='w',
                    c_materials=['w','k','k','k','w'],
                   # ew_materials=[None,3,None,3,None],
                   ew_ABC=[4,4,4],
                   c_ABC=['k','k','k'])
    plt_settings(r'$\mathit{θ}_{rot}$ (°)', location='right', ticks=[0,30,60,90,120,150])
    mym.add_text_to_image('b', 0, 1, ha='left', va='top', weight='semibold', 
                          fontsize=24, pad=6,
                          ew=4, ec='k')
    
    
    
    plt.show()


fig = plt.figure(figsize=(9,16))

# plt.tight_layout()
fig.subplots_adjust(
    # left=0.,    # Left margin of the figure
    # right=1,   # Right margin of the figure
    # top=1,     # Top margin of the figure
    # bottom=0,  # Bottom margin of the figure
    hspace=0.05,  # Vertical space between subplots
    # wspace=0.1   # Horizontal space between subplots
)


# --- 1st subplot (imshow) ---
plt.subplot(411)
plt.imshow(crosscorr_map[ind,0,:,:], vmin=0)
plt_settings(r'$\mathit{C}_{max}$ (a.u.)')
image_settings(
                # c_materials=['k','k','w','w','w'],
                # ew_materials=[None, 3, None, None, None],
                # c_scale='k'
               )

# --- 2nd subplot (histogram) ---
ax_hist = plt.subplot(412)
# --- Plot histogram ---
plt.hist(crosscorr_map[ind,0,:,:].flatten(), bins=100,
         facecolor='skyblue', 
         # edgecolor='navy',
         histtype='stepfilled',)

# --- Plot only threshold lines (no labels, only color and linestyle) ---
plt.axvline(vmin, ls='--', color='k')
plt.axvline(vmax, ls='-', color='dimgray')
# Get the upper end of the y-axis to decide the placement of the text
ymin, ymax = ax_hist.get_ylim()
ypos = ymax * 0.5  # Position 10% below the top

# Add vertical text near each threshold line
ax_hist.annotate(

    f'{pmin:.1f}th-percentile', 
    (vmin, ypos),
    rotation=90, 
    va='center',    # Align text center to the specified point
    ha='left',  # Display on the left side of the line
    color='k',
    xytext=(2, 0),                 # Shift 2pt to the right
    textcoords='offset points',      # Unit = points
)
ax_hist.annotate(
    f'{pmax:.1f}th-percentile', 
    (vmax, ypos),
    rotation=90, 
    va='center',
    ha='left',   # Display on the right side of the line
    color='dimgray',
    xytext=(2, 0),                 # Shift 2pt to the right
    textcoords='offset points',      # Unit = points
    
)

# --- Axis labels etc. ---
plt.xlabel(r'$\mathit{C}_{max}$ (a.u.)')
plt.ylabel('Count')
# ── Shrink the box ──
# Get current position
pos = ax_hist.get_position()
# Shrink: top/bottom 10% each ⇒ height 0.8×, left/right 20% each ⇒ width 0.6×
new_width  = pos.width  * 0.79
new_height = pos.height * 0.75
# Shift amounts
dx = 0.025  # shift +1% in x
dy = 0.008  # shift +2% in y
# Calculate offset for centering
new_x0 = pos.x0 + (pos.width  - new_width ) / 2 + dx
new_y0 = pos.y0 + (pos.height - new_height) / 2 + dy
# Apply new position
ax_hist.set_position([new_x0, new_y0, new_width, new_height])
# ── Shrink done ──
 
# --- 3rd subplot (imshow with vmin/vmax) ---
plt.subplot(413) 
plt.imshow(crosscorr_map[ind,0,:,:], vmin=vmin, vmax=vmax)
image_settings(
                # c_materials=['k','k','w','w','w'],
                # ew_materials=[None, 3, None, None, None],
                # c_scale='k'
               )
plt_settings(r'$\mathit{C}_{max}$ (a.u.)')

# --- 4th subplot (another imshow) ---
plt.subplot(414)
plt.imshow(crosscorr_map[ind,1,:,:], cmap='twilight', interpolation='none')
image_settings(ew_scale=3,
               c_scale='k',
                c_materials=['k','k','k','k','k'],
               ew_materials=[3,3,3,3,3],
               ew_ABC=[4,4,4],
               c_ABC=['k','k','k'])
plt_settings(r'$\mathit{θ}_{rot}$ (°)', ticks=[0,30,60,90,120,150])


plt.show()


#%%
plt.figure()
plt.subplot(121)
plt.imshow(crosscorr_map[:,0,:,:].argmax(axis=0))
plt.subplot(122)
plt.imshow(crosscorr_map[:,0,:,:].std(axis=0))
plt.show()

LCO_image = crosscorr_map[0,0,:,:].copy()
LCO_image[(crosscorr_map[:,0,:,:].argmax(axis=0)!=0)] = 0

plt.figure()
plt.imshow(LCO_image, vmin=0.3)
plt.show()


#%%
def p_norm(img,pmin=pmin,pmax=pmax):
    vmin = np.percentile(img, pmin)
    vmax = np.percentile(img, pmax)
    ret = (img - vmin)/(vmax - vmin)
    ret[ret>1] = 1
    ret[ret<0] = 0
    return ret, vmin, vmax

LCO_corr, LCO_vmin, LCO_vmax = p_norm(crosscorr_map[0,0,:,:], 75,99.9)
Co3O4_corr, Co3O4_vmin, Co3O4_vmax = p_norm(crosscorr_map[2,0,:,:], 75,99.9)
CoO_corr, CoO_vmin, CoO_vmax = p_norm(crosscorr_map[3,0,:,:], 75,99.9)

RGB_image = np.dstack((Co3O4_corr, CoO_corr, LCO_corr))


# materials2 = ['Substrate','Current collector', 'Cathode', None, None]
materials2 = [None, None, None, None, None]

plt.figure(figsize=(8,4))
plt.imshow(RGB_image)
image_settings(sep=3, bool_ABC=True, 
               materials = materials2
)
plt.show()


#%%
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec


# --- Black → pure color gradient (no position specification = stable) ---
cmap_r = LinearSegmentedColormap.from_list('rbar', ['black', (1,0,0)], N=256)
cmap_g = LinearSegmentedColormap.from_list('gbar', ['black', (0,1,0)], N=256)
cmap_b = LinearSegmentedColormap.from_list('bbar', ['black', (0,0,1)], N=256)

# --- Normalize each channel using original scale ---
norm_r = Normalize(vmin=Co3O4_vmin, vmax=Co3O4_vmax)
norm_g = Normalize(vmin=CoO_vmin,   vmax=CoO_vmax)
norm_b = Normalize(vmin=LCO_vmin,   vmax=LCO_vmax)

sm_r = ScalarMappable(norm=norm_r, cmap=cmap_r); sm_r.set_array([])
sm_g = ScalarMappable(norm=norm_g, cmap=cmap_g); sm_g.set_array([])
sm_b = ScalarMappable(norm=norm_b, cmap=cmap_b); sm_b.set_array([])

# --- Place three vertical colorbars on the right ---
fig = plt.figure(figsize=(2.0,2.5))
gs = GridSpec(nrows=3, ncols=1, height_ratios=[1,1,1], hspace=5, figure=fig)

# ax_img = fig.add_subplot(gs[0,0])
# ax_img.imshow(RGB_image)
# image_settings(sep=3, bool_ABC=True, materials=materials2)

cax_r = fig.add_subplot(gs[2,0])
cax_g = fig.add_subplot(gs[1,0])
cax_b = fig.add_subplot(gs[0,0])

orient = 'horizontal'

cb_r = fig.colorbar(sm_r, cax=cax_r, orientation=orient); cb_r.set_label('$\mathit{C}_{max}$ for spinel (a.u.)')
cb_g = fig.colorbar(sm_g, cax=cax_g, orientation=orient); cb_g.set_label('$\mathit{C}_{max}$ for rock salt (a.u.)')
cb_b = fig.colorbar(sm_b, cax=cax_b, orientation=orient); cb_b.set_label('$\mathit{C}_{max}$ for layered (a.u.)')

# for cb in (cb_r, cb_g, cb_b):
    # cb.ax.yaxis.set_major_locator(MaxNLocator(4))
    # cb.outline.set_visible(False)

plt.tight_layout()
plt.show()
