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

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mymodule as mym  # external helper module used for annotations/scale bars

# ----------------------------------
# Settings
# ----------------------------------
cmap = 'cmo.deep_r'
plt.rcParams['image.cmap'] = cmap

pix_size = 0.37994 * 4  # nm per pixel in the DF/vADF image

folder = ''  # base folder
subfolder = 'ewpc_tol10_bg1.0e+01_os512_crop97'
savefolder = os.path.join(folder, subfolder) + os.sep

path_ewpc_data = os.path.join(savefolder, subfolder + '.npy')
path_vADF = os.path.join(savefolder, 'df_img.npy')
path_calib_csv = os.path.join(savefolder, subfolder + '_STO-calibration.csv')
path_simlist = os.path.join(savefolder, 'CC_sim_paths_mirror.txt')

# ----------------------------------
# Load calibration (column名に依存しない取得)
# ----------------------------------
pd_data = pd.read_csv(path_calib_csv, index_col=0)

def _get_val(idx: str) -> float:
    series = pd_data.loc[idx]
    return float(series.values[0])  # 先頭列の値を採用（列名が '0' でなくてもOK）

shift_scale = _get_val('ratio(exp/sim)')
pix_size_cepst_sim = _get_val('pix_size_sim(nm/px)')
pix_size_cepst_exp = _get_val('pix_size_exp(nm/px)')

# ----------------------------------
# Load simulation label list（文字コードフォールバック & ベース名抽出）
# ----------------------------------
def _read_lines_with_fallback(path):
    for enc in ('shift-jis', 'cp932', 'utf-8'):
        try:
            with open(path, 'r', encoding=enc) as f:
                return [line.strip() for line in f.readlines()]
        except Exception:
            continue
    # 最後の手段
    with open(path, 'rb') as f:
        return [line.decode(errors='ignore').strip() for line in f.readlines()]

txt_sim_raw = _read_lines_with_fallback(path_simlist)
txt_sim = [os.path.basename(s) if ('/' in s or '\\' in s) else s for s in txt_sim_raw]

# ----------------------------------
# Load maps
# ----------------------------------
crosscorr_map = np.load(os.path.join(savefolder, 'CCresult_mirror.npy'))  # (n_sims, 11, Y, X)
vADF_image_ = np.load(path_vADF)
# vADFをCCマップのYに合わせて切り詰め（元コード準拠）
vADF_image = vADF_image_[:crosscorr_map.shape[2], :]

# ----------------------------------
# Figure annotations (labels/positions)
# ----------------------------------
xy_A = [0.80, 0.62]
xy_B = [0.43, 0.63]
xy_C = [0.07, 0.50]
ABC = ['A', 'B', 'C']
xy_ABC = [xy_A, xy_B, xy_C]
c_ABC = ['w', 'w', 'w']
ew_ABC = [None, None, None]

xy_STO = [1, 0.01]
xy_SRO = [1, 0.15]
xy_LCO = [1, 0.40]
xy_FIB = [1, 0.75]
xy_Vac = [1, 0.90]
materials = ['Substrate', 'Current collector', 'Cathode', 'Protection layers', 'Vacuum']
xy_materials = [xy_STO, xy_SRO, xy_LCO, xy_FIB, xy_Vac]
c_materials = ['w', 'w', 'w', 'w', 'w']
ew_materials = [None, None, None, None, None]

# ----------------------------------
# Drawing helpers (mymodule をそのまま利用)
# ----------------------------------
def image_settings(
    bool_ABC=True,
    materials=materials,
    scalebar_pos=(0.02, 0.02),
    c_scale='w',
    ew_scale=None,
    ew_ABC=ew_ABC,
    ew_materials=ew_materials,
    c_materials=c_materials,
    c_ABC=c_ABC,
    sep=1.5,
    fontsize_scale=16,
):
    ec_scale = 'w' if c_scale == 'k' else 'k'
    mym.scalebar2(
        pix_size, c=c_scale, bbox_to_anchor=scalebar_pos,
        max_length_ratio_to_width=0.15, sep=sep, ec=ec_scale,
        ew=ew_scale, fontsize=fontsize_scale
    )

    if bool_ABC:
        for i, label in enumerate(ABC):
            xy = xy_ABC[i]
            c = c_ABC[i]
            ec = 'w' if c == 'k' else 'k'
            d_x_arrow, d_y_arrow, d_y_text = -0.03, 0.005, 0.07
            ow = 0 if ew_ABC[i] is None else ew_ABC[i]

            mym.add_text_to_image(
                label, xy[0] + ow / 500, xy[1] + ow / 500,
                color=c, fontsize=16, pad=0, ew=ew_ABC[i], ec=ec
            )

            mym.annotate_simple_curved_arrow(
                plt.gca(), " ",
                (xy[0] + d_x_arrow, xy[1] + d_y_arrow),
                (xy[0] + d_x_arrow, xy[1] + d_y_text),
                xycoords='axes fraction', textcoords='axes fraction',
                mode='straight', rad=0., head_length=3, head_width=4, tail_width=1.2,
                relpos=(0., 0.), outline_width=ow, facecolor=c, outline_color=ec,
                mutation_scale=2, ha='left', fontweight='normal', fontsize=16, shrinkA=0
            )

    for j, name in enumerate(materials):
        if name is None:
            continue
        xy = xy_materials[j]
        c = c_materials[j]
        ec = 'w' if c == 'k' else 'k'
        mym.add_text_to_image(
            name, xy[0], xy[1], ec=ec, color=c, fontsize=14,
            ha='right', va='bottom', pad=3, x_pad=5, ew=ew_materials[j]
        )
    plt.axis('off')

def plt_settings(cbar_title, location='left', ticks=None):
    plt.colorbar(label=cbar_title, shrink=0.8, pad=0.02, location=location, ticks=ticks)

# ----------------------------------
# Example marker on DF image
# ----------------------------------
pos_x = [75, 190]  # (y, x)
plt.figure(figsize=(8, 4))
plt.imshow(
    vADF_image, cmap='gray',
    vmin=np.percentile(vADF_image, 5),
    vmax=np.percentile(vADF_image, 99.95)
)
plt.scatter(pos_x[1], pos_x[0], marker='+', c='red', s=150, lw=3)
image_settings()
plt.show()

# ----------------------------------
# Display C_max and θ_rot maps
# ----------------------------------
ind = 8  # simulation index to visualize

pmin, pmax = 75, 99.9
vmin = np.percentile(crosscorr_map[ind, 0, :, :], pmin)
vmax = np.percentile(crosscorr_map[ind, 0, :, :], pmax)

if ind == 0:
    fig = plt.figure(figsize=(9, 8))
    fig.subplots_adjust(hspace=0.05)

    plt.subplot(211)
    plt.imshow(crosscorr_map[ind, 0, :, :], vmin=vmin, vmax=vmax)
    image_settings(c_materials=['w', 'w', 'k', 'w', 'w'])
    plt_settings(r'$\mathit{C}_{max}$ (a.u.)', location='right')
    mym.add_text_to_image('a', 0, 1, ha='left', va='top', weight='semibold', fontsize=24, pad=6)

    plt.subplot(212)
    plt.imshow(crosscorr_map[ind, 1, :, :], cmap='twilight', interpolation='none')
    image_settings(
        ew_scale=None, c_scale='w',
        c_materials=['w', 'k', 'k', 'k', 'w'],
        ew_ABC=[4, 4, 4], c_ABC=['k', 'k', 'k']
    )
    plt_settings(r'$\mathit{θ}_{rot}$ (°)', location='right', ticks=[0, 30, 60, 90, 120, 150])
    mym.add_text_to_image('b', 0, 1, ha='left', va='top', weight='semibold', fontsize=24, pad=6, ew=4, ec='k')
    plt.show()

fig = plt.figure(figsize=(9, 16))
fig.subplots_adjust(hspace=0.05)

# 1) Cmax raw
plt.subplot(411)
plt.imshow(crosscorr_map[ind, 0, :, :], vmin=0)
plt_settings(r'$\mathit{C}_{max}$ (a.u.)')
image_settings()

# 2) Histogram with percentile lines
ax_hist = plt.subplot(412)
vals = crosscorr_map[ind, 0, :, :].ravel()
plt.hist(vals, bins=100, histtype='stepfilled', facecolor='skyblue')
plt.axvline(vmin, ls='--', color='k'); plt.axvline(vmax, ls='-', color='dimgray')

ymin, ymax = ax_hist.get_ylim()
ypos = ymax * 0.5
ax_hist.annotate(f'{pmin:.1f}th-percentile', (vmin, ypos), rotation=90, va='center', ha='left',
                 color='k', xytext=(2, 0), textcoords='offset points')
ax_hist.annotate(f'{pmax:.1f}th-percentile', (vmax, ypos), rotation=90, va='center', ha='left',
                 color='dimgray', xytext=(2, 0), textcoords='offset points')
plt.xlabel(r'$\mathit{C}_{max}$ (a.u.)'); plt.ylabel('Count')

# shrink the histogram box slightly (元コードのレイアウト調整を踏襲)
pos = ax_hist.get_position()
new_width, new_height = pos.width * 0.79, pos.height * 0.75
dx, dy = 0.025, 0.008
new_x0 = pos.x0 + (pos.width - new_width) / 2 + dx
new_y0 = pos.y0 + (pos.height - new_height) / 2 + dy
ax_hist.set_position([new_x0, new_y0, new_width, new_height])

# 3) Cmax clipped to [vmin, vmax]
plt.subplot(413)
plt.imshow(crosscorr_map[ind, 0, :, :], vmin=vmin, vmax=vmax)
image_settings()
plt_settings(r'$\mathit{C}_{max}$ (a.u.)')

# 4) θ_rot map
plt.subplot(414)
plt.imshow(crosscorr_map[ind, 1, :, :], cmap='twilight', interpolation='none')
image_settings(
    ew_scale=3, c_scale='k',
    c_materials=['k', 'k', 'k', 'k', 'k'],
    ew_materials=[3, 3, 3, 3, 3],
    ew_ABC=[4, 4, 4], c_ABC=['k', 'k', 'k']
)
plt_settings(r'$\mathit{θ}_{rot}$ (°)', ticks=[0, 30, 60, 90, 120, 150])
plt.show()

# ----------------------------------
# Winner index map & std map (across simulations)
# ----------------------------------
plt.figure()
plt.subplot(121); plt.imshow(crosscorr_map[:, 0, :, :].argmax(axis=0)); plt.title('Winner index (Cmax)')
plt.subplot(122); plt.imshow(crosscorr_map[:, 0, :, :].std(axis=0));    plt.title('Std across sims (Cmax)')
plt.tight_layout(); plt.show()

# Masked LCO map example (winner==0)
LCO_image = crosscorr_map[0, 0, :, :].copy()
LCO_image[(crosscorr_map[:, 0, :, :].argmax(axis=0) != 0)] = 0

plt.figure(); plt.imshow(LCO_image, vmin=0.3); plt.title('LCO Cmax masked'); plt.show()

# ----------------------------------
# RGB composite (3 phases)
# ----------------------------------
def p_norm(img, pmin=pmin, pmax=pmax):
    vmin = np.percentile(img, pmin); vmax = np.percentile(img, pmax)
    ret = (img - vmin) / max(vmax - vmin, 1e-12)
    ret = np.clip(ret, 0, 1)
    return ret, vmin, vmax

LCO_corr, LCO_vmin, LCO_vmax = p_norm(crosscorr_map[0, 0, :, :], 75, 99.9)
Co3O4_corr, Co3O4_vmin, Co3O4_vmax = p_norm(crosscorr_map[2, 0, :, :], 75, 99.9)
CoO_corr, CoO_vmin, CoO_vmax = p_norm(crosscorr_map[3, 0, :, :], 75, 99.9)

RGB_image = np.dstack((Co3O4_corr, CoO_corr, LCO_corr))

materials2 = [None, None, None, None, None]
plt.figure(figsize=(8, 4))
plt.imshow(RGB_image)
image_settings(sep=3, bool_ABC=True, materials=materials2)
plt.title('RGB composite (R: spinel, G: rock salt, B: layered)')
plt.show()

# ----------------------------------
# Colorbars for each channel scale
# ----------------------------------
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

cmap_r = LinearSegmentedColormap.from_list('rbar', ['black', (1, 0, 0)], N=256)
cmap_g = LinearSegmentedColormap.from_list('gbar', ['black', (0, 1, 0)], N=256)
cmap_b = LinearSegmentedColormap.from_list('bbar', ['black', (0, 0, 1)], N=256)

norm_r = Normalize(vmin=Co3O4_vmin, vmax=Co3O4_vmax)
norm_g = Normalize(vmin=CoO_vmin,   vmax=CoO_vmax)
norm_b = Normalize(vmin=LCO_vmin,   vmax=LCO_vmax)

sm_r = ScalarMappable(norm=norm_r, cmap=cmap_r); sm_r.set_array([])
sm_g = ScalarMappable(norm=norm_g, cmap=cmap_g); sm_g.set_array([])
sm_b = ScalarMappable(norm=norm_b, cmap=cmap_b); sm_b.set_array([])

fig = plt.figure(figsize=(2.0, 2.5))
gs = GridSpec(nrows=3, ncols=1, height_ratios=[1, 1, 1], hspace=5, figure=fig)

cax_r = fig.add_subplot(gs[2, 0])
cax_g = fig.add_subplot(gs[1, 0])
cax_b = fig.add_subplot(gs[0, 0])

orient = 'horizontal'
cb_r = fig.colorbar(sm_r, cax=cax_r, orientation=orient); cb_r.set_label(r'$\mathit{C}_{max}$ for spinel (a.u.)')
cb_g = fig.colorbar(sm_g, cax=cax_g, orientation=orient); cb_g.set_label(r'$\mathit{C}_{max}$ for rock salt (a.u.)')
cb_b = fig.colorbar(sm_b, cax=cax_b, orientation=orient); cb_b.set_label(r'$\mathit{C}_{max}$ for layered (a.u.)')

plt.tight_layout()
plt.show()
