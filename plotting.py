#!/usr/bin/env python3

import sys
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import scipy as sp
from skimage import morphology
import matplotlib as mpl
from matplotlib import (patheffects, colors)
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import aplpy
import radio_beam
from astropy.io import fits
from astropy import units as u
from astropy import (convolution, coordinates, wcs)

sys.path.append('/lustre/aoc/users/bsvoboda/temp/nestfit')
from nestfit import plotting as nfplot
from nestfit.main import (HdfStore, take_by_components)

from analyze import (VELOS, get_cubestack)


warnings.simplefilter('ignore')  # NOTE stop that matplotlib deprecation warning
plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', direction='out')
plt.rc('ytick', direction='out')


CLR_CMAP = plt.cm.Spectral_r
CLR_CMAP.set_bad('0.5', 1.0)
HOT_CMAP = plt.cm.afmhot
HOT_CMAP.set_bad('0.5', 1.0)
RDB_CMAP = plt.cm.RdBu
RDB_CMAP.set_bad('0.5', 1.0)
VIR_CMAP = plt.cm.viridis
VIR_CMAP.set_bad('0.5', 1.0)

_cmap_list = [(0.5, 0.5, 0.5, 1.0)] + [plt.cm.plasma(i) for i in range(plt.cm.plasma.N)]
NBD_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
        'Discrete Plasma', _cmap_list, len(_cmap_list),
)
NBD_CMAP.set_bad('0.2')

ROOT_PATH = Path('/lustre/aoc/users/bsvoboda/temp/keystone_nestfit')
DATA_PATH = ROOT_PATH / 'data'
PLOT_PATH = ROOT_PATH / 'plots'
RUN_PATH  = ROOT_PATH / 'run'


##############################################################################
#                       NestFit plotting routines
##############################################################################

def store_to_plotter(store, target):
    pad = nfplot.PaddingConfig(
            edge_pads=(0.7, 0.8, 0.6, 0.3),
            sep_pads=(0.2, 0.2),
            cbar_width=0.15,
    )
    plot_dir = str(PLOT_PATH / target)
    spl = nfplot.StorePlotter(store, plot_dir=plot_dir, pad=pad)
    return spl


def plot_test_pix(store, stack, target):
    spl = store_to_plotter(store, target)
    pix_map = {'M17': (99, 167), 'MonR2': (154, 52)}
    pix = pix_map[target]
    nfplot.plot_amm_specfit(spl, stack, pix, n_model=2, zoom=True)
    nfplot.plot_amm_post_stack(spl, pix, n_model=2)
    nfplot.plot_amm_spec_grid(spl, stack, pix, (10, 10))


def make_all_plots(store_prefix, target):
    store = HdfStore(str(RUN_PATH / f'{store_prefix}_{target}'))
    stack = get_cubestack(target)
    spl = store_to_plotter(store, target)
    nfplot.plot_nbest(spl)
    nfplot.plot_conv_nbest(spl)
    nfplot.plot_deblend_peak(spl)
    nfplot.plot_deblend_intintens(spl)
    nfplot.plot_ncomp_metrics(spl)
    nfplot.plot_map_props(spl)
    for conv in (True, False):
        nfplot.plot_evdiff(spl, conv=conv)
        nfplot.plot_quan_props(spl, conv=conv)
        nfplot.plot_err_props(spl, conv=conv)
    plot_test_pix(store, stack, target)
    store.close()


##############################################################################
#                          IRDC based routines
##############################################################################

def open_fits(filen, relative=True):
    path = Path(filen).expanduser()
    if relative:
        return fits.open(str(DATA_PATH / path))
    else:
        return fits.open(str(path))


def default_mask(hdul):
    data = deepcopy(hdul[0].data)
    mask = np.nan_to_num(data) > 0
    mask = morphology.remove_small_objects(mask, min_size=64)
    kernel = morphology.disk(1)
    mask = morphology.opening(mask.squeeze(), kernel)
    return mask


def get_map_center(hdul):
    shape = hdul[0].data.shape
    im_wcs = wcs.WCS(hdul[0].header)
    decim_deg = im_wcs.all_pix2world(shape[1]/2, shape[2]/2, 0, 0)
    ra, dec = decim_deg[0].item(), decim_deg[1].item()
    co = coordinates.SkyCoord(ra, dec, frame='fk5', unit=(u.deg, u.deg))
    return co


def convert_jy_to_k(hdul, mol):
    beam = radio_beam.Beam.from_fits_header(hdul[0].header)
    freq = RESTFREQS[mol] * u.GHz
    conv = (1 * u.Jy).to(u.K, u.brightness_temperature(beam, freq))
    hdul[0].data = hdul[0].data * conv.value
    return conv


def add_label(label, xy, fontsize=10):
    txt = plt.annotate(label, xy=xy, xycoords='axes fraction',
        fontsize=fontsize)
    txt.set_path_effects([patheffects.withStroke(linewidth=4.5,
        foreground='w')])
    return txt


def save_figure(filen, do_eps=True):
    exts = ['png', 'pdf']
    if do_eps:
        exts.append('eps')
    for ext in exts:
        path = Path('plots') / f'{filen}.{ext}'
        plt.savefig(str(path), dpi=300)
        print('-- {0} saved'.format(ext))
    plt.close('all'); plt.cla(); plt.clf()


def fix_label_ticks(gc, co, radius=1.5/60):
    gc.recenter(co.fk5.ra.value, co.fk5.dec.value, radius=radius)
    gc.axis_labels.set_ypad(-7)
    gc.ticks.set_color('black')
    gc.tick_labels.set_xformat('hh:mm:ss')
    gc.tick_labels.set_yformat('dd:mm:ss')
    gc.axis_labels.set_font(size='small')
    gc.tick_labels.set_font(size='x-small')


def add_beam(gc, corner='top left', facecolor='black',
        edgecolor='none'):
    gc.add_beam()
    gc.beam.set_corner(corner)
    gc.beam.set_color(facecolor)
    gc.beam.set_edgecolor(edgecolor)
    gc.beam.set_linewidth(1.0)


def add_colorbar(gc, box, label=None, top=False):
    gc.add_colorbar(box=box, pad=0.2, width=0.1, axis_label_text=label)
    gc.colorbar.set_font(size='x-small')
    gc.colorbar._colorbar.ax.tick_params(direction='in')
    if top:
        gc.colorbar.set_location('top')
        gc.colorbar.set_width(0.1)
        gc.colorbar.set_pad(0.12)
        gc.colorbar._colorbar.ax.tick_params(direction='in')
        gc.colorbar._colorbar.ax.xaxis.set_ticks_position('top')


def add_primary_beam_contour(gc, hdul, power=0.205):
    gc.show_contour(hdul, dimensions=[0,1], slices=[100,0], levels=[power],
            linewidths=0.5, linestyles='dashed', colors='0.25',
            convention='calabretta')


def zero_out_corners(hdu):
    hdu[0].data[ 0, 0] = 0
    hdu[0].data[ 0,-1] = 0
    hdu[0].data[-1, 0] = 0
    hdu[0].data[-1,-1] = 0


def plot_amm_property(store, par_id=0):
    pass


def plot_four_moments(source, mol, matched=True):
    # FIXME due to issues with having 3 axes from STOKES/RA/DEC, `aplpy==2.0.1`
    # does not work and must currently fall-back to `aplpy==1.1.1` that does
    # not wrap the new WCSAxes module in astropy
    filen = 'moments/{0}/{0}_{1}'.format(source, mol)
    img0 = open_fits(filen + '_snr2_smooth0.integrated.pbcor.fits')
    pb = open_fits('images/{0}/{0}_{1}_joint.pb.fits'.format(source, mol))
    if matched:
        img1 = open_fits('images/{0}/{0}_{1}_jfeather.image.mf0.fits'.format(source, mol))
        pbdata = pb[0].data[0,0,:,:].reshape(img1[0].data.shape)
        img1[0].data = img1[0].data / pbdata
    else:
        img1 = open_fits(filen + '_snr2_smooth0.maximum.fits')
    img2 = open_fits(filen + '_snr3_smooth0.weighted_coord.fits')
    img3 = open_fits(filen + '_snr3_smooth0.weighted_dispersion_coord.fits')
    convert_jy_to_k(img0, mol)
    convert_jy_to_k(img1, mol)
    for hdul in (img0, img1, img2, img3):
        data = hdul[0].data
        mask = default_mask(hdul).reshape(data.shape)
        data[~mask] = np.nan
        zero_out_corners(hdul)
    for hdul in (img0, img1):
        data = hdul[0].data
        data[~np.isfinite(data)] = 0.0
    fig = plt.figure(figsize=(8, 6.5))
    split_size = 0.02
    x_margin_size = 0.15
    y_margin_size = 0.075
    delta = 0.06
    x0 = x_margin_size - delta
    y0 = y_margin_size
    dx = (1 - 2 * x_margin_size - split_size) / 2
    dy = (1 - 2 * y_margin_size - split_size) / 2
    x1 = x0 + dx + split_size + 1.5 * delta
    y1 = y0 + dy + split_size + 0.015
    labelx = 0.035
    labely = 0.035
    xlabel = r'$\alpha \ (\mathrm{J2000})$'
    ylabel = r'$\delta \ (\mathrm{J2000})$'
    #alma = open_fits('other_data/alma_b6/calibrated_{0}_joint.image.fits'.format(source))
    #def add_alma_contours(gc):
    #    levels = np.array([3, 20]) * 5e-5  # in Jy/beam
    #    colors = ['0.400', '0.000']
    #    gc.show_contour(alma, levels=levels, colors=colors,
    #            linewidths=0.5, convention='calabretta')
    # Moment 0, integrated intensity
    gc0 = aplpy.FITSFigure(img0, figure=fig, subplot=[x0, y1, dx, dy])
    gc0.show_colorscale(vmin=0, vmax=15, cmap=HOT_CMAP)
    gc0.tick_labels.hide_x()
    gc0.axis_labels.hide_x()
    gc0.axis_labels.set_ytext(ylabel)
    add_label(r'Moment 0 $(\int I \, dv)$', (labelx, labely))
    add_colorbar(gc0, [x0+dx+0.01, y1, 0.02, dy],
        label=r'$\int I \, dv \ \ [\mathrm{K \, km \, s^{-1}}]$')
    add_beam(gc0, facecolor='white')
    # Maximum, peak intensity
    gc1 = aplpy.FITSFigure(img1, figure=fig, subplot=[x1, y1, dx, dy])
    if matched:
        gc1.show_colorscale(vmin=-0.1, vmax=5, cmap=HOT_CMAP, stretch='power',
                exponent=1)
        add_label(r'Matched Peak', (labelx, labely))
        add_colorbar(gc1, [x1+dx+0.01, y1, 0.02, dy],
            label=r'$\mathrm{max}(\phi * I) \ \ [\mathrm{K}]$')
    else:
        gc1.show_colorscale(vmin=0, vmax=10, cmap=HOT_CMAP, stretch='power',
                exponent=1)
        add_label(r'Maximum $(I_\mathrm{pk})$', (labelx, labely))
        add_colorbar(gc1, [x1+dx+0.01, y1, 0.02, dy],
            label=r'$T_\mathrm{pk} \ \ [\mathrm{K}]$')
    gc1.axis_labels.hide()
    gc1.tick_labels.hide()
    add_beam(gc1, facecolor='white')
    # Moment 1, intensity weighted velocity
    gc2 = aplpy.FITSFigure(img2, figure=fig, subplot=[x0, y0, dx, dy])
    vcen = VELOS[source]
    gc2.show_colorscale(vmin=vcen-2, vmax=vcen+2, cmap=RDB_CMAP)
    gc2.axis_labels.set_xtext(xlabel)
    gc2.axis_labels.set_ytext(ylabel)
    add_label(r'Moment 1 $(v_\mathrm{lsr})$', (labelx, labely))
    add_colorbar(gc2, [x0+dx+0.01, y0, 0.02, dy],
        label=r'$v_\mathrm{lsr} \ \ [\mathrm{km \, s^{-1}}]$')
    add_beam(gc2)
    # Moment 2, velocity dispersion
    gc3 = aplpy.FITSFigure(img3, figure=fig, subplot=[x1, y0, dx, dy])
    gc3.show_colorscale(vmin=0.15, vmax=0.8, cmap=CLR_CMAP)
    add_label(r'Moment 2 $(\sigma_v)$', (labelx, labely))
    add_colorbar(gc3, [x1+dx+0.01, y0, 0.02, dy],
        label=r'$\sigma_v \ \ [\mathrm{km \, s^{-1}}]$')
    gc3.tick_labels.hide_y()
    gc3.axis_labels.hide_y()
    gc3.axis_labels.set_xtext(xlabel)
    add_beam(gc3)
    # Common axis options
    center = get_map_center(img0)
    for gc in (gc0, gc1, gc2, gc3):
        fix_label_ticks(gc, center, radius=1.45/60)
        add_primary_beam_contour(gc, pb, power=0.205)
        add_primary_beam_contour(gc, pb, power=0.5)
    out_filen = '{0}_{1}_moment_four_panel'.format(source, mol)
    out_filen = out_filen + '_mf0' if matched else out_filen
    save_figure(out_filen)


class ParmapHandler:
    filen = 'property_maps/{0}/{0}_nh3_parmap.fits'
    filen_relative = True
    props = {
            'tkin': 0, 'e_tkin':  6,
            'texc': 1, 'e_texc':  7,
            'ncol': 2, 'e_ncol':  8,
            'sigm': 3, 'e_sigm':  9,
            'vcen': 4, 'e_vcen': 10,
    }
    labels = {
              'tkin': r'$T_\mathrm{K} \ [\mathrm{K}]$',
            'e_tkin': r'$\delta T_\mathrm{K} \ [\mathrm{K}]$',
              'texc': r'$T_\mathrm{ex} \ [\mathrm{K}]$',
            'e_texc': r'$\delta T_\mathrm{ex} \ [\mathrm{K}]$',
              'ncol': r'$\log(N(\mathrm{p\hbox{-}NH_3})) \ [\mathrm{cm^{-2}}]$',
            'e_ncol': r'$\delta \log(N(\mathrm{p\hbox{-}NH_3})) \ [\mathrm{cm^{-2}}]$',
              'sigm': r'$\sigma_v \ [\mathrm{km\, s^{-1}}]$',
            'e_sigm': r'$\delta \sigma_v \ [\mathrm{km\, s^{-1}}]$',
              'vcen': r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$',
            'e_vcen': r'$\delta v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$',
              'trot': r'$T_\mathrm{rot} \ [\mathrm{K}]$',
            'e_trot': r'$\delta T_\mathrm{rot} \ [\mathrm{K}]$',
    }
    vsys = VELOS
    err_offset = 6
    bad_thresh = 1e3

    def __init__(self, source):
        """
        Parameters
        ----------

        source : str
            Source name, ex. 'G24051'
        """
        self.source = source
        self.hdul = open_fits(self.filen.format(source), relative=self.filen_relative)
        self.header = self.hdul[0].header
        self.wcs = wcs.WCS(self.hdul[0].header)
        self._clean_hdul()
        self.vsystem = self.vsys[source]

    def _clean_hdul(self):
        ncol = self.get_hdu('ncol').data.copy()
        e_ncol = self.get_hdu('e_ncol').data.copy()
        ncol[(e_ncol > 0.5) | (ncol > 16.9)] = np.nan
        mask = np.isnan(ncol).squeeze()
        data = self.hdul[0].data
        for ii in range(5):
            data[ii][mask] = np.nan

    def get_label(self, prop):
        return self.labels[prop]

    def get_hdu(self, prop, snr_thresh=3, rel_velo=False):
        """
        Parameters
        ----------
        prop : str
        snr_thresh : number, default 3
            SNR cut for pixel selection based on the parameter's associated fit
            uncertainty value.
        rel_velo : bool, default False
            Subtract the source systemic velocity from the velocity centroid
            values.
        """
        # The values come from the cold_ammonia model which uses the Swift
        # (2005) approximation, so to return rotation temperature, just apply
        # equation A6.
        do_rot_conv = prop == 'trot'
        if do_rot_conv:
            prop = 'tkin'
        if prop == 'e_trot':
            prop = 'e_tkin'
        ix = self.props[prop]
        data = self.hdul[0].data[ix:ix+1,:,:]
        data[data == 0] = np.nan
        data[data > self.bad_thresh] = np.nan
        if do_rot_conv:
            dT0 = 41.18  # (2,2)-(1,1) energy difference in K
            data = data*(1 + (data/dT0)*np.log(1 + 0.6*np.exp(-15.7/data)))**(-1)
        if not prop.startswith('e_'):
            ix_e = ix + self.err_offset
            err = self.hdul[0].data[ix_e:ix_e+1,:,:]
            data[np.abs(data / err) < snr_thresh] = np.nan
        if rel_velo and prop == 'vcen':
            data -= self.vsystem
        header = self.header.copy()
        hdu = fits.PrimaryHDU(data=data, header=self.header)
        return hdu


class KeystoneParmapHandler(ParmapHandler):
    filen = 'data/{0}_parameter_maps_all_rebase_multi.fits'
    filen_relative = False
    vsys = {
            'CygX_N':    4.2,  # km/s
            'CygX_S':    2.0,
            'M16':      21.2,
            'M17':      19.2,
            'MonR1':     5.1,
            'MonR2':    10.2,
            'NGC2264':   6.3,
            'NGC7538': -53.6,
            'Rosette':  13.4,
            'W3':      -43.9,
            'W3_west': -36.1,
            'W48':      36.3,
            }

    def _clean_hdul(self):
        data = self.hdul[0].data
        for ii in range(5):
            data[ii][(data[ii] == 0.0) & (data[ii+6] == 0.0)] = np.nan
        mask = (
                # tkin
                (data[0] < 5) | (data[0] > 40) |
                (data[6] > 5) |
                # ncol
                (data[2] < 12.1) | (data[2] > 16) |
                (data[8] > 2) |
                # sigm
                (data[3] < 0.051) | (data[3] > 2.0) |
                (data[9] > 2) |
                # vcen
                (data[10] > 1)
        )
        for ii in range(5):
            data[ii][mask] = np.nan


def plot_four_nh3_properties(source):
    pmh = ParmapHandler(source)
    pb = fits.open(PDir.vla_map_name(source, 'nh3_11', ext='pb.fits'))
    fig = plt.figure(figsize=(8, 6.5))
    split_size = 0.02
    x_margin_size = 0.15
    y_margin_size = 0.075
    delta = 0.06
    x0 = x_margin_size - delta
    y0 = y_margin_size
    dx = (1 - 2 * x_margin_size - split_size) / 2
    dy = (1 - 2 * y_margin_size - split_size) / 2
    x1 = x0 + dx + split_size + 1.5 * delta
    y1 = y0 + dy + split_size + 0.015
    labelx = 0.035
    labely = 0.035
    xlabel = r'$\alpha \ (\mathrm{J2000})$'
    ylabel = r'$\delta \ (\mathrm{J2000})$'
    def add_ncol_contours(gc):
        levels = np.array([14.75, 15.25])  # log(N / cm^2)
        colors = ['0.20', '0.00']
        gc.show_contour(data=pmh.get_hdu('ncol'), levels=levels,
                colors=colors, linewidths=0.5, convention='calabretta')
    # NH3 column density
    gc0 = aplpy.FITSFigure(pmh.get_hdu('ncol'), figure=fig, subplot=[x0, y1, dx, dy])
    gc0.show_colorscale(vmin=14.0, vmax=15.5, cmap=VIR_CMAP)
    gc0.tick_labels.hide_x()
    gc0.axis_labels.hide_x()
    gc0.axis_labels.set_ytext(ylabel)
    add_label(r'Column Density', (labelx, labely))
    add_colorbar(gc0, [x0+dx+0.01, y1, 0.02, dy],
        label=r'$\log(N(\mathrm{NH_3}) \, [\mathrm{cm^{-2}}])$')
    # Kinetic temperature
    gc1 = aplpy.FITSFigure(pmh.get_hdu('tkin'), figure=fig, subplot=[x1, y1, dx, dy])
    gc1.show_colorscale(vmin=10, vmax=18, cmap=VIR_CMAP)
    gc1.axis_labels.hide()
    gc1.tick_labels.hide()
    add_label(r'Kinetic Temp.', (labelx, labely))
    add_colorbar(gc1, [x1+dx+0.01, y1, 0.02, dy],
        label=r'$T_\mathrm{K} \, [\mathrm{K}]$')
    # Velocity centroid
    gc2 = aplpy.FITSFigure(pmh.get_hdu('vcen'), figure=fig, subplot=[x0, y0, dx, dy])
    vcen = VELOS[source]
    gc2.show_colorscale(vmin=vcen-2, vmax=vcen+2, cmap=RDB_CMAP)
    gc2.axis_labels.set_xtext(xlabel)
    gc2.axis_labels.set_ytext(ylabel)
    add_label(r'Velo. Centroid', (labelx, labely))
    add_colorbar(gc2, [x0+dx+0.01, y0, 0.02, dy],
        label=r'$v_\mathrm{lsr} \ \ [\mathrm{km \, s^{-1}}]$')
    # Velocity dispersion
    gc3 = aplpy.FITSFigure(pmh.get_hdu('sigm'), figure=fig, subplot=[x1, y0, dx, dy])
    gc3.show_colorscale(vmin=0.15, vmax=0.8, cmap=CLR_CMAP)
    add_label(r'Velo. Dispersion', (labelx, labely))
    add_colorbar(gc3, [x1+dx+0.01, y0, 0.02, dy],
        label=r'$\sigma_v \ \ [\mathrm{km \, s^{-1}}]$')
    gc3.tick_labels.hide_y()
    gc3.axis_labels.hide_y()
    gc3.axis_labels.set_xtext(xlabel)
    # Common axis options
    center = get_map_center([pmh.get_hdu('ncol')])
    for gc in (gc0, gc1, gc2, gc3):
        fix_label_ticks(gc, center, radius=1.45/60)
        add_beam(gc)
        add_primary_beam_contour(gc, pb, power=0.205)
        add_primary_beam_contour(gc, pb, power=0.5)
    save_figure(f'{source}_parmap_four_panel')


def get_keystone_prior_pdfs(nbins=100):
    all_bins = [
            np.linspace(lo, hi, nbins)
            for lo, hi in [
                (  7.0, 30.0),  # trot, K
                (  2.9, 12.0),  # texc, K
                ( 12.0, 17.0),  # ncol, log(cm^-2)
                (  0.0,  2.0),  # sigm, km/s
                (-10.0, 10.0),  # vcen, km/s (relative)
            ]
    ]
    u = np.linspace(0, 1, nbins)
    x_voff = 10.00 * u -  5.00
    x_trot = 23.00 * u +  7.00
    x_tex  =  9.26 * u +  2.80
    x_ntot =  3.50 * u + 12.50
    x_sigm =  2.00 * u +  0.067
    all_x = (x_trot, x_tex, x_ntot, x_sigm, x_voff)
    # prior PDFs values
    f_voff = sp.stats.beta( 5.0, 5.0).pdf(u)
    f_trot = sp.stats.beta( 3.0, 4.5).pdf(u)
    f_tex  = sp.stats.beta( 1.0, 3.5).pdf(u)
    f_ntot = sp.stats.beta( 4.5, 6.0).pdf(u)
    f_sigm = sp.stats.beta( 1.5, 5.0).pdf(u)
    all_dist = (f_trot, f_tex, f_ntot, f_sigm, f_voff)
    return all_bins, all_x, all_dist


def plot_keystone_post_stack(store):
    pmh = KeystoneParmapHandler('MonR2')
    nbest = store.hdf['/products/conv_nbest'][...]
    # dimensions (m, p, b, l)
    pmap = store.hdf['/products/nbest_MAP'][...]
    pmap[:,0,:,:] -= pmh.vsystem
    # dimensions (r, m, p, h, b, l)
    post = store.hdf['/products/post_pdfs'][...]
    post = take_by_components(post, nbest)  # -> (m, p, h, b, l)
    pdf_bins = store.hdf['/products/pdf_bins'][...]
    pdf_bins[0] -= pmh.vsystem
    all_bins, all_x, all_dist = get_keystone_prior_pdfs()
    props = ['trot', 'texc', 'ncol', 'sigm', 'vcen']
    ix = [1, 2, 3, 4, 0]
    fig, axes = plt.subplots(ncols=1, nrows=len(props), figsize=(4, 6))
    for ii, prop, bins, x, dist, ax in zip(ix, props, all_bins, all_x, all_dist, axes):
        # maximum likelihood distribution
        vals = pmh.get_hdu(prop, rel_velo=True, snr_thresh=5).data.squeeze()
        hist, _, _ = ax.hist(vals.flatten(), bins=bins, density=True,
                color='dodgerblue', histtype='step', linewidth=1.0, zorder=20)
        hist_max = hist.max()
        # prior distribution
        ax.plot(x, dist*hist_max/dist.max(), 'm-', zorder=21)
        # average/summed posterior distibution
        # mask where values are also well-constrained in the ML fits
        apdf = np.nansum(post[...,~np.isnan(vals)], axis=(0, 3))[ii,:]
        ax.fill_between(pdf_bins[ii], apdf*hist_max/apdf.max(), color='0.7',
                step='mid', zorder=0)
        # MAP histogram values
        map_vals = pmap[:,ii,~np.isnan(vals)]
        ax.hist(map_vals.flatten(), bins=bins, density=True,
                color='firebrick', histtype='step', linewidth=1.0, zorder=22)
        ax.set_ylim(0, hist.max()*1.1)
        ax.set_xlim(bins.min(), bins.max())
        ax.set_xlabel(pmh.get_label(prop))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    ax.set_ylabel('PDF')
    plt.tight_layout(h_pad=0.5)
    save_figure(f'priors_KS_post_stack_trot_monr2', do_eps=False)


def plot_ncol_texc_hist2d(all_pmh=None):
    if all_pmh is None:
        all_pmh = [ParmapHandler(s) for s in TARGETS if s != 'G285_mosaic']
    texc = np.concatenate([pmh.get_hdu('texc').data.flatten() for pmh in all_pmh])
    ncol = np.concatenate([pmh.get_hdu('ncol').data.flatten() for pmh in all_pmh])
    t_bins = np.linspace( 3.5,  8.0, 50)
    n_bins = np.linspace(13.0, 15.5, 50)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist2d(texc, ncol, bins=(t_bins, n_bins), cmin=1, cmap=CLR_CMAP)
    ax.plot([3.5, 8], [14.05, 14.55], 'w-')
    ax.plot([3.5, 8], [14.05, 14.55], 'k:')
    ax.set_xlabel(ParmapHandler.labels['texc'])
    ax.set_ylabel(ParmapHandler.labels['ncol'])
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tight_layout(h_pad=0.5)
    save_figure(f'ncol_texc_hist2d', do_eps=False)


def plot_keystone_ncol_texc_hist2d(all_pmh=None):
    if all_pmh is None:
        all_pmh = [
                KeystoneParmapHandler(s)
                for s in KeystoneParmapHandler.vsys.keys()
        ]
    texc = np.concatenate([pmh.get_hdu('texc').data.flatten() for pmh in all_pmh])
    ncol = np.concatenate([pmh.get_hdu('ncol').data.flatten() for pmh in all_pmh])
    t_bins = np.linspace( 3.5,  8.0, 50)
    n_bins = np.linspace(13.0, 15.5, 50)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist2d(texc, ncol, bins=(t_bins, n_bins), cmin=1, cmap=CLR_CMAP)
    ax.plot([3.5, 8], [14.05, 14.55], 'w-')
    ax.plot([3.5, 8], [14.05, 14.55], 'k:')
    ax.set_xlabel(ParmapHandler.labels['texc'])
    ax.set_ylabel(ParmapHandler.labels['ncol'])
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tight_layout(h_pad=0.5)
    save_figure(f'KS_ncol_texc_hist2d', do_eps=False)


