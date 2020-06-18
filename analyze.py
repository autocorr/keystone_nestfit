#!/usr/bin/env python3

import sys
import shutil
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import scipy as sp

import spectral_cube
from astropy import convolution
from astropy.io import fits

sys.path.append('/lustre/aoc/users/bsvoboda/temp/nestfit')
import nestfit as nf
from nestfit import (
        Distribution,
        Prior,
        ResolvedPlacementPrior,
        ConstantPrior,
        PriorTransformer,
)


FWHM = 2 * np.sqrt(2 * np.log(2))
DATA_PATH = Path('data')
TARGETS = ['MonR2', 'M17']
VELOS = {
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
LINE_NAMES = {1: '11', 2: '22'}
KIND_EXT = {'image': '', 'mom0': '_mom0_QA', 'rms': '_rms_QA'}


def read_cube(target, line=1):
    line_name = LINE_NAMES[line]
    path = DATA_PATH / f'{target}_NH3_{line_name}_all_rebase_multi.fits'
    cube = spectral_cube.SpectralCube.read(str(path), memmap=False)
    # Cubes are big-endian, convert byteorder through cast
    cube._data = cube._data.astype(float, copy=False)
    return cube


def read_noise(target, line=1):
    line_name = LINE_NAMES[line]
    path = DATA_PATH / f'{target}_NH3_{line_name}_all_rebase_multi_rms_QA.fits'
    hdul = fits.open(str(path))
    data = hdul[0].data
    noise_map = nf.NoiseMap(data)
    return noise_map


def get_cubestack(target):
    cubes = [
            nf.DataCube(
                read_cube(target, line=i),
                noise_map=read_noise(target, line=i),
                trans_id=i,
            )
            for i in range(1, 3)
    ]
    return nf.CubeStack(cubes)


def get_keystone_priors(target, size=500):
    vsys = VELOS[target]
    u = np.linspace(0, 1, size)
    # prior distribution x axes
    if target == 'MonR2':
        # 0 voff [-5.00,   5.0] km/s  (centered on vsys)
        # 1 trot [ 7.00,  30.0] K
        # 2 tex  [ 2.80,  12.0] K
        # 3 ntot [12.50,  16.0] log(cm^-2)
        # 4 sigm [    C, C+2.0] km/s  (with min sigma "C")
        x_voff = 10.00 * u -  5.00 + vsys
        x_trot = 23.00 * u +  7.00
        x_tex  =  9.26 * u +  2.80
        x_ntot =  3.50 * u + 12.50
        x_sigm =  2.00 * u +  0.067
    elif target == 'M17':
        # 0 voff [-6.00,   6.0] km/s  (centered on vsys)
        # 1 trot [ 7.00,  40.0] K
        # 2 tex  [ 2.80,  15.0] K
        # 3 ntot [12.50,  16.0] log(cm^-2)
        # 4 sigm [    C, C+2.0] km/s  (with min sigma "C")
        x_voff = 12.00 * u -  6.00 + vsys
        x_trot = 33.00 * u +  7.00
        x_tex  = 12.26 * u +  2.80
        x_ntot =  3.50 * u + 12.50
        x_sigm =  2.00 * u +  0.067
    # prior PDFs values
    f_voff = sp.stats.beta( 5.0, 5.0).pdf(u)
    f_trot = sp.stats.beta( 3.0, 4.5).pdf(u)
    f_tex  = sp.stats.beta( 1.0, 3.5).pdf(u)
    f_ntot = sp.stats.beta( 4.5, 6.0).pdf(u)
    f_sigm = sp.stats.beta( 1.5, 5.0).pdf(u)
    # and distribution instances
    d_voff = Distribution(x_voff, f_voff)
    d_trot = Distribution(x_trot, f_trot)
    d_tex  = Distribution(x_tex,  f_tex)
    d_ntot = Distribution(x_ntot, f_ntot)
    d_sigm = Distribution(x_sigm, f_sigm)
    # interpolation values, transformed to the intervals:
    priors = np.array([
            ResolvedPlacementPrior(
                Prior(d_voff, 0),
                Prior(d_sigm, 4),
                scale=1.2,
            ),
            Prior(d_trot, 1),
            Prior(d_tex,  2),
            Prior(d_ntot, 3),
            ConstantPrior(0, 5),
    ])
    return PriorTransformer(priors)


def get_runner(stack, utrans, ncomp=1):
    nlon, nlat = stack.spatial_shape
    spec_data, has_nans = stack.get_spec_data(nlon//2, nlat//2)
    assert not has_nans
    runner = nf.AmmoniaRunner.from_data(spec_data, utrans, ncomp=ncomp)
    return runner


def get_bins(target, nbins=200):
    vsys = VELOS[target]
    bin_minmax = [
            (vsys-5.0, vsys+5.0),  # vcen
            ( 7.0, 30.0),  # trot
            ( 2.8, 12.0),  # tex
            (12.5, 16.5),  # ncol
            ( 0.0,  2.0),  # sigm
            ( 0.0,  1.0),  # orth
    ]
    bins = np.array([
            np.linspace(lo, hi, nbins)
            for (lo, hi) in bin_minmax
    ])
    return bins


def if_exists_delete_store(name):
    filen = f'{name}.store'
    if Path(filen).exists():
        print(f'-- Deleting {filen}')
        shutil.rmtree(filen)


def run_nested(target, store_prefix, nproc=8):
    store_name = f'run/{store_prefix}_{target}'
    if_exists_delete_store(store_name)
    utrans = get_keystone_priors(target)
    runner_cls = nf.AmmoniaRunner
    stack = get_cubestack(target)
    fitter = nf.CubeFitter(stack, utrans, runner_cls, ncomp_max=2,
            mn_kwargs={'nlive': 500}, nlive_snr_fact=20)
    fitter.fit_cube(store_name=store_name, nproc=nproc)


def run_nested_all(store_prefix, nproc=8):
    for target in TARGETS:
        run_nested(target, store_prefix, nproc=nproc)


def get_info_kernel(nrad):
    hpbw = 32  # arcsec
    pix_size = 8.8  # arcsec
    beam_sigma_pix = hpbw / FWHM / pix_size
    k_arr = nf.get_indep_info_kernel(beam_sigma_pix, nrad=nrad)
    k_arr = nf.apply_circular_mask(k_arr, radius=3)
    post_kernel = convolution.CustomKernel(k_arr)
    return post_kernel


def postprocess_run(target, store_prefix):
    print(f':: Post-processing {target}')
    # Standard deviation in pixels: 0.85 -> FWHM 17.6 (cf. HPBW / 2 = 16 as)
    evid_kernel = convolution.Gaussian2DKernel(0.85)
    post_kernel = get_info_kernel(3)  # 3.5 pixel radius circular window
    utrans = get_keystone_priors(target)
    par_bins = get_bins(target)
    store_name = f'run/{store_prefix}_{target}'
    store = nf.HdfStore(store_name)
    stack = get_cubestack(target)
    runner = get_runner(stack, utrans, ncomp=1)
    # begin post-processing steps
    nf.aggregate_run_attributes(store)
    nf.convolve_evidence(store, evid_kernel)
    nf.aggregate_run_products(store)
    nf.aggregate_run_pdfs(store, par_bins=par_bins)
    nf.convolve_post_pdfs(store, post_kernel, evid_weight=False)
    nf.quantize_conv_marginals(store)
    nf.deblend_hf_intensity(store, stack, runner)
    store.close()


def serial_postprocess(store_prefix):
    for target in TARGETS:
        postprocess_run(target, store_prefix)


def parallel_postprocess(store_prefix, nproc=12):
    args = zip(
            TARGETS_NOMOS,
            [store_prefix] * len(TARGETS_NOMOS),
    )
    with Pool(nproc) as pool:
        pool.starmap(postprocess_run, args)


if __name__ == '__main__':
    prefix = 'nested'
    args = sys.argv[1:]
    assert len(args) > 0
    assert args[0] in ('--run-nested', '--post-proc')
    flag = args[0]
    if flag == '--run-nested':
        #run_nested_all(prefix, nproc=16)
        run_nested('MonR2', 'nested_s1.2', nproc=16)
    elif flag == '--post-proc':
        #parallel_postprocess(prefix, nproc=12)
        postprocess_run('M17', 'nested')


