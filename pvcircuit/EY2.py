# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package.
    pvcircuit.EY     use Ripalda's Tandem proxy spectra for energy yield
"""

import copy
import glob
import multiprocessing as mp
import os
from functools import lru_cache
import matplotlib.pyplot as plt

import numpy as np  # arrays
import pandas as pd
from parse import parse
from scipy import constants
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from tqdm.autonotebook import tqdm, trange

import pvcircuit as pvc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import pvcircuit as pvc
from scipy import constants
from scipy.optimize import curve_fit, fsolve
from scipy.special import erfc
import itertools

from scipy.signal import find_peaks

from pvcircuit.physics_helpers import fit_sandia_simple
from pvlib.pvsystem import singlediode
import pvlib

import os
import sys

from glob import glob
from datetime import datetime, timedelta
import time
from tqdm.notebook import trange
import timeit
import pickle

# set path for NREL meteorological package and import
# sys.path.append("../../NREL_Meteorological")
from pvcircuit import environmental, Spectra, Meteorological, sync
import warnings
from scipy.optimize import OptimizeWarning
import pickle
from scipy.interpolate import interp1d


# from pvcircuit import Tandem3T
# from pvcircuit import Spectra, Meteorological, sync
from pvcircuit.physics_helpers import fit_sandia_simple
from pvlib.pvsystem import singlediode

#  from 'Tandems' project
# vectoriam = np.vectorize(physicaliam)

# standard data
PVCPATH = os.path.dirname(os.path.dirname(__file__))
DATAPATH = os.path.join(PVCPATH, "data", "")  # Data files here
# datapath = os.path.abspath(os.path.relpath('../data/', start=__file__))
# datapath = pvcpath.replace('/pvcircuit','/data/')
AM15G = os.path.join(DATAPATH, "ASTMG173.csv")


@lru_cache(maxsize=100)
def VMloss(type3T, bot, top, ncells):
    # calculates approximate loss factor for VM strings of 3T tandems
    if bot == 0:
        endloss = 0
    elif top == 0:
        endloss = 0
    elif type3T == "r":
        endloss = max(bot, top) - 1
    elif type3T == "s":
        endloss = bot + top - 1
    else:  # not Tandem3T
        endloss = 0
    lossfactor = 1 - endloss / ncells
    return lossfactor


# @lru_cache(maxsize=100)
def VMlist(mmax):
    # generate a list of VM configurations + 'MPP'=4T and 'CM'=2T
    # mmax < 10 for formating reasons

    sVM = ["MPP", "CM", "VM11"]
    for m in range(mmax + 1):
        for n in range(1, m):
            lcd = 2
            if m / lcd == round(m / lcd) and n / lcd == round(n / lcd):
                # print(n,m, 'skip2')
                continue
            lcd = 3
            if m / lcd == round(m / lcd) and n / lcd == round(n / lcd):
                # print(n,m, 'skip3')
                continue
            lcd = 5
            if m / lcd == round(m / lcd) and n / lcd == round(n / lcd):
                # print(n,m, 'skip5')
                continue
            # print(n,m, 'ok')
            sVM.append("VM" + str(m) + str(n))
    return sVM


def sandia_T(poa_global, wind_speed, temp_air):
    """Sandia solar cell temperature model
    Adapted from pvlib library to avoid using pandas dataframes
    parameters used are those of 'open_rack_cell_polymerback'
    """

    a = -3.56
    b = -0.075
    deltaT = 3

    E0 = 1000.0  # Reference irradiance

    temp_module = poa_global * np.exp(a + b * wind_speed) + temp_air

    temp_cell = temp_module + (poa_global / E0) * (deltaT)

    return temp_cell


# def _calc_yield_async(i, bot, top, type3T, Jscs, Egs, TempCell, devlist, oper):
#     model = devlist[i]
#     if type3T == "2T":  # Multi2T
#         for ijunc in range(model.njuncs):
#             # model.j[ijunc].set(Eg=Egs[ijunc], Jext=Jscs[i, ijunc], TC=TempCell[i])
#             model.j[ijunc].set(Eg=Egs[i,ijunc], Jext=Jscs[i, ijunc], TC=25)
#         mpp_dict = model.MPP()  # oper ignored for 2T
#         Pmax = mpp_dict["Pmp"]
#     elif type3T in ["s", "r"]:  # Tandem3T
#         model.top.set(Eg=Egs[i,0], Jext=Jscs[i, 0], TC=TempCell[i])
#         # model.top.set(Eg=Egs[0], Jext=Jscs[i, 0], TC=TempCell)
#         model.bot.set(Eg=Egs[i,1], Jext=Jscs[i, 1], TC=TempCell[i])
#         # model.bot.set(Eg=Egs[1], Jext=Jscs[i, 1], TC=25)
#         if oper == "MPP":
#             tempRz = model.Rz
#             model.set(Rz=0)
#             iv3T = model.MPP()
#             model.set(Rz=tempRz)
#         elif oper == "CM":
#             ln, iv3T = model.CM()
#         elif oper[:2] == "VM":
#             ln, iv3T = model.VM(bot, top)
#         else:
#             iv3T = pvc.iv3T.IV3T("bogus")
#             iv3T.Ptot[0] = 0
#         Pmax = iv3T.Ptot[0]
#     else:
#         Pmax = 0.0
#     # outPowerMP[i] = Pmax *1e4
#     return Pmax * 1e4


def _calc_yield_async_2T(tc_jscs, bc_jscs, tc_egs, bc_egs, TempCell, devlist):
    Pmax_out = np.zeros(len(tc_jscs))
    for i in range(len(tc_jscs)):
        model = devlist[i]
        model.top.set(Eg=tc_egs[i], Jext=tc_jscs[i], TC=TempCell[i])
        # model.top.set(Eg=Egs[0], Jext=Jscs[i, 0], TC=TempCell)
        model.bot.set(Eg=bc_egs[i], Jext=bc_jscs[i], TC=TempCell[i])
        ln, iv3T = model.CM()
        Pmax = iv3T.Ptot[0]
        Pmax_out[i] = Pmax * 1

    return Pmax_out


def _calc_yield_async_3T(tc_jscs, bc_jscs, tc_egs, bc_egs, TempCell, devlist):
    Pmax_out = np.zeros(len(tc_jscs))
    for i in range(len(tc_jscs)):
        model = devlist[i]
        model.top.set(Eg=tc_egs[i], Jext=tc_jscs[i], TC=TempCell[i])
        # model.top.set(Eg=Egs[0], Jext=Jscs[i, 0], TC=TempCell)
        model.bot.set(Eg=bc_egs[i], Jext=bc_jscs[i], TC=TempCell[i])
        ln, iv3T = model.VM(2, 1)
        Pmax = iv3T.Ptot[0]
        Pmax_out[i] = Pmax * 1

    return Pmax_out


def _calc_yield_async_4T(tc_jscs, bc_jscs, tc_egs, bc_egs, TempCell, devlist):
    Pmax_out = np.zeros(len(tc_jscs))
    for i in range(len(tc_jscs)):
        model = devlist[i]
        model.top.set(Eg=tc_egs[i], Jext=tc_jscs[i], TC=TempCell[i])
        # model.top.set(Eg=Egs[0], Jext=Jscs[i, 0], TC=TempCell)
        model.bot.set(Eg=bc_egs[i], Jext=bc_jscs[i], TC=TempCell[i])
        iv3T = model.MPP()
        Pmax = iv3T.Ptot[0]
        Pmax_out[i] = Pmax * 1

    return Pmax_out


def get_jv_params(fp_jv, plot_fits: bool = False, ilim: float = None):
    data = pd.read_csv(fp_jv)
    A = 1  # [cm^2]

    # Measured terminal voltage.
    voltage = data["v"].to_numpy(np.double)  # [V]
    # Measured terminal current.
    current = data["i"].to_numpy(np.double) / 1000 * A  # [A/cm^2]

    sort_id = np.argsort(voltage)

    voltage = voltage[sort_id]
    current = current[sort_id]

    voc_guess_idx = np.where(current >= 0)[0][-1]
    voltage = voltage[: voc_guess_idx + 2]
    current = current[: voc_guess_idx + 2]

    # fit iv
    # (psc_isc, psc_io, psc_rs, psc_rsh, psc_nNsVth) = psc_params
    diode_params = fit_sandia_simple(voltage, current)  # [A/cm^2]  # [A/cm^2]  # [Ohm cm^2]  # [Ohm cm^2]

    if ilim:
        while diode_params[2] < 0:
            ilim -= 0.01
            # fit iv
            diode_params = fit_sandia_simple(voltage, current, ilim=ilim)  # [A/cm^2]  # [A/cm^2]  # [Ohm cm^2]  # [Ohm cm^2]

    if plot_fits:
        fit = singlediode(*diode_params, ivcurve_pnts=30)
        fig, ax = plt.subplots()
        ax.plot(voltage, current, "k*", label="data")
        ax.plot(fit["v"], fit["i"], "k--", label="fit")

    return diode_params


def wavelength_to_photonenergy(wavelength):
    return constants.h * constants.c / (wavelength * 1e-9) / constants.e


def photonenergy_to_wavelength(photonenergy):
    return constants.h * constants.c / (photonenergy * 1e-9) / constants.e


def _normalize(eqe: pd.DataFrame) -> pd.DataFrame:
    eqe_min = np.nanmin(eqe)
    eqe_max = np.nanmax(eqe)
    return (eqe - eqe_min) / (eqe_max - eqe_min)


def _eq_solve_Eg(Eg, *data):
    x, y = data
    return np.trapz(x * y, x) / np.trapz(y, x) - Eg


def _gaussian(x, a, x0, sigma):
    return 1 * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def calc_Eg_Rau(eqe, fit_gaussian=True):
    # using [1] U. Rau, B. Blank, T. C. M. Müller, and T. Kirchartz,
    # “Efficiency Potential of Photovoltaic Materials and Devices Unveiled by Detailed-Balance Analysis,”
    # Phys. Rev. Applied, vol. 7, no. 4, p. 044016, Apr. 2017, doi: 10.1103/PhysRevApplied.7.044016.
    # extended by gaussian fit

    # Define the Gaussian function
    wavelength = eqe.index.values
    y = eqe.values
    x = wavelength_to_photonenergy(wavelength)

    # convert wavelength to photon energy
    y_grad = np.abs(np.gradient(y))
    # y_grad = np.abs(np.diff(y.values, prepend=np.nan))
    # y_grad = y.diff().abs().values

    # filter tail to avoid eqe dips at end/beginning of measurement
    y_grad = y_grad[(x < (x.max() + x.min()) / 2)]
    y = y[(x < (x.max() + x.min()) / 2)]
    x = x[(x < (x.max() + x.min()) / 2)]
    # normalize data
    y_grad = _normalize(y_grad)

    # Find local maxima
    maxima_indices, _ = find_peaks(y_grad, distance=len(x) / 5)

    # Get x, y values of local maxima
    # maxima_x = x[maxima_indices]
    # maxima_y = y_grad[maxima_indices]

    # get the index of the maximum, using only the maxima with the lowest energy/highest wavelength
    # y_diff_max_idx = maxima_indices.max()
    y_diff_max_idx = np.nanargmax(y_grad)

    # get the max coordinates
    x_diff_max = x[y_diff_max_idx]
    y_diff_max = y_grad[y_diff_max_idx]

    # define lower threshold
    p_ab = np.exp(-2) * y_diff_max
    thres = 0.5
    # p_ab = thres * y_diff_max
    # find the index of the low-energy side where P(a) is max(P(Eg)/2)
    a_cond = np.where((y_grad < p_ab) & (x < x_diff_max))[0]
    if len(a_cond > 0):
        a_idx = np.nanmin(a_cond)
    else:
        a_idx = len(x) - 1
    a = x[a_idx]
    p_a = y_grad[a_idx]
    # find the index of the high-energy side where P(b) is max(P(Eg)/2)
    b_idx = np.nanmax(np.where((y_grad < p_ab) & (x > x_diff_max))[0])
    b = x[b_idx]
    p_b = y_grad[b_idx]

    x_target = x[a_idx : b_idx - 1 : -1]
    y_target = y_grad[a_idx : b_idx - 1 : -1]

    if fit_gaussian:
        # initial guesses from weighted arithmetic mean and weighted sample sigma
        mean = sum(x_target * y_target) / sum(y_target)
        sigma = np.sqrt(sum(y_target * (x_target - mean) ** 2) / sum(y_target))

        fit_res = curve_fit(
            _gaussian,
            x_target,
            y_target,
            p0=[max(y_target), mean, sigma],
        )
        x_fit = np.linspace(x[b_idx], x[a_idx], 100)
        y_fit = _gaussian(x_fit, *fit_res[0])

        # fig,ax = plt.subplots(1)
        # ax.plot(x, y_grad,'.')
        # ax.plot(x_fit, y_fit)
        # ax.plot(x_target, y_target, ".r")
        # ax.plot(x_diff_max, y_diff_max, "r*")
        # ax.plot(a, p_a, "g*")
        # ax.plot(b, p_b, "b*")
        # plt.plot(x_fit - fit_res[0][1], y_fit)
        # plt.plot(x_fit, y_fit)
        # ax.set_xlim(1.1, 1.8)
        # ax.set_ylabel(r"$\frac{{\mathrm{d}}EQE}{{\mathrm{d}}E_{\nu}}$")
        # ax.set_xlabel(r"Photon energy $E_{\nu}$ [eV]")

        x_fit = x_fit[y_fit >= thres * y_fit.max()]
        y_fit = y_fit[y_fit >= thres * y_fit.max()]
        sigma = fit_res[0][2]

    else:
        fit_res = [None]
        x_fit = x_target
        y_fit = y_target
        sigma = None

    res = fsolve(_eq_solve_Eg, 1.0, args=(x_fit, y_fit))
    bandgap = res[0]
    return bandgap, sigma


def integrate_eqe(eqe, spectra=None):
    if spectra is None:
        # spectra = pd.read_csv(path_to_am15_spectra, header=[*range(0,3)])
        spectra_full = pd.read_csv(AM15G, header=[2])
        spectra_full.set_index(spectra_full.columns[0], inplace=True)

        # spectra = spectra_full.reindex(eqe.index)
        # if spectra.isnull().any(axis=1).any():
        #     spectra = spectra.reindex(index=eqe.index).interpolate(method="index")

        spectra = spectra_full["global"]

    # jsc = (
    #     eqe.apply(lambda x: np.trapz(y=x * spectra["global"] / qe._wavelength_to_photonenergy(eqe.index), x=eqe.index)) / 10
    # )
    spec_interp_func = interp1d(spectra.index, spectra, axis=0)
    spectra_interp = pd.DataFrame(spec_interp_func(eqe.index.astype(np.float64)).astype(np.float64), index=eqe.index)

    jsc = (
        np.trapz(
            y=eqe.values.reshape(-1, 1) * spectra_interp.values / wavelength_to_photonenergy(eqe.index.values.reshape(-1, 1)),
            x=eqe.index.values.reshape(-1, 1),
            axis=0,
        )
        / 10
    )

    return jsc


def calc_ape(spectra):
    """
    Calcualtes the average photon energy (APE) of the spectra
    """
    wavelength = spectra.columns.astype(np.float64)
    phi = spectra * (wavelength * 1e-9) / constants.h / constants.c
    ape = np.trapz(x=wavelength, y=spectra.values) / constants.e / np.trapz(x=wavelength, y=phi.values)
    return ape


def si_eg_shift(temperature, bandgap_25, p):
    return (p[0] * temperature + p[1]) * bandgap_25


def si_sigma_shift(temperature, sigma_25, p):
    return (p[0] * temperature + p[1]) * sigma_25


def psc_eg_shift(temperature, bandgap_25, p, t_split):
    res = np.zeros_like(temperature)

    t_filter = temperature > t_split
    res[t_filter] = p[0] * temperature[t_filter] + p[1]
    res[~t_filter] = p[0] * t_split + p[1]
    # res = pd.Series(res, index = temperature.index)
    return res * bandgap_25


def psc_sigma_shift(temperature, sigma_25, p):
    return (p[0] * temperature + p[1]) * sigma_25


def shift_eqe_tcbc(
    tc_eqe_ref,
    tc_bandgap_25,
    tc_sigma_25,
    tc_bandgaps,
    tc_sigmas,
    bc_eqe_ref,
    bc_bandgap_25,
    bc_sigma_25,
    bc_bandgaps,
    bc_sigmas,
    spectra,
):
    vec_erfc = np.vectorize(erfc)
    tc_trans = None
    # in case values are percentages

    lam = tc_eqe_ref.index.values

    tc_eqe = tc_eqe_ref.values
    tc_Ey = constants.h * constants.c / (lam * 1e-9) / constants.e  # [eV]

    tc_lam_eqe_saturation_idx = np.argmax(tc_eqe * lam)
    tc_eqe_saturation = tc_eqe[tc_lam_eqe_saturation_idx]
    # using 25 degC EQE for saturation
    tc_eqe_saturation = tc_eqe_ref[lam > photonenergy_to_wavelength(tc_bandgap_25 + 2 * tc_sigma_25)].iloc[0]

    bc_eqe = bc_eqe_ref.values
    bc_Ey = constants.h * constants.c / (lam * 1e-9) / constants.e  # [eV]

    bc_lam_eqe_saturation_idx = np.argmax(bc_eqe * lam)
    bc_eqe_saturation = bc_eqe[bc_lam_eqe_saturation_idx]
    # using 25 degC EQE for saturation
    bc_eqe_saturation = bc_eqe_ref[lam > photonenergy_to_wavelength(bc_bandgap_25 + 2 * bc_sigma_25)].iloc[0]

    # if len(spectra) > 1:
    #     spec_interp_func = interp1d(spectra.index, spectra,axis=0)
    #     spectra_interp = pd.DataFrame(spec_interp_func(spectra.index.astype(np.float64)).astype(np.float64), index = spectra.index.astype(np.float64))

    # else:
    #     spectra_interp = spectra.T

    tc_bandgaps_arr = np.tile(tc_bandgaps, [len(tc_Ey), 1])
    tc_sigmas_arr = np.tile(tc_sigmas, [len(tc_Ey), 1])
    tc_erfc_arr = (tc_bandgaps_arr - tc_Ey.reshape(-1, 1)) / (tc_sigmas_arr * np.sqrt(2))
    tc_eqe_filter = np.tile(lam, [len(tc_bandgaps), 1]).T > photonenergy_to_wavelength(tc_bandgaps_arr + 2 * tc_sigmas_arr)
    tc_eqe_new_arr = np.tile(tc_eqe, [len(tc_bandgaps), 1]).T
    tc_abs_arr = vec_erfc(tc_erfc_arr) * 0.5 * tc_eqe_saturation
    # tc_abs_arr = vec_erfc(tc_erfc_arr) * 0.5 * np.tile(tc_eqe_new_arr[tc_eqe_filter.argmax(axis=0),:].reshape(1,-1), (tc_eqe_new_arr.shape[0],1))
    tc_eqe_new_arr = tc_eqe_new_arr * ~tc_eqe_filter + tc_abs_arr * tc_eqe_filter

    # fig,ax = plt.subplots()
    # ax.plot(lam,tc_abs_arr[:,0], "k--")
    # ax.plot(lam,tc_eqe_new_arr[:,0] * ~tc_eqe_filter[:,0], "r--")
    # ax.plot(lam,tc_abs_arr[:,0] * tc_eqe_filter[:,0], "b--")
    # ax.plot(lam,tc_eqe_ref, "m-")

    tc_trans = None
    if tc_trans is None:
        tc_trans = 1 - _normalize(tc_eqe_new_arr)

    eqe_max_idx = np.argmax(tc_eqe_new_arr, axis=0)
    filter_idx = (tc_eqe_new_arr < 0.01) & (tc_eqe_new_arr > eqe_max_idx)
    tc_trans[filter_idx] = 1

    bc_bandgaps_arr = np.tile(bc_bandgaps, [len(bc_Ey), 1])
    bc_sigmas_arr = np.tile(bc_sigmas, [len(bc_Ey), 1])
    bc_erfc_arr = (bc_bandgaps_arr - bc_Ey.reshape(-1, 1)) / (bc_sigmas_arr * np.sqrt(2))
    bc_eqe_filter = np.tile(lam, [len(bc_bandgaps), 1]).T > photonenergy_to_wavelength(bc_bandgaps_arr + 2 * bc_sigmas_arr)
    bc_eqe_new_arr = np.tile(bc_eqe, [len(bc_bandgaps), 1]).T
    bc_abs_arr = vec_erfc(bc_erfc_arr) * 0.5 * bc_eqe_saturation
    # bc_abs_arr = vec_erfc(bc_erfc_arr) * 0.5 * np.tile(bc_eqe_new_arr[bc_eqe_filter.argmax(axis=0),:].reshape(1,-1), (bc_eqe_new_arr.shape[0],1))
    bc_eqe_new_arr = bc_eqe_new_arr * ~bc_eqe_filter + bc_abs_arr * bc_eqe_filter

    # Apply TC transmittance to BC
    bc_eqe_new_arr = bc_eqe_new_arr * tc_trans

    spec_interp_func = interp1d(spectra.columns.astype(float), spectra.T, axis=0)
    spectra_interp = pd.DataFrame(spec_interp_func(lam), index=lam)

    tc_jscs = (
        np.trapz(
            y=tc_eqe_new_arr * spectra_interp.values / wavelength_to_photonenergy(lam).reshape(-1, 1),
            x=lam.reshape(-1, 1),
            axis=0,
        )
        / 10
    )
    bc_jscs = (
        np.trapz(
            y=bc_eqe_new_arr * spectra_interp.values / wavelength_to_photonenergy(lam).reshape(-1, 1),
            x=lam.reshape(-1, 1),
            axis=0,
        )
        / 10
    )

    # fig,ax = plt.subplots(1,2)
    # ax[0].plot(lam,tc_eqe_new_arr)
    # ax[0].plot(tc_eqe_ref, "r--")

    # ax[1].plot(lam,bc_eqe_new_arr)
    # ax[1].plot(bc_eqe_ref, "r--")
    # ax.plot(tc_eqe_new_arr)
    # bc_eqe_filtered = tc_trans_interp * bc_eqe_interp

    # # get the bandgap to cut-off eqe of bottom cell where eqe of the top cell is weak
    # eg_tc, sigma_tc = calc_Eg_Rau(tc_eqe_interp)
    # lam_cut_psc = photonenergy_to_wavelength(eg_tc + 2 * sigma_tc)

    # bc_eqe_filtered[tc_eqe_interp.index <= lam_cut_psc] = 0

    return tc_jscs/1e3, bc_jscs/1e3


def closest_list_entry(value, lst):
    closest = min(lst, key=lambda ele: abs(ele - value))
    return closest


class EnergyYield:
    """
    Class to run energy yield model
    """

    # standard data
    PVCPATH = os.path.dirname(os.path.dirname(__file__))
    DATAPATH = os.path.join(PVCPATH, "data", "")  # Data files here
    # datapath = os.path.abspath(os.path.relpath('../data/', start=__file__))
    # datapath = pvcpath.replace('/pvcircuit','/data/')
    ASTMFILE = os.path.join(DATAPATH, "ASTMG173.csv")

    def __init__(self, model, eg):
        self.model = model
        self.eg = eg

        self.date_start = None
        self.date_end = None
        self.resampler = None

        # environmental data
        self.daytime = None  # daytime vector
        self.ambient_temperature = None  # [degC]
        self.wind_speed = None  # [m/s]
        self.cell_temperature = None  # [degC]

        # spectral data
        self.wavelength = None
        self.spectra = None  # transpose for integration with QE

        # diode parameters
        self.top_cell_diode_params = None
        self.bottom_cell_diode_params = None
        # band gaps
        self.top_cell_eg_list = None
        self.bottom_cell_eg_list = None

        # EQE data
        self.top_cell_eqe = None
        self.bottom_cell_eqe = None

        # Transmittance data
        self.top_cell_transmittance = None

        # parameters to shift temperature dependent EQE
        self.si_eg_shift_p = [-6.47214956e-04, 1.01632828e00]
        self.si_sigma_shift_p = [0.00959188, 0.76558903]

        self.psc_eg_shift_p = [2.59551019e-04, 9.91138163e-01]
        self.psc_eg_shift_tsplit = 32
        self.psc_sigma_shift_p = [0.00358866, 0.90074156]

        # parameters to scale current accroding to band gap
        self.psc_jsc_eg = [-0.02434883, 0.06083894]
        self.si_jsc_eg = [0.02868742, -0.02959744]

        self.area = 1  # [cm^2]

        dfrefspec = pd.read_csv(self.ASTMFILE, index_col=0, header=2)

        self.ref_wvl = dfrefspec.index.to_numpy(dtype=np.float64, copy=True)

        #  # calculate from spectral proxy data only
        # self.SpecPower = pd.Series(np.trapz(spectra, x=wavelength), index=spectra.index) # optical power of each spectrum
        # self.RefPower = np.trapz(pvc.qe.refspec, x=self.ref_wvl, axis=0)  # optical power of each reference spectrum
        # self.TempCell = sandia_T(self.SpecPower, self.wind, self.temp)
        # self.inPower = self.SpecPower  # * self.NTime  # spectra power*(fractional time)
        # self.outPower = None
        # # construct a results dataframe for better handling
        # self.models=[]
        # self.operation_modes=[]
        # self.tandem_types=[]
        # self.EnergyIn = np.trapz(self.SpecPower, self.daytime.values.astype(np.int64)) / 1e9 / 60  # kWh/m2/yr

        # self.average_photon_energy = None # is calcluated when running calc_ape

    def load_meteo(self, date_start, date_end, resampler, force_reload:bool=False):
        self.date_start = date_start
        self.date_end = date_end
        self.resampler = resampler

        meteo_data = None
        spectra_class = None

        meteo_fp = r'./meteo_data/meteo_data.p'
        spectra_fp = r'./meteo_data/spectra_data.p'
        if os.path.exists(meteo_fp) and not force_reload:
            with open(meteo_fp, 'rb') as fid:
                meteo_data = pickle.load(fid)

        if os.path.exists(spectra_fp) and not force_reload:
            with open(spectra_fp, 'rb') as fid:
                spectra_class = pickle.load(fid)

        if not isinstance(meteo_data,Meteorological) and not isinstance(spectra_class,Spectra) or date_start not in meteo_data.data.index or date_end not in meteo_data.data.index or date_start.date() not in spectra_class.spectra.index.date or date_end.date() not in spectra_class.spectra.index.date:
            print("No file or date range out of previous data. Reloading meteo data and spectra")
            meteo_data = pvc.Meteorological.get_midc_environmental(date_start, date_end)
            spectra_class = pvc.Spectra.get_srrl_wiser_global_track(date_start, date_end)

            with open(meteo_fp, 'wb') as fid:
                pickle.dump(meteo_data,fid)

            with open(spectra_fp, 'wb') as fid:
                pickle.dump(spectra_class,fid)

        else:
            print("Stored meteo data and spectra reloaded")
            spectra_class.spectra = spectra_class.spectra.loc[date_start:date_end]
            meteo_data.data = meteo_data.data.loc[date_start:date_end]

        spectra_resampled = spectra_class.spectra.resample(resampler).median()
        meteo_resampled = meteo_data.data.resample(resampler).median()

        spectra_sync, meteo_sync = pvc.sync(spectra_resampled, meteo_resampled)

        # set class attributes
        self.daytime = spectra_sync.index
        self.wavelength = spectra_sync.columns.astype(np.int32)
        self.spectra = spectra_sync
        self.meteo = meteo_sync
        self.ambient_temperature = meteo_sync["Deck Dry Bulb Temp [deg C]"]
        self.wind_speed = meteo_sync["Avg Wind Speed @ 19ft [m/s]"]
        self.dir_diff_ratio = meteo_sync["Direct CHP1-1 [W/m^2]"] / meteo_sync["Diffuse CM22-1 (vent/cor) [W/m^2]"]

        # calculate class attributes
        self.irradiance = pd.Series(
            sc.trapz(self.spectra, x=self.wavelength), index=self.spectra.index
        )  # optical power of each spectrum
        self.cell_temperature = sandia_T(self.irradiance, self.wind_speed, self.ambient_temperature)
        self.average_photon_energy = self.get_ape()

    def filter_meteo(self):
        # remove nan
        # nan_filter = self.spectra.isna().any(axis=1) | self.meteo.isna().any(axis=1)
        # spectra_sync = self.spectra[~nan_filter]
        # meteo_sync = self.meteo[~nan_filter]

        # # remove negative spectral data

        # neg_spectra_filter = spectra_sync < -1e-2
        # neg_spectra_filter = (neg_spectra_filter.sum(axis=1) / spectra_sync.shape[1]) > 0.01
        # dir_diff_ratio = meteo_sync["Direct CHP1-1 [W/m^2]"] / meteo_sync["Diffuse CM22-1 (vent/cor) [W/m^2]"]
        # ape = calc_ape(spectra_sync)
        # general_filter = (ape > 0) & (ape < 2.2) & (~neg_spectra_filter) & (dir_diff_ratio > 0)

        # spectra_sync = spectra_sync[general_filter]
        # meteo_sync = meteo_sync[general_filter]

        spectra_sync = self.spectra
        meteo_sync = self.meteo

        # remove nan data
        nan_filter = self.spectra.isna().any(axis=1) | self.meteo.isna().any(axis=1)
        # remove negative spectral data
        neg_spectra_filter = spectra_sync < -1e-2
        neg_spectra_filter = (neg_spectra_filter.sum(axis=1) / spectra_sync.shape[1]) > 0.01
        # remove data where direct to diffuse ratio is negative
        dir_diff_ratio_filter = meteo_sync["Direct CHP1-1 [W/m^2]"] / meteo_sync["Diffuse CM22-1 (vent/cor) [W/m^2]"] <= 0
        # remove unusual low/high APE
        ape = calc_ape(spectra_sync)
        ape_filter = (ape <= 0) & (ape >= 2.2)
        # The spectral data is measured with two sensors. Cut off is a around 900 nm. Remove data where one sensor didn't measure correctly
        short_waves_integral = np.trapz(self.spectra.iloc[:,self.wavelength <= 900], x=self.wavelength[self.wavelength <= 900])
        long_waves_integral = np.trapz(self.spectra.iloc[:,self.wavelength > 900], x=self.wavelength[self.wavelength > 900])
        short_long_filter = (np.abs(short_waves_integral / long_waves_integral) < 1e-3) | (np.abs(long_waves_integral / short_waves_integral) < 1e-3)

        # short_long_ratio = short_waves_integral / long_waves_integral
        # short_long_ratio_filter = (short_long_ratio<1) & (long_waves_integral > 50)

        # general_filter = (ape > 0) & (ape < 2.2) & (~neg_spectra_filter) & (dir_diff_ratio > 0)
        combined_filter = ape_filter | (neg_spectra_filter) | (dir_diff_ratio_filter) | (nan_filter) | (short_long_filter)# | (short_long_ratio_filter)

        # spectra_sync = spectra_sync[general_filter]
        # meteo_sync = meteo_sync[general_filter]

        replace2nan_ids = spectra_sync[combined_filter].index
        spectra_sync.loc[replace2nan_ids] = np.nan
        meteo_sync.loc[replace2nan_ids] = np.nan

        nan_filter = self.spectra.isna().any(axis=1) | self.meteo.isna().any(axis=1)
        spectra_sync = spectra_sync[~nan_filter]
        meteo_sync = meteo_sync[~nan_filter]


    def load_top_cell_diode_params(self, fp_iv_data: dict):
        # ensure list
        # fp_iv_data = [fp_iv_data] if isinstance(fp_iv_data, str) else fp_iv_data

        # for fp_iv in fp_iv_data:
        #     diode_params.append(get_jv_params(fp_iv))

        tc_eg_list = []
        diode_params = []
        for k, v in fp_iv_data.items():
            tc_eg_list.append(k)
            diode_params.append(get_jv_params(v))

        diode_params = np.array(diode_params)

        self.top_cell_eg_list = tc_eg_list
        self.top_cell_diode_params = diode_params
        # self.psc_jsc_eg = np.polyfit(tc_eg_list, diode_params[:,0],1)

    def load_top_cell_eqe(self, fp_eqe_data: dict):
        reader_settings = {"index_col": 0, "usecols": [0, 1]}
        # eqe = pd.read_excel(fp_eqe_data, **reader_settings).dropna().squeeze("columns")
        # self.top_cell_eqe = eqe

        top_cell_eqes = {}
        for k, v in fp_eqe_data.items():
            top_cell_eqes[k] = pd.read_excel(v, **reader_settings).dropna().squeeze("columns")

        self.top_cell_eqes = top_cell_eqes

    def load_top_cell_transmittance(self, fp_transmittance_data):
        reader_settings = {"index_col": 0, "usecols": [0, 1]}
        # transmittance = pd.read_excel(fp_transmittance_data, **reader_settings).dropna().squeeze("columns")
        # self.top_cell_transmittance = transmittance

        transmittances = {}
        for k, v in fp_transmittance_data.items():
            transmittances[k] = pd.read_csv(v, **reader_settings).dropna().squeeze("columns")

        self.top_cell_transmittances = transmittances

    def load_top_cell_eqeT(self):
        ...

    def load_bottom_cell_diode_params(self, fp_iv_data: dict):
        # diode_params = get_jv_params(fp_iv_data)
        # diode_params = np.array(diode_params)

        # bc_eg_list = []
        # diode_params = []
        # for k,v in fp_iv_data.items():
        #     bc_eg_list.append(k)
        #     diode_params.append(get_jv_params(v))
        # self.bc_eg_list = bc_eg_list

        bc_eg_list = []
        diode_params = []
        for k, v in fp_iv_data.items():
            bc_eg_list.append(k)
            diode_params.append(get_jv_params(v, ilim=0.6))

        diode_params = np.array(diode_params)

        self.bottom_cell_eg_list = bc_eg_list
        diode_params = np.array(diode_params)
        self.bottom_cell_diode_params = diode_params

    def load_bottom_cell_eqe(self, fp_eqe_data):
        reader_settings = {"index_col": 0, "usecols": [0, 1]}
        # eqe = pd.read_excel(fp_eqe_data, **reader_settings).dropna().squeeze("columns")

        bottom_cell_eqes = {}
        for k, v in fp_eqe_data.items():
            bottom_cell_eqes[k] = pd.read_excel(v, **reader_settings).dropna().squeeze("columns")

        self.bottom_cell_eqes = bottom_cell_eqes

    def load_bottom_cell_eqeT(self):
        ...

    def make_mean_tandem(self):
        top_cell_joratios = []
        top_cell_n = []
        top_cell_rs = []
        top_cell_rsh = []

        bottom_cell_joratios = []
        bottom_cell_n = []
        bottom_cell_rs = []
        bottom_cell_rsh = []

        jo_scale = 1000
        A = 1  # [cm^2]
        TC = 25  # [degC]

        for i, top_cell_diode_params in enumerate(self.top_cell_diode_params):
            isc, i0, rs, rsh, nsVth = top_cell_diode_params
            j0 = i0 / A
            n = nsVth / pvc.junction.Vth(TC)

            tc_eqe = self.top_cell_eqes[self.top_cell_eg_list[i]]
            bc_eqe = self.bottom_cell_eqes[self.bottom_cell_eg_list[0]]

            # in case values are percentages
            if any(tc_eqe.values > 1):
                tc_eqe = tc_eqe.div(100)

            tc_eqe.index = np.round(tc_eqe.index.astype(np.double), 1)
            bc_eqe.index = np.round(bc_eqe.index.astype(np.double), 1)

            eqe_scaler = isc / A * 1e3 / integrate_eqe(tc_eqe)
            tc_eqe = tc_eqe * eqe_scaler

            comb_lam = tc_eqe.index.union(bc_eqe.index)
            tc_eqe = tc_eqe.reindex(index=comb_lam, fill_value=0.0).interpolate(method="index")

            eg = calc_Eg_Rau(tc_eqe)
            jdb = pvc.junction.Jdb(TC=TC, Eg=eg[0])
            top_cell_joratios.append(jo_scale * j0 / (jdb * jo_scale) ** (1.0 / n))

            top_cell_rs.append(rs * A)
            top_cell_rsh.append(rsh * A)
            top_cell_n.append(n)

        for i, bottom_cell_diode_params in enumerate(self.bottom_cell_diode_params):
            isc, i0, rs, rsh, nsVth = bottom_cell_diode_params
            j0 = i0 / A
            n = nsVth / pvc.junction.Vth(TC)
            eg = calc_Eg_Rau(self.bottom_cell_eqes[self.bottom_cell_eg_list[i]])
            jdb = pvc.junction.Jdb(TC=TC, Eg=eg[0])
            bottom_cell_joratios.append(jo_scale * j0 / (jdb * jo_scale) ** (1.0 / n))

            bottom_cell_rs.append(rs * A)
            bottom_cell_rsh.append(rsh * A)
            bottom_cell_n.append(n)

        self.model.set(name="PskSidev3T", Rz=0)
        self.model.top.set(
            J0ratio=[np.median(top_cell_joratios)],
            Gsh=1/np.median(top_cell_rsh),
            Rser=np.median(top_cell_rs),
            n=[np.median(top_cell_n)],
            beta=0,
        )
        self.model.bot.set(
            J0ratio=[np.median(bottom_cell_joratios)],
            Gsh=1/np.median(bottom_cell_rsh),
            Rser=np.median(bottom_cell_rs),
            n=[np.median(bottom_cell_n)],
            Eg=eg[0],
            beta=20,
        )

    def prepare_ey(self, tc_eg: float = 1.7, bc_eg: float = 1.12):
        tc_eg = closest_list_entry(tc_eg, self.top_cell_eg_list)
        bc_eg = closest_list_entry(bc_eg, self.bottom_cell_eg_list)

        tc_eqe = self.top_cell_eqes[tc_eg]
        # tc_trans = self.top_cell_transmittance[tc_eg]
        tc_trans = None
        bc_eqe = self.bottom_cell_eqes[bc_eg]

        if tc_trans is None:
            tc_trans = 1 - _normalize(tc_eqe)

        # in case values are percentages
        if any(tc_eqe.values > 1):
            tc_eqe = tc_eqe.div(100)
        if any(tc_trans.values > 1):
            tc_trans = tc_trans.div(100)
        if any(bc_eqe.values > 1):
            bc_eqe = bc_eqe.div(100)

        # # round in case wavelength has many floating points
        tc_eqe.index = np.round(tc_eqe.index.astype(np.double), 1)
        tc_trans.index = np.round(tc_trans.index.astype(np.double), 1)
        bc_eqe.index = np.round(bc_eqe.index.astype(np.double), 1)

        # interpolate with eqe's superseding transmission wavelengths
        comb_lam = tc_eqe.index.union(bc_eqe.index)
        df_psk_eqe_interp = tc_eqe.reindex(index=comb_lam, fill_value=0.0).interpolate(method="index")
        df_psk_trans_interp = tc_trans.reindex(index=comb_lam).interpolate(method="index")
        df_si_eqe_interp = bc_eqe.reindex(index=comb_lam, fill_value=0.0).interpolate(method="index")

        eqe_max_idx = df_psk_eqe_interp.idxmax()
        filter_idx = (df_psk_eqe_interp.values < 0.01) & (df_psk_trans_interp.index.values > eqe_max_idx)
        df_psk_trans_interp[filter_idx] = 1

        df_si_eqe_filtered = df_psk_trans_interp * df_si_eqe_interp

        # get the bandgap to cut-off eqe of bottom cell where eqe of the top cell is weak
        eg_psc, sigma_psc = calc_Eg_Rau(df_psk_eqe_interp)
        lam_cut_psc = photonenergy_to_wavelength(eg_psc + 2 * sigma_psc)

        df_si_eqe_filtered[df_psk_eqe_interp.index <= lam_cut_psc] = 0

        # commbine Psk and filter EQE
        df_tandem_eqe = pd.concat([df_psk_eqe_interp, df_si_eqe_filtered], axis=1)
        df_tandem_eqe.columns = ["Psk", "Si"]

        # scale tandem EQE
        df_psk_eqe_interp = df_tandem_eqe.iloc[:, 0]
        df_si_eqe_filtered = df_tandem_eqe.iloc[:, 1]

        # scale tandem EQE

        # scale eqe such that they correspond to linear current fit at 1 sun
        jext_top = np.polyval(self.psc_jsc_eg, tc_eg)
        jext_bot = np.polyval(self.si_jsc_eg, tc_eg)

        jext_top_eqe_1sun = integrate_eqe(df_psk_eqe_interp) / 1e3
        jext_bot_eqe_1sun = integrate_eqe(df_si_eqe_filtered) / 1e3

        jtop_scale = jext_top / jext_top_eqe_1sun
        jbot_scale = jext_bot / jext_bot_eqe_1sun

        df_psk_eqe_interp = df_psk_eqe_interp * jtop_scale
        df_si_eqe_filtered = df_si_eqe_filtered * jbot_scale

        # make sure scaling is correct
        assert np.isclose(integrate_eqe(df_psk_eqe_interp)[0] / 1e3, jext_top, rtol=1e-2)
        assert np.isclose(integrate_eqe(df_si_eqe_filtered)[0] / 1e3, jext_bot, rtol=1e-2)

        self.top_cell_eqe = df_psk_eqe_interp
        self.bottom_cell_eqe = df_si_eqe_interp
        self.bottom_cell_eqe_filtered = df_si_eqe_filtered

        eg_si, sigma_si = calc_Eg_Rau(df_si_eqe_interp)
        self.bottom_cell_egs = si_eg_shift(self.cell_temperature, eg_si, self.si_eg_shift_p)
        sigmas_si = si_sigma_shift(self.cell_temperature, sigma_si, self.si_sigma_shift_p)

        eg_psc, sigma_psc = calc_Eg_Rau(df_psk_eqe_interp)
        self.top_cell_egs = psc_eg_shift(self.cell_temperature, eg_psc, self.psc_eg_shift_p, self.psc_eg_shift_tsplit)
        sigmas_psc = psc_sigma_shift(self.cell_temperature, sigma_psc, self.psc_sigma_shift_p)

        self.model.top.set(Eg=eg_psc, Jext=jext_top)
        self.model.bot.set(Eg=eg_si, Jext=jext_bot)

        self.top_cell_currents, self.bottom_cell_currents = shift_eqe_tcbc(
            df_psk_eqe_interp,
            eg_psc,
            sigma_psc,
            [self.top_cell_egs],
            [sigmas_psc],
            df_si_eqe_filtered,
            eg_si,
            sigma_si,
            [self.bottom_cell_egs],
            [sigmas_si],
            self.spectra,
        )

    def get_ape(self):
        """
        Calcualtes the average photon energy (APE) of the spectra
        """
        phi = self.spectra * (self.wavelength * 1e-9) / constants.h / constants.c
        return np.trapz(x=self.wavelength, y=self.spectra.values) / constants.e / np.trapz(x=self.wavelength, y=phi.values)

    def cellbandgaps(self, EQE, TC=25):
        # subcell Egs for a given EQE class
        self.Jdbs, self.Egs = EQE.Jdb(TC)  # Eg from EQE same at all temperatures

    def cellcurrents(self, EQE, STC=False):
        # subcell currents and Egs and under self TMY for a given EQE class

        if STC:
            self.JscSTCs = EQE.Jint(pvc.qe.refspec) / 1000.0
        else:
            self.Jscs = EQE.Jint(self.spectra.T, xspec=self.wavelength) / 1000.0

    def cellEYeffMP(self):
        # max power of a cell under self TMY
        # self.Jscs and self.Egs must be calculate first using cellcurrents
        # Inputs
        # cell 'model' can be 'Multi2T' or 'Tandem3T'
        #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
        # Outputs
        # - EYeff energy yield efficiency = EY/YearlyEnergy
        # - EY energy yield of cell [kWh/m2/yr]

        # bot, top, ratio, type3T = cellmodeldesc(model, oper)  # ncells does not matter here

        # calc EY, etc

        # Split data into chunks for workers
        max_chunk_size = 200
        cpu_count = mp.cpu_count()
        chunk_ids = np.arange(len(self.top_cell_currents))
        chunk_size = min(len(chunk_ids) // cpu_count, max_chunk_size)

        chunks = [chunk_ids[i : i + chunk_size] for i in range(0, len(chunk_ids), chunk_size)]

        print(f"running with {cpu_count} pools")
        with tqdm(total=len(self.top_cell_currents), leave=True, desc=f"Processing 2T") as pbar:
            dev_list = np.array([copy.deepcopy(self.model) for _ in range(len(self.top_cell_currents))])
            with mp.Pool(cpu_count) as pool:

                def callback(*args):
                    # callback
                    pbar.update(len(args[0]))
                    return

                # Assign tasks to workers
                jobs = [
                    pool.apply_async(
                        _calc_yield_async_2T,
                        args=(
                            self.top_cell_currents[chunk],
                            self.bottom_cell_currents[chunk],
                            self.top_cell_egs[chunk],
                            self.bottom_cell_egs[chunk],
                            self.cell_temperature[chunk],
                            dev_list[chunk],
                        ),
                        callback=callback,
                    )
                    for chunk in chunks
                ]
                # Get results from workers
                power_2T = [item for job in jobs for item in job.get()]

            # For debugging without workers
            # jobs = [_calc_yield_async_2T(
            #             self.top_cell_currents[chunk],
            #             self.bottom_cell_currents[chunk],
            #             self.top_cell_egs[chunk],
            #             self.bottom_cell_egs[chunk],
            #             self.cell_temperature[chunk],
            #             dev_list[chunk],
            #         )
            #     for chunk in chunks
            # ]
            # # Get results from workers
            # results = [item for job in jobs for item in job]


        with tqdm(total=len(self.top_cell_currents), leave=True, desc=f"Processing 3T") as pbar:
            dev_list = np.array([copy.deepcopy(self.model) for _ in range(len(self.top_cell_currents))])
            with mp.Pool(cpu_count) as pool:

                def callback(*args):
                    # callback
                    pbar.update(len(args[0]))
                    return

                # Assign tasks to workers
                jobs = [
                    pool.apply_async(
                        _calc_yield_async_3T,
                        args=(
                            self.top_cell_currents[chunk],
                            self.bottom_cell_currents[chunk],
                            self.top_cell_egs[chunk],
                            self.bottom_cell_egs[chunk],
                            self.cell_temperature[chunk],
                            dev_list[chunk],
                        ),
                        callback=callback,
                    )
                    for chunk in chunks
                ]
                # Get results from workers
                power_3T = [item for job in jobs for item in job.get()]

        with tqdm(total=len(self.top_cell_currents), leave=True, desc=f"Processing 4T") as pbar:
            dev_list = np.array([copy.deepcopy(self.model) for _ in range(len(self.top_cell_currents))])
            with mp.Pool(cpu_count) as pool:

                def callback(*args):
                    # callback
                    pbar.update(len(args[0]))
                    return

                # Assign tasks to workers
                jobs = [
                    pool.apply_async(
                        _calc_yield_async_4T,
                        args=(
                            self.top_cell_currents[chunk],
                            self.bottom_cell_currents[chunk],
                            self.top_cell_egs[chunk],
                            self.bottom_cell_egs[chunk],
                            self.cell_temperature[chunk],
                            dev_list[chunk],
                        ),
                        callback=callback,
                    )
                    for chunk in chunks
                ]
                # Get results from workers
                power_4T = [item for job in jobs for item in job.get()]


        self.power_out = pd.DataFrame({"2T":power_2T, "3T":power_3T,"4T":power_4T})
        # self.energy_out = self.power_out.apply(lambda p: np.trapz(p, self.daytime.values.astype(np.int64)) / 1e9 / 3600,axis=0) # works only if there are no discontinueties in the data.
        self.energy_out = self.power_out.sum() * 60 / 3600 # energy per cell area so in this case [Wh/cm^2]
        self.energy_in = self.irradiance.sum() * 60 / 3600 # energy from irradiance, so in this case [Wh/m^2]
        self.ehe = self.energy_out / self.energy_in * 1e4 * 1e2 # *1e4 to account for areas [Wh/cm^2] and [Wh/m^2] and *1e2 for percent value

    def calc_ape(self):
        """
        Calcualtes the average photon energy (APE) of the spectra
        """

        phi = self.spectra * (self.wavelength * 1e-9) / constants.h / constants.c
        self.average_photon_energy = (
            np.trapz(x=self.wavelength, y=self.spectra.values) / constants.e / np.trapz(x=self.wavelength, y=phi.values)
        )

    def filter_ape(self, min_ape: float = 0, max_ape: float = 10):
        """
        filter the average photon energy (APE)

        Args:
            min_ape (float, optional): min value of th APE. Defaults to 0.
            max_ape (float, optional): max value of the APE. Defaults to 10.
        """
        if self.average_photon_energy is None:
            self.get_ape()

        self_copy = copy.deepcopy(self)
        ape_mask = (self_copy.average_photon_energy > min_ape) & (self_copy.average_photon_energy < max_ape)

        self_copy.daytime = self_copy.daytime[ape_mask]
        self_copy.average_photon_energy = self_copy.average_photon_energy[ape_mask]
        self_copy.spectra = self_copy.spectra[ape_mask]
        self_copy.irradiance = self_copy.irradiance[ape_mask]
        self_copy.cell_temperature = self_copy.cell_temperature[ape_mask]

        assert len(self_copy.spectra) == len(self_copy.SpecPower) == len(self_copy.TempCell) == len(self_copy.average_photon_energy)
        return self_copy

    def filter_spectra(self, min_spectra: float = 0, max_spectra: float = 10):
        """
        spectral data

        Args:
            min_spectra (float, optional): min value of the spectra. Defaults to 0.
            max_spectra (float, optional): max value of the spectra. Defaults to 10.
        """

        self_copy = copy.deepcopy(self)
        spectra_mask = (self_copy.spectra >= min_spectra).all(axis=1) & (self_copy.spectra < max_spectra).all(axis=1)
        self_copy.daytime = self_copy.daytime[spectra_mask]
        self_copy.average_photon_energy = self_copy.average_photon_energy[spectra_mask]
        self_copy.spectra = self_copy.spectra[spectra_mask]
        self_copy.SpecPower = self_copy.SpecPower[spectra_mask]
        self_copy.TempCell = self_copy.TempCell[spectra_mask]

        assert len(self.spectra) == len(self.SpecPower) == len(self.TempCell) == len(self.average_photon_energy)
        return self_copy

    def filter_custom(self, filter_array: bool):
        """
        Applys a custom filter ot the meteo data
        Args:
            filter_array (bool): Filter array to apply to the data
        """
        # assert len(filter_array) == len(self.spectra) == len(self.SpecPower) == len(self.TempCell)

        self_copy = copy.deepcopy(self)

        for attr_name in vars(self):
            if hasattr(getattr(self_copy, attr_name), "__len__"):
                attr = getattr(self_copy, attr_name)
                if len(attr) == len(filter_array):
                    setattr(self_copy, attr_name, attr[filter_array])

        # assert len(self.spectra) == len(self.SpecPower) == len(self.TempCell) == len(self.average_photon_energy)
        return self_copy

    def reindex(self, index: bool, method="nearest", tolerance=pd.Timedelta(seconds=30)):
        """
        Reindex according to indexer
        Args:
            filter_array (bool): Filter array to apply to the data
        """

        self_copy = copy.deepcopy(self)

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, pd.DataFrame) or isinstance(attr, pd.Series):
                setattr(self_copy, attr_name, attr.reindex(index=index, method=method, tolerance=tolerance))

        return self_copy

    def bin_data(self, nr_of_ape_bins, nr_of_temperature_bins, *, plot_binning=False, plot_spectra=False, export=False):
        ape_dataframe = pd.DataFrame(
            {
                "timestamps": self.spectra.index,
                "irradiance": self.irradiance.values,
                "cell_temperature": self.cell_temperature.values,
            },
            index=self.average_photon_energy,
        )
        ape_dataframe = ape_dataframe

        ape_dataframe_apesorted = ape_dataframe.sort_index()
        ape_dataframe_apesorted["powerCumSum"] = ape_dataframe_apesorted["irradiance"].cumsum()
        total_power = ape_dataframe_apesorted["powerCumSum"].iloc[-1]
        bin_power = total_power / nr_of_ape_bins

        if plot_binning:
            fig, ax = plt.subplots(1, constrained_layout=True)
            ax = ape_dataframe_apesorted["powerCumSum"].plot(color="b")
        ape_bin_edges = np.zeros(nr_of_ape_bins + 1)
        ape_bin_mids = np.zeros(nr_of_ape_bins)
        ape_bin_counts = np.zeros(nr_of_ape_bins)
        ape_bin_edges[0] = ape_dataframe.index.min() * 0.99
        for ape_bin in range(1, nr_of_ape_bins + 1):
            max_loop_ape = (
                ape_dataframe_apesorted["powerCumSum"][ape_dataframe_apesorted["powerCumSum"] <= ape_bin * bin_power]
                .dropna()
                .index[-1]
            )
            ape_bin_edges[ape_bin] = max_loop_ape
            ape_bin_mids[ape_bin - 1] = (ape_bin_edges[ape_bin - 1] + ape_bin_edges[ape_bin]) / 2

            ape_bin_counts[ape_bin - 1] = len(
                ape_dataframe_apesorted[
                    (ape_dataframe_apesorted.index > ape_bin_edges[ape_bin - 1])
                    & (ape_dataframe_apesorted.index <= ape_bin_edges[ape_bin])
                ]
            )

            if plot_binning:
                ax.plot([0, max_loop_ape], [ape_bin * bin_power, ape_bin * bin_power], "r--", lw=1)
                ax.plot([max_loop_ape, max_loop_ape], [0 * bin_power, ape_bin * bin_power], "r--", lw=1)

        if plot_binning:
            ax.set_xlabel("Average photon energy $E_\mathrm{ape}$ [eV]")
            ax.set_ylabel("Cumulative irradiance $G_{\mathrm{GNI}}$ [W m$^{-2}$]")
            ax.set_box_aspect(1)

            if export:
                # fig.savefig(r'C:\Users\rwitteck\OneDrive - NREL\Publications\2023\EHE-paper\Tex\figures\apebinning.pgf', format="pgf", backend="pgf")
                # fig.savefig(r'C:\Users\rwitteck\OneDrive - NREL\Publications\2023\EHE-paper\Tex\figures\apebinning.pdf', format="pdf", bbox_inches = 'tight')
                fig.savefig(
                    rf"C:\Users\rwitteck\OneDrive - NREL\Publications\2023\EHE-paper\Tex\figures\eape_binning.{EXPORT_FMT}",
                    format=EXPORT_FMT,
                    dpi=EXPORT_DPI,
                )

        self.ape_bin_mids = ape_bin_mids
        self.ape_bin_edges = ape_bin_edges
        self.ape_bin_counts = ape_bin_counts

        ape_spectra = pd.DataFrame(
            np.zeros([len(self.ape_bin_mids), len(self.wavelength)]),
            index=self.ape_bin_mids,
            columns=self.wavelength,
        )

        ape_temperatures = pd.DataFrame(
            np.zeros([len(self.ape_bin_mids), nr_of_temperature_bins]),
            index=self.ape_bin_mids,
        )

        ape_times = pd.DataFrame(
            np.zeros([len(self.ape_bin_mids), nr_of_temperature_bins]),
            index=self.ape_bin_mids,
        )

        irradiance = []
        total_bin_energy = []
        total_bin_times = []
        bin2plot = 5
        bin_spectra2plot = 0
        for idx_ape, ape_bin in enumerate(self.ape_bin_mids):
            # get a mask for the current APE bin range
            ape_bin_mask = (ape_dataframe.index > self.ape_bin_edges[idx_ape]) & (
                ape_dataframe.index <= self.ape_bin_edges[idx_ape + 1]
            )
            # filter the spectra with this mask to get all spectra for this APE range
            bin_spectra = self.spectra[ape_bin_mask]

            # get the timestamps and determine time deltas for each spectra
            bin_times = bin_spectra.index
            bin_time_deltas = bin_times.to_series().diff().astype(np.int64) / 1e9 # [s]
            _, day_idx = np.unique(bin_time_deltas.index.date, return_index=True)
            # assume that the time delta at the beginning of a day is 60 seconds to avoid night artifacts
            bin_time_deltas[day_idx] = 60 # [s]

            # set all to 60s
            bin_time_deltas[:] = 60 # [s]

            # weight all spectra with the time and divide by total time to get averaged spectra
            ape_spectra.iloc[idx_ape, :] = np.sum(bin_spectra * bin_time_deltas.values.reshape(-1, 1)) / np.sum(bin_time_deltas)

            # weight all spectra with the energy and divide by total energy to get averaged spectra
            bin_irradiances = pd.Series(np.trapz(x=bin_spectra.columns.astype(np.float32), y=bin_spectra), index=bin_spectra.index)
            bin_energies = bin_irradiances * bin_time_deltas
            total_bin_energy.append(np.sum(bin_energies) / 3600 / 1000)

            # ape_spectra.iloc[idx_ape,:] = np.sum(bin_spectra * bin_energies.values.reshape(-1,1)) / np.sum(bin_energies)

            irradiance.append(np.trapz(x=ape_spectra.columns.astype(np.float32), y=ape_spectra.iloc[idx_ape, :]))
            total_bin_times.append(bin_time_deltas.sum())

            bin_temps = pd.cut(ape_dataframe["cell_temperature"][ape_bin_mask], bins=nr_of_temperature_bins, include_lowest=True)
            bin_temps_unique = bin_temps.value_counts()
            for idx_t, bin_temp in enumerate(bin_temps_unique.index):
                bin_temp_mask = bin_temps == bin_temp

                if not any(bin_temp_mask):
                    # raise ValueError("Empty temperature mask")
                    continue

                ape_times.iloc[idx_ape, idx_t] = np.sum(bin_time_deltas[bin_temp_mask.values])
                ape_temperatures.loc[ape_bin, idx_t] = bin_temp.mid # bin temperature in each bin

            if bin2plot == idx_ape:
                bin_spectra2plot = bin_spectra
        # recalculate the ape of the spectra from the binning
        phi = ape_spectra * (ape_spectra.columns * 1e-9) / constants.h / constants.c
        ape_spectra = ape_spectra.set_index(
            np.trapz(x=ape_spectra.columns, y=ape_spectra) / constants.e / np.trapz(x=ape_spectra.columns, y=phi.values)
        )

        self.binned_spectra = ape_spectra
        self.binned_cell_temperature = ape_temperatures
        self.binned_times = ape_times

        if plot_spectra:
            fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 15))

            ape_dataframe_apesorted = ape_dataframe.sort_index()
            ape_dataframe_apesorted["powerCumSum"] = ape_dataframe_apesorted["irradiance"].cumsum() * 60 / 1000 / 3600
            total_power = ape_dataframe_apesorted["powerCumSum"].iloc[-1]
            bin_power = total_power / len(self.ape_bin_mids)

            axs[0].plot(ape_dataframe_apesorted["powerCumSum"], color="#0079C2", zorder=11)

            # i = 0
            # axs[0].plot([0, ape_bin_edges[i]], [i * bin_power, i * bin_power], "r--", lw=1)
            # axs[0].plot([ape_bin_edges[i], ape_bin_edges[i]], [0 * bin_power, i * bin_power], "r--", lw=1)
            for i in range(1, len(self.ape_bin_edges)):
                axs[0].plot([0, self.ape_bin_edges[i]], [i * bin_power, i * bin_power], "--", color="#FE6523", lw=0.6)
                axs[0].plot(
                    [self.ape_bin_edges[i], self.ape_bin_edges[i]], [0 * bin_power, i * bin_power], "--", color="#FE6523", lw=0.6
                )

            axs[0].set_box_aspect(1)
            axs[0].set_xlabel("Average photon energy $E_\mathrm{ape}$ [eV]")
            axs[0].set_ylabel("Cum. energy $E_{\mathrm{GNI}}$ [kWh m$^{-2}$]")
            axs[0].set_title("(a)", loc="center")

            axs[1].plot(
                ape_spectra.iloc[bin2plot, :].T / ape_spectra.iloc[bin2plot, :].T.max(),
                label=f"{self.ape_bin_mids[bin2plot]:.2f} eV",
                color="#0079C2",
            )
            axs[1].set_xlabel("Wavelength $\\lambda$ [nm]")
            # ax.set_ylabel("Spectral irradiance $G_{\mathrm{GNI},\\lambda}$ [Wm$^{-2}$nm$^{-1}$]")
            axs[1].set_ylabel("Norm. spec. irradiance $G_{\mathrm{GNI},\\lambda}$ [1]")
            axs[1].set_box_aspect(1)
            axs[1].set_title("(b)", loc="center")

            skips = bin_spectra2plot.shape[0] // 100
            skips = 20
            grey_spectra = axs[1].plot(
                bin_spectra2plot.iloc[::skips, :].T / bin_spectra2plot.iloc[::skips, :].T.max(),
                c=(0.0, 0.0, 0.0, 0.01),
                lw=1,
                zorder=0,
            )
            print(
                f"every {skips} of {bin_spectra2plot.shape[0]} {(bin_spectra2plot.shape[0]/skips) / bin_spectra2plot.shape[0] * 100}%"
            )
            print(f"{len(grey_spectra)} of {bin_spectra2plot.shape[0]} {len(grey_spectra) / bin_spectra2plot.shape[0] * 100}%")

            proxy_spectra = ape_spectra.iloc[bin2plot, :].T / ape_spectra.iloc[bin2plot, :].T.max()
            axs[1].annotate(
                "proxy spectrum",
                xy=(800, proxy_spectra.loc[800]),
                xycoords="data",
                xytext=(1000, 0.90),
                textcoords="data",
                fontsize="x-small",
                color="#0079C2",
                arrowprops=dict(arrowstyle="->", color="#0079C2"),
            )
            axs[1].annotate(
                "bin spectra",
                xy=(1000, 0.425),
                xycoords="data",
                xytext=(1100, 0.6),
                textcoords="data",
                fontsize="x-small",
                color="gray",
                arrowprops=dict(arrowstyle="->", color="gray"),
            )

            marker_scaler = 360
            irradiance = np.array(irradiance)
            total_bin_times = np.array(total_bin_times)
            total_bin_energy = np.array(total_bin_energy)
            totalEnergy = np.sum(irradiance * total_bin_times)
            # sp = ax[0].scatter(
            #     meteo_spectra_filtered.average_photon_energy, meteo_spectra_filtered.SpecPower, 1, dir_diff_ratio, marker=".", cmap="plasma"
            # )
            dir_diff_ratio = self.dir_diff_ratio
            dir_diff_sorter = np.argsort(dir_diff_ratio)
            sp = axs[2].scatter(
                self.average_photon_energy[dir_diff_sorter],
                self.irradiance[dir_diff_sorter],
                1,
                dir_diff_ratio[dir_diff_sorter],
                marker=".",
                cmap="plasma",
            )

            fig.colorbar(sp, ax=axs[2], label="direct normal / diffuse horizontal", fraction=0.05, pad=0.04)
            # ax[0].scatter(bin_group["ape"], bin_group["powerIn"], bin_group["powerIn"] * bin_group["totalTime"] / totalEnergy * marker_scaler, None, marker=".",  color='k')

            axs[2].scatter(
                ape_spectra.index,
                irradiance,
                irradiance * total_bin_times / totalEnergy * marker_scaler,
                None,
                marker=".",
                color=(0, 0.8, 0.2, 1),
            )
            # axs[2].plot(
            #     ape_spectra.index, irradiance, marker=".", color=(0, 0.8, 0.2, 1), mfc=(0, 0.8, 0.2, 1), mec=(0, 0.8, 0.2, 1)
            # )
            axs[2].set_box_aspect(1)

            axs[2].set_ylabel("Integrated spectrum $G_{\mathrm{GNI}}$ [W m$^{-2}$]")
            # ax[0].set_ylabel("Integrated spectrum $G_{\mathrm{GNI}}$ [W m$^{-2}$]")
            # ax[1].set_xlabel("Average photon energy $E_\mathrm{ape}$ [eV]")
            axs[2].set_xlabel("Average photon energy $E_\mathrm{ape}$ [eV]")
            axs[2].set_title("(c)", loc="center")
            # Increase the number of ticks
            axs[2].locator_params(axis="x", nbins=7)

            axs[2].set_xlim(1, 2.3)
            axs[2].set_ylim(0, 1600)

            # axs[2].legend(['Measured spectra', 'Proxy spectra'], loc='upper left', facecolor='none', handletextpad=-.5, frameon=False, bbox_to_anchor=(-.05, 1), borderpad=0, fontsize='x-small')
            axs[2].annotate(
                "measured spectra",
                xy=(1.4, 600),
                xycoords="data",
                xytext=(1.05, 1500),
                textcoords="data",
                fontsize="x-small",
                color=(181 / 255, 47 / 255, 138 / 255, 1),
                arrowprops=dict(arrowstyle="->", color=(181 / 255, 47 / 255, 138 / 255, 1)),
            )
            axs[2].annotate(
                "proxy spectra",
                xy=(ape_spectra.index[np.argmax(irradiance)], irradiance.max()),
                xycoords="data",
                xytext=(1.75, 1400),
                textcoords="data",
                fontsize="x-small",
                color=(0, 0.8, 0.2, 1),
                arrowprops=dict(arrowstyle="->", color=(0, 0.8, 0.2, 1)),
            )

            if export:
                diff_ape = np.diff(ape_bin_mids)
                if np.isclose(diff_ape.max(), diff_ape.min(), atol=0.1):
                    # fig.savefig(rf'C:\Users\rwitteck\OneDrive - NREL\Publications\2023\EHE-paper\Tex\figures\eape_spectra.pdf', format="pdf", bbox_inches = 'tight')
                    fig.savefig(
                        rf"C:\Users\rwitteck\OneDrive - NREL\Publications\2023\EHE-paper\Tex\figures\dape_spectra.{EXPORT_FMT}",
                        format=EXPORT_FMT,
                        dpi=EXPORT_DPI,
                        bbox_inches="tight",
                    )
                else:
                    fig.savefig(
                        rf"C:\Users\rwitteck\OneDrive - NREL\Publications\2023\EHE-paper\Tex\figures\eape_spectra.{EXPORT_FMT}",
                        format=EXPORT_FMT,
                        dpi=EXPORT_DPI,
                        bbox_inches="tight",
                    )

    def run_ey_binning(self):

        # Convert binned spectra and temperature to conform to calculation when loading the spectral and meteorological data.
        spectra = pd.DataFrame(
            self.binned_spectra.values.repeat(self.binned_cell_temperature.shape[1], axis=0),
            index=self.binned_spectra.index.repeat(self.binned_cell_temperature.shape[1]), columns = self.wavelength
        )
        # self.cell_temperature = pd.DataFrame(self.binned_cell_temperature.values.flatten(),index = self.binned_spectra.index.repeat(self.binned_cell_temperature.shape[1]))
        cell_temperature = self.binned_cell_temperature.values.flatten()

        irradiance = pd.Series(np.trapz(spectra, x=self.wavelength), index=spectra.index)  # optical power of each spectrum

        eg_si, sigma_si = calc_Eg_Rau(self.bottom_cell_eqe)
        self.bottom_cell_egs = si_eg_shift(cell_temperature, eg_si, self.si_eg_shift_p)
        sigmas_si = si_sigma_shift(cell_temperature, sigma_si, self.si_sigma_shift_p)

        eg_psc, sigma_psc = calc_Eg_Rau(self.top_cell_eqe)
        self.top_cell_egs = psc_eg_shift(cell_temperature, eg_psc, self.psc_eg_shift_p, self.psc_eg_shift_tsplit)
        sigmas_psc = psc_sigma_shift(cell_temperature, sigma_psc, self.psc_sigma_shift_p)

        self.top_cell_currents, self.bottom_cell_currents = shift_eqe_tcbc(
            self.top_cell_eqe,
            eg_psc,
            sigma_psc,
            [self.top_cell_egs],
            [sigmas_psc],
            self.bottom_cell_eqe_filtered,
            eg_si,
            sigma_si,
            [self.bottom_cell_egs],
            [sigmas_si],
            spectra,
        )

        # Split data into chunks for workers
        max_chunk_size = 200
        cpu_count = mp.cpu_count()
        chunk_ids = np.arange(len(self.top_cell_currents))
        chunk_size = min(max(len(chunk_ids) // cpu_count, 1), max_chunk_size) # Either max(len(chunk_ids) // cpu_count, 1) to prevent 0 chunk_size and not bigger than max_chunk_size to prevent running into memory handling issues

        chunks = [chunk_ids[i : i + chunk_size] for i in range(0, len(chunk_ids), chunk_size)]

        print(f"running binning with {cpu_count} pools")
        with tqdm(total=len(self.top_cell_currents), leave=True, desc=f"Processing 2T binning") as pbar:
            dev_list = np.array([copy.deepcopy(self.model) for _ in range(len(self.top_cell_currents))])
            with mp.Pool(cpu_count) as pool:

                def callback(*args):
                    # callback
                    pbar.update(len(args[0]))
                    return

                # Assign tasks to workers
                jobs = [
                    pool.apply_async(
                        _calc_yield_async_2T,
                        args=(
                            self.top_cell_currents[chunk],
                            self.bottom_cell_currents[chunk],
                            self.top_cell_egs[chunk],
                            self.bottom_cell_egs[chunk],
                            cell_temperature[chunk],
                            dev_list[chunk],
                        ),
                        callback=callback,
                    )
                    for chunk in chunks
                ]
                # Get results from workers
                results2T = np.array([item for job in jobs for item in job.get()])

        with tqdm(total=len(self.top_cell_currents), leave=True, desc=f"Processing 3T binning") as pbar:
            dev_list = np.array([copy.deepcopy(self.model) for _ in range(len(self.top_cell_currents))])
            with mp.Pool(cpu_count) as pool:

                def callback(*args):
                    # callback
                    pbar.update(len(args[0]))
                    return

                # Assign tasks to workers
                jobs = [
                    pool.apply_async(
                        _calc_yield_async_3T,
                        args=(
                            self.top_cell_currents[chunk],
                            self.bottom_cell_currents[chunk],
                            self.top_cell_egs[chunk],
                            self.bottom_cell_egs[chunk],
                            cell_temperature[chunk],
                            dev_list[chunk],
                        ),
                        callback=callback,
                    )
                    for chunk in chunks
                ]
                # Get results from workers
                results3T = np.array([item for job in jobs for item in job.get()])

        with tqdm(total=len(self.top_cell_currents), leave=True, desc=f"Processing 4T binning") as pbar:
            dev_list = np.array([copy.deepcopy(self.model) for _ in range(len(self.top_cell_currents))])
            with mp.Pool(cpu_count) as pool:

                def callback(*args):
                    # callback
                    pbar.update(len(args[0]))
                    return

                # Assign tasks to workers
                jobs = [
                    pool.apply_async(
                        _calc_yield_async_4T,
                        args=(
                            self.top_cell_currents[chunk],
                            self.bottom_cell_currents[chunk],
                            self.top_cell_egs[chunk],
                            self.bottom_cell_egs[chunk],
                            cell_temperature[chunk],
                            dev_list[chunk],
                        ),
                        callback=callback,
                    )
                    for chunk in chunks
                ]
                # Get results from workers
                results4T = np.array([item for job in jobs for item in job.get()])

            # For debugging without workers
            # jobs = [_calc_yield_async_2T(
            #             self.top_cell_currents[chunk],
            #             self.bottom_cell_currents[chunk],
            #             self.top_cell_egs[chunk],
            #             self.bottom_cell_egs[chunk],
            #             cell_temperature[chunk],
            #             dev_list[chunk],
            #         )
            #     for chunk in chunks
            # ]
            # # Get results from workers
            # results = [item for job in jobs for item in job]

        # binning_results = pd.DataFrame(np.zeros([len(tc_eg_list) * nr_of_ape_bins * nr_of_temp_bins * len(rs_top_list) * len(jsc_bot_loss_list), len(cols)]), columns=cols)
        # self.ey_results = pd.DataFrame([[eg_psc]*len(results),results], columns=cols)

        self.binning_results = pd.DataFrame(
            {
                "TCEg": self.top_cell_egs,
                "BCEg": self.bottom_cell_egs,
                "BinTime": self.binned_times.values.flatten(),
                "BinAPE": spectra.index,
                "BinCellTemp" : cell_temperature,
                "BinPowerIn" : irradiance.values,
                "BinPower2T": results2T,
                "BinPower3T": results3T,
                "BinPower4T": results4T,
                "BinEnergy2T": np.sum(results2T * self.binned_times.values.flatten()) / 3600, # [Wh]
                "BinEnergy3T": np.sum(results3T * self.binned_times.values.flatten()) / 3600, # [Wh]
                "BinEnergy4T": np.sum(results4T * self.binned_times.values.flatten()) / 3600, # [Wh]
                "BinEnergyIn": np.sum(irradiance.values * self.binned_times.values.flatten()),
                "BinEHE2T": np.sum(results2T) / np.sum(irradiance.values),
                "BinEHE3T": np.sum(results3T) / np.sum(irradiance.values),
                "BinEHE4T": np.sum(results4T) / np.sum(irradiance.values),
            }
        )

    # def run_ey_binning(self):
    #     cols = [
    #         "Eg",
    #         "EgsPsc",
    #         "EgsSi",
    #         "totalTime",
    #         "binCount",
    #         "ape",
    #         "temp",
    #         "ape_bin_width",
    #         "temp_bin_width",
    #         "powerIn",
    #         "power2Tout",
    #         "power4Tout",
    #         "psc_jscs",
    #         "psc_vocs",
    #         "psc_ffs",
    #         "si_jscs",
    #         "si_vocs",
    #         "si_ffs",
    #         "rs_top",
    #         "jsc_bot_loss",
    #     ]

    #     # Additional losses for 4T
    #     # additional top series resistance
    #     rs_top_list = [0, 3.731540]  # Ohm c^2
    #     rs_top_list = [0, 3.731540 / 2]  # Ohm c^2
    #     jsc_bot_loss_list = [1, 0.95]  # percentage loss

    #     rs_top_list = [1.86]  # Ohm c^2
    #     jsc_bot_loss_list = [1 - 0.022]  # percentage loss

    #     rs_top_list = [0]  # Ohm c^2
    #     jsc_bot_loss_list = [1]  # percentage loss

    #     # binning_results = pd.DataFrame(np.zeros([len(tc_eg_list) * nr_of_ape_bins * nr_of_temp_bins * len(rs_top_list) * len(jsc_bot_loss_list), len(cols)]), columns=cols)
    #     ey_results = pd.DataFrame(
    #         np.zeros([np.prod(self.binned_cell_temperature.shape) * len(rs_top_list) * len(jsc_bot_loss_list), len(cols)]), columns=cols
    #     )

    #     row_counter = 0

    #     irradiance = []
    #     inital_rs_top = self.model.top.Rser

    #     for rs_top in rs_top_list:
    #         for jsc_bot_loss in jsc_bot_loss_list:
    #             for idx_ape in trange(len(self.binned_spectra)):
    #                 # get a mask for the current APE bin range
    #                 bin_spec = self.binned_spectra.iloc[idx_ape,:]
    #                 irradiance.append(np.trapz(x=bin_spec.index, y=bin_spec.values))

    #                 # bin temperature in each bin
    #                 # apebin_temps = ape_dataframe["TempCell"][ape_bin_mask]
    #                 for idx_temp,bin_temp in enumerate(self.binned_cell_temperature.iloc[idx_ape,:]):

    #                     bin_temp_times = self.binned_times.iloc[idx_ape,idx_temp]

    #                     bin_temp = np.array([bin_temp])
    #                     # for id_eg in range(len(tc_eg_list)):
    #                     for id_eg in [3]:
    #                         eg = self.top_cell_eg_list[id_eg]

    #                         eg_si, sigma_si = calc_Eg_Rau(self.bottom_cell_eqe, fit_gaussian=True)
    #                         egs_si = si_eg_shift(bin_temp, eg_si, self.si_eg_shift_p)
    #                         sigmas_si = si_sigma_shift(bin_temp, sigma_si, self.si_sigma_shift_p)

    #                         eg_psc, sigma_psc = calc_Eg_Rau(self.top_cell_eqe)
    #                         egs_psc = psc_eg_shift(bin_temp, eg_psc, self.psc_eg_shift_p, self.psc_eg_shift_tsplit)
    #                         sigmas_psc = psc_sigma_shift(bin_temp, sigma_psc, self.psc_sigma_shift_p)

    #                         # band gap shift
    #                         tc_current, bc_current = shift_eqe_tcbc(self.bottom_cell_eqe, eg_psc, sigma_psc, [egs_psc], [sigmas_psc], self.bottom_cell_eqe, eg_si, sigma_si, [egs_si], [sigmas_si], pd.DataFrame(bin_spec).T)

    #                         self.model.top.set(Jext=tc_current / 1e3, TC=bin_temp, Eg=egs_psc)
    #                         self.model.bot.set(Jext=bc_current / 1e3, TC=bin_temp, Eg=egs_si)

    #                         self.model.top.set(Rser=inital_rs_top)
    #                         _, iv3t = self.model.CM()
    #                         power2T = iv3t.Ptot[0]

    #                         self.model.top.set(Rser=inital_rs_top + rs_top)
    #                         self.model.bot.set(Jext=self.bottom_cell_currents / 1e3 * jsc_bot_loss, TC=bin_temp)

    #                         iv3t = self.model.MPP()
    #                         power4T = iv3t.Ptot[0]

    #                         ptop = np.abs(iv3t.Ito * iv3t.Vzt)[0]
    #                         pbot = np.abs(iv3t.Iro * iv3t.Vrz)[0]

    #                         iv3t = self.model.Isc3()
    #                         isc_top = np.abs(iv3t.Ito[0])
    #                         isc_bot = np.abs(iv3t.Iro[0])

    #                         iv3t = self.model.Voc3()
    #                         voc_top = np.abs(iv3t.Vzt[0])
    #                         voc_bot = np.abs(iv3t.Vrz[0])

    #                         ey_results.at[row_counter, "Eg"] = eg
    #                         ey_results.at[row_counter, "EgsPsc"] = self.top_cell_egs
    #                         ey_results.at[row_counter, "EgsSi"] = self.bottom_cell_egs
    #                         ey_results.at[row_counter, "totalTime"] = np.sum(bin_temp_times)  # [s]
    #                         ey_results.at[row_counter, "binCount"] = bin_temps_unique[bin_temp]
    #                         # ey_results.at[row_counter, "ape"] = ape_bin_mids[idx_ape]
    #                         ey_results.at[row_counter, "ape"] = self.ape_spectra.index[idx_ape]
    #                         ey_results.at[row_counter, "temp"] = bin_temp.mid
    #                         ey_results.at[row_counter, "ape_bin_width"] = (
    #                             self.ape_bin_edges[idx_ape + 1] - self.ape_bin_edges[idx_ape]
    #                         )
    #                         ey_results.at[row_counter, "temp_bin_width"] = bin_temp
    #                         ey_results.at[row_counter, "powerIn"] = powerIn
    #                         ey_results.at[row_counter, "power2Tout"] = power2T
    #                         ey_results.at[row_counter, "power4Tout"] = power4T
    #                         ey_results.at[row_counter, "psc_jscs"] = self.top_cell_currents
    #                         ey_results.at[row_counter, "psc_vocs"] = voc_top
    #                         ey_results.at[row_counter, "psc_ffs"] = ptop / isc_top / voc_top
    #                         ey_results.at[row_counter, "si_jscs"] = self.bottom_cell_currents
    #                         ey_results.at[row_counter, "si_vocs"] = voc_bot
    #                         ey_results.at[row_counter, "si_ffs"] = pbot / isc_bot / voc_bot

    #                         ey_results.at[row_counter, "rs_top"] = rs_top
    #                         ey_results.at[row_counter, "jsc_bot_loss"] = jsc_bot_loss
    #                         row_counter += 1
