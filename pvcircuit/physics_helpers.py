"""
The ``sde`` module contains functions to fit the single diode equation.

Function names should follow the pattern "fit_" + fitting method.

"""

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Minimizer, Parameters, report_fit

# set constant for numpy.linalg.lstsq parameter rcond
# rcond=-1 for numpy<1.14, rcond=None for numpy>=1.14
# TODO remove after minimum numpy version >= 1.14
minor = int(np.__version__.split('.')[1])
if minor < 14:
    RCOND = -1
else:
    RCOND = None


def fit_sandia_simple(voltage, current, v_oc=None, i_sc=None, v_mp_i_mp=None,
                      vlim=0.2, ilim=0.1):
    r"""
    Fits the single diode equation (SDE) to an IV curve.

    Parameters
    ----------
    voltage : ndarray
        1D array of `float` type containing voltage at each point on the IV
        curve, increasing from 0 to ``v_oc`` inclusive. [V]

    current : ndarray
        1D array of `float` type containing current at each point on the IV
        curve, from ``i_sc`` to 0 inclusive. [A]

    v_oc : float, default None
        Open circuit voltage. If not provided, ``v_oc`` is taken as the
        last point in the ``voltage`` array. [V]

    i_sc : float, default None
        Short circuit current. If not provided, ``i_sc`` is taken as the
        first point in the ``current`` array. [A]

    v_mp_i_mp : tuple of float, default None
        Voltage, current at maximum power point. If not provided, the maximum
        power point is found at the maximum of ``voltage`` \times ``current``.
        [V], [A]

    vlim : float, default 0.2
        Defines portion of IV curve where the exponential term in the single
        diode equation can be neglected, i.e.
        ``voltage`` <= ``vlim`` x ``v_oc``. [V]

    ilim : float, default 0.1
        Defines portion of the IV curve where the exponential term in the
        single diode equation is significant, approximately defined by
        ``current`` < (1 - ``ilim``) x ``i_sc``. [A]

    Returns
    -------
    photocurrent : float
        photocurrent [A]
    saturation_current : float
        dark (saturation) current [A]
    resistance_series : float
        series resistance [ohm]
    resistance_shunt : float
        shunt (parallel) resistance [ohm]
    nNsVth : float
        product of thermal voltage ``Vth`` [V], diode ideality factor
        ``n``, and number of series cells ``Ns``. [V]

    Raises
    ------
    RuntimeError if parameter extraction is not successful.

    Notes
    -----
    Inputs ``voltage``, ``current``, ``v_oc``, ``i_sc`` and ``v_mp_i_mp`` are
    assumed to be from a single IV curve at constant irradiance and cell
    temperature.

    :py:func:`fit_sandia_simple` obtains values for the five parameters for
    the single diode equation [1]_:

    .. math::

        I = I_{L} - I_{0} (\exp \frac{V + I R_{s}}{nNsVth} - 1)
        - \frac{V + I R_{s}}{R_{sh}}

    See :py:func:`pvlib.pvsystem.singlediode` for definition of the parameters.

    The extraction method [2]_ proceeds in six steps.

    1. In the single diode equation, replace :math:`R_{sh} = 1/G_{p}` and
       re-arrange

    .. math::

        I = \frac{I_{L}}{1 + G_{p} R_{s}} - \frac{G_{p} V}{1 + G_{p} R_{s}}
        - \frac{I_{0}}{1 + G_{p} R_{s}} (\exp(\frac{V + I R_{s}}{nN_sV_{th}})
        - 1)

    2. The linear portion of the IV curve is defined as
       :math:`V \le vlim \times v_{oc}`. Over this portion of the IV curve,

    .. math::

        \frac{I_{0}}{1 + G_{p} R_{s}} (\exp(\frac{V + I R_{s}}{nN_sV_{th}})
        - 1) \approx 0

    3. Fit the linear portion of the IV curve with a line.

    .. math::

        I &\approx \frac{I_{L}}{1 + G_{p} R_{s}}
        - \frac{G_{p}}{1 + G_{p}R_{s}} V
        &= \beta_{0} + \beta_{1} V

    4. The exponential portion of the IV curve is defined by
       :math:`\beta_{0} + \beta_{1} \times V - I > ilim \times i_{sc}`.
       Over this portion of the curve,
       :math:`\exp((V + IR_s)/{nN_sV_{th}}) \gg 1` so that

    .. math::

        \exp(\frac{V + I R_{s}}{nN_sV_{th}}) - 1 \approx
        \exp(\frac{V + I R_{s}}{nN_sV_{th}})

    5. Fit the exponential portion of the IV curve.

    .. math::

        \log(\beta_{0} - \beta_{1} V - I)
        &\approx \log(\frac{I_{0}}{1 + G_{p} R_{s}} + \frac{V}{nN_sV_{th}}
        + \frac{I R_{s}}{nN_sV_{th}}) \\
        &= \beta_{2} + \beta_{3} V + \beta_{4} I

    6. Calculate values for ``IL, I0, Rs, Rsh,`` and ``nNsVth`` from the
       regression coefficents :math:`\beta_{0}, \beta_{1}, \beta_{3}` and
       :math:`\beta_{4}`.


    References
    ----------
    .. [1] S.R. Wenham, M.A. Green, M.E. Watt, "Applied Photovoltaics" ISBN
       0 86758 909 4
    .. [2] C. B. Jones, C. W. Hansen, "Single Diode Parameter Extraction from
       In-Field Photovoltaic I-V Curves on a Single Board Computer", 46th IEEE
       Photovoltaic Specialist Conference, Chicago, IL, 2019
    """

    # If not provided, extract v_oc, i_sc, v_mp and i_mp from the IV curve data
    if v_oc is None:
        v_oc = voltage[-1]
    if i_sc is None:
        i_sc = current[0]
    if v_mp_i_mp is not None:
        v_mp, i_mp = v_mp_i_mp
    else:
        v_mp, i_mp = _find_mp(voltage, current)

    # Find beta0 and beta1 from linear portion of the IV curve
    beta0, beta1 = _sandia_beta0_beta1(voltage, current, vlim, v_oc)

    # Find beta3 and beta4 from the exponential portion of the IV curve
    beta3, beta4 = _sandia_beta3_beta4(voltage, current, beta0, beta1, ilim,
                                       i_sc, v_oc)

    # calculate single diode parameters from regression coefficients
    return _sandia_simple_params(beta0, beta1, beta3, beta4, v_mp, i_mp, v_oc)


def _find_mp(voltage, current):
    """
    Finds voltage and current at maximum power point.

    Parameters
    ----------
    voltage : ndarray
        1D array containing voltage at each point on the IV curve, increasing
        from 0 to v_oc inclusive, of `float` type. [V]

    current : ndarray
        1D array containing current at each point on the IV curve, decreasing
        from i_sc to 0 inclusive, of `float` type. [A]

    Returns
    -------
    v_mp, i_mp : tuple
        voltage ``v_mp`` and current ``i_mp`` at the maximum power point. [V],
        [A]
    """
    p = voltage * current
    idx = np.argmax(p)
    return voltage[idx], current[idx]


def _sandia_beta0_beta1(v, i, vlim, v_oc):
    # Used by fit_sandia_simple.
    # Get intercept and slope of linear portion of IV curve.
    # Start with V =< vlim * v_oc, extend by adding points until slope is
    # negative (downward).
    beta0 = np.nan
    beta1 = np.nan
    first_idx = np.searchsorted(v, vlim * v_oc)
    for idx in range(first_idx, len(v)):
        coef = np.polyfit(v[:idx], i[:idx], deg=1)
        if coef[0] < 0:
            # intercept term
            beta0 = coef[1].item()
            # sign change of slope to get positive parameter value
            beta1 = -coef[0].item()
            break
    if any(np.isnan([beta0, beta1])):
        raise RuntimeError("Parameter extraction failed: beta0={}, beta1={}"
                           .format(beta0, beta1))
    else:
        return beta0, beta1

# define objective function: returns the array to be minimized
def fcn2min(params, x, data):
    """Model a decaying sine wave and subtract data."""
    beta2 = params['beta2']
    beta3 = params['beta3']
    beta4 = params['beta4']
    model = beta2 * x[:,0] + beta3 * x[:,1] +  beta4 * x[:,2]
    return model - data



def _sandia_beta3_beta4(voltage, current, beta0, beta1, ilim, i_sc, v_oc):
    # Used by fit_sde_sandia.
    # Subtract the IV curve from the linear fit.
    ilim = 0.6
    vlim = 0.95
    y = beta0 - beta1 * voltage - current
    x = np.array([np.ones_like(voltage), voltage, current]).T
    # Select points where y > ilim * i_sc to regress log(y) onto x
    # idx = (y > ilim * i_sc)
    idx = (voltage > vlim * v_oc) & (y > 0)
    result = np.linalg.lstsq(x[idx], np.log(y[idx]), rcond=RCOND)
    coef = result[0]
    beta3 = coef[1].item()
    beta4 = coef[2].item()

    # while beta4 < 0 and ilim > 0:
    #     ilim -=0.01
    #     idx = (y > ilim * i_sc)
    #     result = np.linalg.lstsq(x[idx], np.log(y[idx]), rcond=RCOND)
    #     coef = result[0]
    #     beta3 = coef[1].item()
    #     beta4 = coef[2].item()


    # fig,ax = plt.subplots()
    # ax.plot(voltage[idx], np.log(y[idx]),"x")
    # ax.plot(voltage[idx], coef[0] + coef[1] * voltage[idx] + coef[2] * current[idx], "--")
    rs_old = np.inf
    rs = beta4 / beta3
    eps = 1e-2

    rs_diff_best = np.inf
    beta4_best = -1
    beta3_best = 1

    while np.abs(rs_old/rs - 1) > eps and vlim > 0.2:
    # while np.abs(rs_old/rs - 1) > eps and ilim > 0:
        ilim -=0.001
        vlim -=0.001
        # if ilim < 0:
        #     print("ilim error")
        rs_old = rs
        # fit iv
        # idx = (y > ilim * i_sc)
        idx = (voltage > vlim * v_oc) & (y > 0)

        try:
            # np.seterr(all='raise')

            result = np.linalg.lstsq(x[idx], np.log(y[idx]), rcond=RCOND)

            # np.seterr(all='warn')

            coef = result[0]
            beta3 = coef[1].item()
            beta4 = coef[2].item()
            rs = beta4 / beta3
            if np.abs(rs_old/rs - 1) < rs_diff_best and rs >= 0:
                beta3_best = beta3
                beta4_best = beta4
        except np.linalg.LinAlgError:
            rs_old = 0
            rs = 1

        if rs <= 0:
            rs_old = 0
            rs = 1
            # beta4 = beta4old * eps * 0.01

            # ax.plot(voltage[idx], current[idx],".")


        if vlim < 0.2:
            print("use best so far")
            beta3 = beta3_best
            beta4 = beta4_best
    # create a set of Parameters
    # params = Parameters()
    # params.add('beta2', value=1)
    # params.add('beta3', value=1)
    # params.add('beta4', value=1, min = 0)

    # do fit, here with the default leastsq algorithm
    # minner = Minimizer(fcn2min, params, fcn_args=(x[idx], np.log(y[idx])))
    # result = minner.minimize()
    # report_fit(result)
    # beta3 = result.params['beta3'].value
    # beta4 = result.params['beta4'].value

    if any(np.isnan([beta3, beta4])):
        raise RuntimeError("Parameter extraction failed: beta3={}, beta4={}"
                           .format(beta3, beta4))
    else:
        return beta3, beta4


def _sandia_simple_params(beta0, beta1, beta3, beta4, v_mp, i_mp, v_oc):
    # Used by fit_sandia_simple.
    nNsVth = 1.0 / beta3
    rs = beta4 / beta3
    gsh = beta1 / (1.0 - rs * beta1)
    rsh = 1.0 / gsh
    iph = (1 + gsh * rs) * beta0
    # calculate I0
    io_vmp = _calc_I0(v_mp, i_mp, iph, gsh, rs, nNsVth)
    io_voc = _calc_I0(v_oc, 0, iph, gsh, rs, nNsVth)
    if any(np.isnan([io_vmp, io_voc])) or ((io_vmp <= 0) and (io_voc <= 0)):
        raise RuntimeError("Parameter extraction failed: I0 is undetermined.")
    elif (io_vmp > 0) and (io_voc > 0):
        io = 0.5 * (io_vmp + io_voc)
    elif (io_vmp > 0):
        io = io_vmp
    else:  # io_voc > 0
        io = io_voc
    return iph, io, rs, rsh, nNsVth




def _calc_I0(voltage, current, iph, gsh, rs, nNsVth):
    return (iph - current - gsh * (voltage + rs * current)) / \
        np.expm1((voltage + rs * current) / nNsVth)



# def _sandia_beta3_beta4(voltage, current, beta0, beta1, ilim, i_sc, v_oc):
#     # Used by fit_sde_sandia.
#     # Subtract the IV curve from the linear fit.
#     idx = len(voltage)-3

#     y = beta0 - beta1 * voltage - current

#     idx_min = np.where(y < 0)[0].max() + 1
#     x = np.array([np.ones_like(voltage), voltage, current]).T
#     # Select points where y > ilim * i_sc to regress log(y) onto x
#     # idx = (y > ilim * i_sc)
#     # idx = (voltage > vlim * v_oc) & (y > 0)
#     result = np.linalg.lstsq(x[idx:], np.log(y[idx:]), rcond=RCOND)
#     coef = result[0]
#     beta3 = coef[1].item()
#     beta4 = coef[2].item()

#     # while beta4 < 0 and ilim > 0:
#     #     ilim -=0.01
#     #     idx = (y > ilim * i_sc)
#     #     result = np.linalg.lstsq(x[idx], np.log(y[idx]), rcond=RCOND)
#     #     coef = result[0]
#     #     beta3 = coef[1].item()
#     #     beta4 = coef[2].item()


#     # fig,ax = plt.subplots()
#     # ax.plot(voltage[idx], np.log(y[idx]),"x")
#     # ax.plot(voltage[idx], coef[0] + coef[1] * voltage[idx] + coef[2] * current[idx], "--")
#     rs_old = np.inf
#     rs = beta4 / beta3
#     eps = 1e-5

#     rs_diff_best = np.inf
#     beta4_best = -1
#     beta3_best = 1

#     while np.abs(rs_old/rs - 1) > eps and idx > idx_min:
#     # while np.abs(rs_old/rs - 1) > eps and ilim > 0:
#         idx -= 1
#         # if ilim < 0:
#         #     print("ilim error")
#         rs_old = rs
#         # fit iv
#         # idx = (y > ilim * i_sc)

#         try:
#             np.seterr(all='raise')

#             result = np.linalg.lstsq(x[idx:], np.log(y[idx:]), rcond=RCOND)

#             np.seterr(all='warn')

#             coef = result[0]
#             beta3 = coef[1].item()
#             beta4 = coef[2].item()
#             rs = beta4 / beta3
#             if np.abs(rs_old/rs - 1) < rs_diff_best and beta3 >= 0 and beta4 >= 0:
#                 beta3_best = beta3
#                 beta4_best = beta4
#         except np.linalg.LinAlgError:
#             rs_old = 0
#             rs = 1

#         if beta3 <= 0 or beta4 <= 0:
#             rs_old = 0
#             rs = 1
#             # beta4 = beta4old * eps * 0.01

#             # ax.plot(voltage[idx], current[idx],".")


#         if idx <= idx_min + 1:
#             # print("use best so far")
#             beta3 = beta3_best
#             beta4 = beta4_best


#     # create a set of Parameters
#     # params = Parameters()
#     # params.add('beta2', value=1)
#     # params.add('beta3', value=1)
#     # params.add('beta4', value=1, min = 0)

#     # do fit, here with the default leastsq algorithm
#     # minner = Minimizer(fcn2min, params, fcn_args=(x[idx], np.log(y[idx])))
#     # result = minner.minimize()
#     # report_fit(result)
#     # beta3 = result.params['beta3'].value
#     # beta4 = result.params['beta4'].value

#     if any(np.isnan([beta3, beta4])):
#         raise RuntimeError("Parameter extraction failed: beta3={}, beta4={}"
#                            .format(beta3, beta4))
#     else:
#         return beta3, beta4