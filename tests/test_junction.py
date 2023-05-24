import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pvlib import ivtools, pvsystem

import pvcircuit as pvc
from pvcircuit import Junction, Multi2T

if __name__ == "__main__":

    root = os.path.dirname(os.path.abspath(__file__))

    fp = os.path.join(root,"data","Pvsk_1.70MA-free_JV.csv")
#     fp = os.path.join(root,"IBC2x2.csv")

    A = 0.122
    TC = 25  # [degC]
    Eg = 1.8  # [eV]

    data = pd.read_csv(fp)
    # Measured terminal voltage.
    voltage = data['v'].to_numpy(np.double) #[V]
    # Measured terminal current.
    current = data['i'].to_numpy(np.double) / 1000 * A #[A]

    sort_id = np.argsort(voltage)

    voltage = voltage[sort_id]
    current = current[sort_id]

    photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = ivtools.sde.fit_sandia_simple(voltage, current)
    d_fitres = pvsystem.singlediode(photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth, ivcurve_pnts=100, method='brentq')

    fit_voltage = d_fitres['v']
    fit_current = d_fitres['i']

    Jext = photocurrent / A  # [A/cm^2]
    n = nNsVth / pvc.junction.Vth(TC)
    J0ref = saturation_current / A
    J0scale = 1000
    Rser = resistance_series * A
    Gsh = 1 / (resistance_shunt * A)
    # pvc.junction.DB_PREFIX
    Jdb = pvc.junction.Jdb(TC=TC, Eg=Eg)
    # j0=(self.Jdb * self.J0scale)**(1./self.n) * self.J0ratio / self.J0scale
    J0ratio = J0scale * J0ref / (Jdb * J0scale) ** (1.0 / n)

    PVK = Multi2T(name="Psk", area=A, Jext=Jext, Eg_list=[Eg], n=[n], J0ratio=[J0ratio])
    PVK.set(Rs2T=Rser, Gsh=Gsh)
    PVK.j[0]


    MPP = PVK.MPP()

    Voc = MPP["Voc"]
    Isc = MPP["Isc"]

    pvc_voltage_set = np.linspace(0, Voc)
    pvc_current_set = np.linspace(0, Isc)

    pvc_voltage_calc = np.zeros_like(pvc_voltage_set)
    pvc_current_calc = np.zeros_like(pvc_current_set)

    V2Tvect = np.vectorize(PVK.V2T)
    I2Tvect = np.vectorize(PVK.I2T)

    pvc_current_calc = I2Tvect(pvc_voltage_set)
    pvc_voltage_calc = V2Tvect(pvc_current_set)

    Vboth = np.concatenate((pvc_voltage_calc, pvc_voltage_set), axis=None)
    Iboth = np.concatenate((pvc_current_set, pvc_current_calc), axis=None)
            # sort
    p = np.argsort(Vboth)
    Vlight = Vboth[p]
    Ilight = -1 * Iboth[p]


    fig,ax = plt.subplots()
    ax.plot(voltage,current, ".")
    ax.plot(Vlight,Ilight)
    # ax.plot(Vl,Il, "--")
    plt.show()
