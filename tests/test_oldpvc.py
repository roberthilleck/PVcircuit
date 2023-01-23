import os

import numpy as np
import pandas as pd
import pytest

from pvcircuit import Multi2T, Tandem3T


@pytest.fixture
def multi2T():
    return Multi2T()


@pytest.fixture
def tandem3T():
    return Tandem3T()


def test_Multi2T(multi2T):

    with open(os.path.join("tests", "Multi2T.txt"), "r", encoding="utf8") as fid:
        multi2T_in = [line.rstrip().strip() for line in fid]

    for i, r in enumerate(multi2T_in):
        multi2T_out = multi2T.__str__().split("\n")[i].strip()
        assert multi2T_out in multi2T_in
        # if multi2T_out not in multi2T_in:
        #     print(multi2T_in)


def test_Multi2T_MPP(multi2T):

    multi2T_in = {}
    with open(os.path.join("tests", "Multi2T_MPP.txt"), "r", encoding="utf8") as fid:
        for line in fid:
            data = line.rstrip().split(":")
            multi2T_in[data[0]] = float(data[1])

    assert multi2T_in == multi2T.MPP()


def test_Multi2T_IV(multi2T):

    multi2T_in = pd.read_csv(os.path.join("tests", "Multi2T_IV.csv"), index_col=0)

    MPP = multi2T.MPP()
    voltages = np.linspace(-0.2, MPP["Voc"])
    currents = np.linspace(-0.2, MPP["Isc"])

    I2T = np.vectorize(multi2T.I2T)
    V2T = np.vectorize(multi2T.V2T)

    Vboth = np.concatenate((voltages, V2T(currents)), axis=None)
    Iboth = np.concatenate((I2T(voltages), currents), axis=None)
    # sort
    p = np.argsort(Vboth)
    Vlight = Vboth[p]
    Ilight = Iboth[p]
    multi2T_out = pd.DataFrame({"v": Vlight, "i": Ilight}).dropna()

    pd.testing.assert_series_equal(multi2T_in["v"], multi2T_out["v"])
    pd.testing.assert_series_equal(multi2T_in["i"], multi2T_out["i"])


def test_Tandem3T(tandem3T):

    with open(os.path.join("tests", "Tandem3T.txt"), "r", encoding="utf8") as fid:
        tandem3T_in = [line.rstrip().strip() for line in fid]

    for i, r in enumerate(tandem3T_in):
        tandem3T_out = tandem3T.__str__().split("\n")[i].strip()
        assert tandem3T_out in tandem3T_in
        # if tandem3T_out not in tandem3T_in:
        # print(tandem3T_in)


def test_Tandem3T_MPP(tandem3T):

    with open(os.path.join("tests", "Tandem3T_MPP.txt"), "r", encoding="utf8") as fid:
        tandem3T_in = [line.rstrip().strip() for line in fid]

    for i, r in enumerate(tandem3T_in):
        tandem3T_out = tandem3T.MPP().__str__().split("\n")[i].strip()
        assert tandem3T_out in tandem3T_in
        
def test_Tandem3T_CM(tandem3T):

    with open(os.path.join("tests", "Tandem3T_CM.txt"), "r", encoding="utf8") as fid:
        tandem3T_in = [line.rstrip().strip() for line in fid]

    for i, r in enumerate(tandem3T_in):
        tandem3T_out = tandem3T.CM().__str__().split("\n")[i].strip()
        assert tandem3T_out in tandem3T_in
        
        
def test_Tandem3T_VM21(tandem3T):

    with open(os.path.join("tests", "Tandem3T_VM21.txt"), "r", encoding="utf8") as fid:
        tandem3T_in = [line.rstrip().strip() for line in fid]

    for i, r in enumerate(tandem3T_in):
        tandem3T_out = tandem3T.VM(2,1).__str__().split("\n")[i].strip()
        assert tandem3T_out in tandem3T_in
        
def test_Tandem3T_VM32(tandem3T):

    with open(os.path.join("tests", "Tandem3T_VM32.txt"), "r", encoding="utf8") as fid:
        tandem3T_in = [line.rstrip().strip() for line in fid]

    for i, r in enumerate(tandem3T_in):
        tandem3T_out = tandem3T.VM(3,2).__str__().split("\n")[i].strip()
        assert tandem3T_out in tandem3T_in
        
        
def test_Tandem3T_VM32_set(tandem3T):
    tandem3T.top.set(Eg=1.87, J0ratio=[ 80., 22.], Jext=0.0131, Gsh=1e-8, Rser = 0.1)   
    tandem3T.bot.set(Eg=1.419, J0ratio=[10., 15.], Jext=0.0128, Gsh= 5e-5, Rser=0.2, beta=3, area=0.89)
    tandem3T.set(Rz=1)

    with open(os.path.join("tests", "Tandem3T_VM32_set.txt"), "r", encoding="utf8") as fid:
        tandem3T_in = [line.rstrip().strip() for line in fid]

    for i, r in enumerate(tandem3T_in):
        tandem3T_out = tandem3T.VM(3,2).__str__().split("\n")[i].strip()
        assert tandem3T_out in tandem3T_in
        
    
    dev2T = Multi2T.from_3T(tandem3T)
    
    with open(os.path.join("tests", "Tandem3T_to_2Tcopy.txt"), "r", encoding="utf8") as fid:
        multi2T_in = [line.rstrip().strip() for line in fid]

    for i, r in enumerate(multi2T_in):
        multi2T_out = dev2T.__str__().split("\n")[i].strip()
        assert multi2T_out in multi2T_in


if __name__ == "__main__":
    test_Multi2T_MPP(Multi2T())
