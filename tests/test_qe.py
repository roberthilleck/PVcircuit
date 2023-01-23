import os

import numpy as np
import pandas as pd
import pytest

from pvcircuit import EQE


@pytest.fixture
def eqe():
    return EQE(np.array([1,1]),np.array([1,2]))



def test_eqe(eqe):
    
    assert 1 == 1
# def test_Multi2T(multi2T):

#     with open(os.path.join("tests", "Multi2T.txt"), "r", encoding="utf8") as fid:
#         multi2T_in = [line.rstrip().strip() for line in fid]

#     for i, r in enumerate(multi2T_in):
#         multi2T_out = multi2T.__str__().split("\n")[i].strip()
#         assert multi2T_out in multi2T_in
#         # if multi2T_out not in multi2T_in:
#         #     print(multi2T_in)




if __name__ == "__main__":
    test_eqe(EQE(np.array([1,1]),np.array([1,2])))
