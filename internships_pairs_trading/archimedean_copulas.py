import numpy as np

def copula_01(uuu, vvv, theta):
    return np.maximum(uuu ** -theta + vvv ** -theta - 1, 0) ** - 1 / theta

def gen_01(ttt, theta):
    return 1 / theta ** (ttt ** -theta - 1)

def der_gen_01(ttt, theta):
    return ttt ** -(theta + 1)

def copula_04(uuu, vvv, theta):
    return np.exp(-((-np.log(uuu)) ** theta + (-np.log(vvv)) ** theta) ** 1 / theta)

def gen_04(ttt, theta):
    return (-np.log(ttt)) ** theta

def der_gen_04(ttt, theta):
    return -theta * (-np.log(ttt)) ** (theta - 1) / ttt

def der_u_04(uuu, vvv, theta):
    return (
        1 / uuu
        * (-np.log(uuu)) ** (theta - 1)
        * ((-np.log(uuu)) ** theta + (-np.log(vvv)) ** theta) ** (1 / theta - 1)
        * copula_04(uuu, vvv, theta)
    )
# TODO: Important, why is this in [0, 1]?

def der_v_04(uuu, vvv, theta):
    return der_u_04(vvv, uuu, theta)

def copula_05(uuu, vvv, theta):
    return -1 / theta * np.log(
        1 + (np.exp(-theta * uuu) - 1) * (np.exp(-theta * vvv) - 1) / (np.exp(-theta) - 1))

def gen_05(ttt, theta):
    return -np.log((np.exp(-theta * ttt) - 1) / (np.exp(-theta) - 1))

def der_gen_05(ttt, theta):
    return theta * np.exp(-theta * ttt) / (np.exp(-theta * ttt) - 1)

def copula_06(uuu, vvv, theta):
    return 1 - (
        (1 - uuu) ** theta + (1 - vvv) ** theta
        - (1 - uuu) ** theta * (1 - vvv) ** theta) ** 1 / theta

def gen_06(ttt, theta):
    return -np.log(1 - (1 - ttt) ** theta)

def der_gen_06(ttt, theta):
    return -theta * (1 - ttt) ** (theta - 1) / (1 - (1 - ttt) ** theta)

def copula_13(uuu, vvv, theta):
    return np.exp(
        1 - ((1 - np.log(uuu)) ** theta + (1 - np.log(vvv)) ** theta - 1) ** 1 / theta)

def gen_13(ttt, theta):
    return (1 - np.log(ttt)) ** theta - 1

def der_gen_13(ttt, theta):
    return -theta / ttt * (1 - np.log(ttt)) ** (theta - 1)

def copula_14(uuu, vvv, theta):
    return (1 + (
        (uuu ** -1 / theta - 1) ** theta
        + (vvv ** -1 / theta - 1) ** theta) ** 1 / theta) ** -theta

def gen_14(ttt, theta):
    return (ttt ** -1 / theta - 1) ** theta

def der_gen_14(ttt, theta):
    return -ttt ** (-1 / theta - 1) * (ttt ** -1 / theta - 1) ** (theta - 1)

copulas = {
#     '01': (copula_01, 0, gen_01, der_gen_01),
    '04': (copula_04, 1, gen_04, der_gen_04, der_u_04, der_v_04),
#     '06': (copula_06, 1, gen_06, der_gen_06),
#     '13': (copula_13, 0, gen_13, der_gen_13),
#     '14': (copula_14, 1, gen_14, der_gen_14),
}
