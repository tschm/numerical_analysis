from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from sto_euler_schemes import (
    gen_rand_cov_mat,
    homogeneous,
    path_independent,
    time_independent,
    generic,
    generate_path_euler)


def mean_gbm(time):
    return 1e-3 * np.sin(time)

def vol_gbm(time):
    return 0.3 + 0.1 * np.cos(time)

def gbm_generic_form(dtime, time, dbrown, x_t, mean, vol):
    return (mean(time) * dtime + vol(time) * dbrown) * x_t

bachelier_generic_form = lambda dtime, time, dbrown, x_t, mean, vol: (
    gbm_generic_form(dtime, time, dbrown, 1, mean, vol))

bachelier = partial(bachelier_generic_form, x_t=None)

def cir(dtime, time, dbrown, r_t, strenght, long_term_mean, vol):
    return (
        strenght * (long_term_mean - r_t) * dtime
        + vol * r_t ** 0.5 * dbrown)  # time independent

def hull_white(dtime, time, dbrown, r_t, theta, alpha, vol):
    return (theta(time) - alpha(time) * r_t) * dtime + vol(time) * dbrown

schemes = {
    'gbm': (bachelier, homogeneous, 'Geometric Brownian Motion'),
    'gbm_generic_form': (gbm_generic_form, generic),
    'bachelier': (bachelier, path_independent, 'Bachelier Process'),
    'bachelier_generic_form': (bachelier_generic_form, generic),
    'cir': (cir, time_independent, 'Cox-Ingersoll-Ross Process'),
    'cir_generic_form': (cir, generic),
    'hull_white': (hull_white, generic, 'Hull-White Process'),
}


inits = [2, 3, 4, 5]
cov = gen_rand_cov_mat(len(inits))
print('covariance matrix')
print(cov)
print('\n')

def check_coherence(scheme_name, scheme_specs, init_vals, cov_mat, **kwargs):
    print(scheme_name)
    print(generate_path_euler(
        *scheme_specs[:2], init_vals, 10, 10, cov_mat, 0, **kwargs)[1])
    print('\n')

for name, specs in schemes.items():
    if 'b' in name:
        check_coherence(name, specs, inits, cov, vol=vol_gbm, mean=mean_gbm)

for name, specs in schemes.items():
    if 'cir' in name:
        check_coherence(name, specs, inits, cov, strenght=1, long_term_mean=2, vol=0.1)

def plot_scheme_path(scheme_name, scheme_specs, init_vals, cov_mat, save=True, **kwargs):
    plt.figure(figsize=(16, 9))
    plt.plot(*generate_path_euler(
        *scheme_specs[:2], init_vals, 10, 1000, cov_mat, **kwargs))
    plt.title(scheme_specs[2])
    if save:
        plt.savefig(f'{scheme_name}.png', dpi=300)
    plt.show()

plot_scheme_path(
    'gbm', schemes['gbm'], inits, cov, mean=mean_gbm, vol=vol_gbm)
plot_scheme_path(
    'bachelier', schemes['bachelier'], inits, cov, mean=mean_gbm, vol=vol_gbm)
plot_scheme_path(
    'cir', schemes['cir'], inits, cov, strenght=1, long_term_mean=2, vol=0.1)
plot_scheme_path(
    'hull_white', schemes['hull_white'], inits, cov, theta=mean_gbm, alpha=vol_gbm, vol=vol_gbm)
