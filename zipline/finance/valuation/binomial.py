from pathlib import Path
import numpy as np
from ctypes import CDLL, c_int, c_double

fn = Path(__file__).resolve().parent / "optlib.so"
lib = CDLL(fn)


def binomial_american_call_value(s, k, r, t, vol, steps=100):
    fcn = lib.binomial_american_call_value
    fcn.argtype = [c_double, c_double, c_double, c_double, c_double, c_int]
    fcn.restype = c_double

    s_, k_, r_, t_, vol_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(vol),
        c_int(steps),
    )

    return fcn(s_, k_, r_, t_, vol_, steps_)


def binomial_american_call_delta(s, k, r, t, vol, steps=100):
    fcn = lib.binomial_american_call_delta
    fcn.argtype = [c_double, c_double, c_double, c_double, c_double, c_int]
    fcn.restype = c_double

    s_, k_, r_, t_, vol_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(vol),
        c_int(steps),
    )

    return fcn(s_, k_, r_, t_, vol_, steps_)


def binomial_american_call_gamma(s, k, r, t, vol, steps=100):
    fcn = lib.binomial_american_call_gamma
    fcn.argtype = [c_double, c_double, c_double, c_double, c_double, c_int]
    fcn.restype = c_double

    s_, k_, r_, t_, vol_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(vol),
        c_int(steps),
    )

    return fcn(s_, k_, r_, t_, vol_, steps_)


def binomial_american_call_theta(s, k, r, t, vol, steps=100):
    fcn = lib.binomial_american_call_theta
    fcn.argtype = [c_double, c_double, c_double, c_double, c_double, c_int]
    fcn.restype = c_double

    s_, k_, r_, t_, vol_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(vol),
        c_int(steps),
    )

    return fcn(s_, k_, r_, t_, vol_, steps_)


def binomial_american_call_vega(s, k, r, t, vol, steps=100):
    fcn = lib.binomial_american_call_vega
    fcn.argtype = [c_double, c_double, c_double, c_double, c_double, c_int]
    fcn.restype = c_double

    s_, k_, r_, t_, vol_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(vol),
        c_int(steps),
    )

    return fcn(s_, k_, r_, t_, vol_, steps_)


def binomial_american_call_rho(s, k, r, t, vol, steps=100):
    fcn = lib.binomial_american_call_rho
    fcn.argtype = [c_double, c_double, c_double, c_double, c_double, c_int]
    fcn.restype = c_double

    s_, k_, r_, t_, vol_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(vol),
        c_int(steps),
    )

    return fcn(s_, k_, r_, t_, vol_, steps_)


def binomial_american_call_implied_volatility_brent(
    s, k, r, t, call_option_price, steps=100, x_1=-2.0, x_2=2.0, tol=1.0e-6
):
    fcn = lib.binomial_american_call_implied_volatility_brent
    fcn.argtype = [
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_int,
        c_double,
        c_double,
        c_double,
    ]
    fcn.restype = c_double

    s_, k_, r_, t_, call_option_price_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(call_option_price),
        c_int(steps),
    )

    x_1_, x_2_, tol_ = c_double(x_1), c_double(x_2), c_double(tol)

    result = fcn(s_, k_, r_, t_, call_option_price_, steps_, x_1_, x_2_, tol_)

    return np.nan if result <= 1.0e-6 else result


def binomial_american_put_value(s, k, r, t, vol, steps=100):
    fcn = lib.binomial_american_put_value
    fcn.argtype = [c_double, c_double, c_double, c_double, c_double, c_int]
    fcn.restype = c_double

    s_, k_, r_, t_, vol_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(vol),
        c_int(steps),
    )

    return fcn(s_, k_, r_, t_, vol_, steps_)


def binomial_american_put_delta(s, k, r, t, vol, steps=100):
    fcn = lib.binomial_american_put_delta
    fcn.argtype = [c_double, c_double, c_double, c_double, c_double, c_int]
    fcn.restype = c_double

    s_, k_, r_, t_, vol_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(vol),
        c_int(steps),
    )

    return fcn(s_, k_, r_, t_, vol_, steps_)


def binomial_american_put_gamma(s, k, r, t, vol, steps=100):
    fcn = lib.binomial_american_put_gamma
    fcn.argtype = [c_double, c_double, c_double, c_double, c_double, c_int]
    fcn.restype = c_double

    s_, k_, r_, t_, vol_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(vol),
        c_int(steps),
    )

    return fcn(s_, k_, r_, t_, vol_, steps_)


def binomial_american_put_theta(s, k, r, t, vol, steps=100):
    fcn = lib.binomial_american_put_theta
    fcn.argtype = [c_double, c_double, c_double, c_double, c_double, c_int]
    fcn.restype = c_double

    s_, k_, r_, t_, vol_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(vol),
        c_int(steps),
    )

    return fcn(s_, k_, r_, t_, vol_, steps_)


def binomial_american_put_vega(s, k, r, t, vol, steps=100):
    fcn = lib.binomial_american_put_vega
    fcn.argtype = [c_double, c_double, c_double, c_double, c_double, c_int]
    fcn.restype = c_double

    s_, k_, r_, t_, vol_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(vol),
        c_int(steps),
    )

    return fcn(s_, k_, r_, t_, vol_, steps_)


def binomial_american_put_rho(s, k, r, t, vol, steps=100):
    fcn = lib.binomial_american_put_rho
    fcn.argtype = [c_double, c_double, c_double, c_double, c_double, c_int]
    fcn.restype = c_double

    s_, k_, r_, t_, vol_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(vol),
        c_int(steps),
    )

    return fcn(s_, k_, r_, t_, vol_, steps_)


def binomial_american_put_implied_volatility_brent(
    s, k, r, t, call_option_price, steps=100, x_1=-2.0, x_2=2.0, tol=1.0e-6
):
    fcn = lib.binomial_american_put_implied_volatility_brent
    fcn.argtype = [
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_int,
        c_double,
        c_double,
        c_double,
    ]
    fcn.restype = c_double

    s_, k_, r_, t_, call_option_price_, steps_ = (
        c_double(s),
        c_double(k),
        c_double(r),
        c_double(t),
        c_double(call_option_price),
        c_int(steps),
    )

    x_1_, x_2_, tol_ = c_double(x_1), c_double(x_2), c_double(tol)

    result = fcn(s_, k_, r_, t_, call_option_price_, steps_, x_1_, x_2_, tol_)

    return np.nan if result <= 1.0e-6 else result
