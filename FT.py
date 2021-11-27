import numpy as np


def ft1(times, amplitudes, omegas):
    delta_t = times[1] - times[0]
    X = np.zeros((len(omegas),), dtype=np.complex128)
    for i in range(len(omegas)):
        X[i] = np.dot(amplitudes, np.exp(-1j * omegas[i] * times))
    return X * delta_t


def ift1(times, spectrum, omegas):
    delta_omega = omegas[1] - omegas[0]
    x = np.zeros((len(times),))
    for i in range(len(times)):
        x[i] = np.dot(spectrum, np.exp(1j * omegas * times[i])).real
    return x * (delta_omega / np.pi)


def ft2(times, amplitudes, omegas):
    delta_t = times[1] - times[0]
    X = np.zeros((len(omegas),), dtype=np.complex128)
    for i in range(len(omegas)):
        edt = np.exp(1j * omegas[i] * delta_t)
        k = (2 - edt - (1 / edt)) / (delta_t * omegas[i] ** 2)
        X[i] = k * np.dot(amplitudes, np.exp(-1j * omegas[i] * times))
    return X


def ift2(times, spectrum, omegas):
    delta_omega = omegas[1] - omegas[0]
    x = np.zeros((len(times),))
    for i in range(len(times)):
        e_d_omega = np.exp(1j * times[i] * delta_omega)
        k = (2 - e_d_omega - (1 / e_d_omega)) / (delta_omega * times[i] ** 2)
        x[i] = (k * np.dot(spectrum, np.exp(1j * times[i] * omegas))).real
    return x / np.pi


def dft(amplitudes):
    N = len(amplitudes)
    X = np.zeros((N,), dtype=np.complex128)
    n = np.arange(N)
    for k in range(N):
        e = np.exp(-2j * np.pi * k * n / N)
        X[k] = np.dot(amplitudes, e)
    return X / np.sqrt(N)


def idft(spectrum):
    N = len(spectrum)
    restored_signal = np.zeros((N,), dtype=np.complex128)
    k = np.arange(N)
    for n in range(N):
        e = np.exp(2j * np.pi * k * n / N)
        restored_signal[n] = np.dot(spectrum, e)
    return restored_signal / np.sqrt(N)
