import numpy as np


def ft1(times, amplitudes, omegas):
    delta_t = times[1] - times[0]
    X = []
    for omega in omegas:
        e = np.exp(-1j * omega * times)
        X.append(np.dot(amplitudes, e) * delta_t)
    return np.array(X)


def ift1(times, spectrum, omegas):
    delta_omega = omegas[1] - omegas[0]
    x = []
    for time in times:
        e = np.exp(1j * omegas * time)
        x.append(np.dot(spectrum, e).real * delta_omega)
    return np.array(x) / np.pi


def ft2(times, amplitudes, omegas):
    delta_t = times[1] - times[0]
    X = []
    for omega in omegas:
        edt = np.exp(1j * omega * delta_t)
        phi = (np.exp(-1j * omega * times) / (delta_t * omega ** 2)) * (2 - edt - (1 / edt))
        X.append(np.dot(amplitudes, phi))
    return np.array(X)


def ift2(times, spectrum, omegas):
    delta_omega = omegas[1] - omegas[0]
    x = []
    for time in times:
        e_d_omega = np.exp(1j * time * delta_omega)
        psi = (np.exp(1j * time * omegas) / (delta_omega * time ** 2)) * (2 - e_d_omega - (1 / e_d_omega))
        x.append(np.dot(spectrum, psi).real)
    return np.array(x) / np.pi


def dft(amplitudes):
    N = len(amplitudes)
    X = []
    n = np.arange(N)
    for k in range(N):
        e = np.exp(-2j * np.pi * k * n / N)
        X.append(np.dot(amplitudes, e))
    return np.array(X)


def idft(spectrum):
    N = len(spectrum)
    restored_signal = []
    k = np.arange(N)
    for n in range(N):
        e = np.exp(2j * np.pi * k * n / N)
        restored_signal.append(np.dot(spectrum, e))
    return np.array(restored_signal) / N


def fft(amplitudes):
    return np.fft.fft(amplitudes)


def ifft(spectr):
    return np.fft.ifft(spectr)
