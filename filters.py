import numpy as np


def moving_average(signal, N, pos_central_elem):
    if pos_central_elem == 'left':
        for i in range(len(signal)):
            part = signal[i: i + N]
            signal[i] = sum(part) / len(part)
    else:
        step = int((N - 1) / 2) if N % 2 != 0 else int(N / 2)
        for i in range(len(signal)):
            step_left = 0 if i - step < 0 else i - step
            part = signal[step_left: i + step]
            signal[i] = sum(part) / len(part)
    return signal


def windows_filters(signal, window, **kwargs):
    N = len(signal)
    if window == 'hann':
        w = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif window == 'hamming':
        alpha = kwargs['alpha']
        w = np.array([alpha - (1 - alpha) * np.cos(2 * np.pi * n / N) for n in range(N)])
    elif window == 'blackman':
        alpha = kwargs['alpha']
        a0 = (1 - alpha) / 2
        a1 = 1 / 2
        a2 = alpha / 2
        w = np.array([a0 - a1 * np.cos(2 * np.pi * n / N) + a2 * np.cos(4 * np.pi * n / N) for n in range(N)])
    elif window == 'raised_cosine':
        alpha = kwargs['alpha']
        omega0 = kwargs['omega0']
        w = np.zeros((N,))
        for omega in range(N):
            if abs(omega) <= omega0 * (1 - alpha):
                w[omega] = 1
            if omega0 * (1 - alpha) <= abs(omega) <= omega0 * (1 + alpha):
                w[omega] = 1/2 * (1 - np.sin((np.pi * (abs(omega) - omega0)) / (2 * alpha * omega0)))
    for i in range(len(signal)):
        signal[i] *= w[i]
    return signal
