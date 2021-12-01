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


def windows_smoothing(signal, window, **kwargs):
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
    for i in range(len(signal)):
        signal[i] *= w[i]
    return signal


def low_pass_filter(fig, freq_cutoff, delta_t, amplitudes):
    N = len(amplitudes)
    delta_omega = 2 * np.pi / (N * delta_t)
    omegas = np.array([k * delta_omega for k in range(N)])
    spectrum = np.fft.fft(amplitudes)

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.plot(omegas, abs(spectrum), 'b', label="Spectrum")
    ax1.set_xlabel('Freq')
    ax1.set_ylabel('Amplitude')

    for i in range(len(omegas)):
        if abs(spectrum[i]) >= freq_cutoff:
            spectrum[i] = 0

    ax1.plot(omegas, abs(spectrum), 'g', label="Spectrum")

    restored_amplitudes = np.fft.ifft(spectrum)
    return restored_amplitudes
