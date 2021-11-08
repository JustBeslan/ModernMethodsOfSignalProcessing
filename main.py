import time

import matplotlib.pyplot as plt
from scipy.io import wavfile

from FT import *


def time_of_function(function):
    def wrapped(*args):
        start_time = time.perf_counter_ns()
        res = function(*args)
        print(time.perf_counter_ns() - start_time)
        return res

    return wrapped


@time_of_function
def task_3_2():
    signal = np.copy(amplitudes)
    N = 50
    for i in range(len(signal)):
        part = signal[i: i + N]
        signal[i] = sum(part) / len(part)
    ax.plot(times, signal, 'g', label="Filtered Signal", linestyle='--')
    ax.legend()
    plt.show()


@time_of_function
def task_3():
    signal = np.copy(amplitudes)
    N = 21
    step = int((N - 1) / 2) if N % 2 != 0 else int(N / 2)
    for i in range(len(signal)):
        step_left = 0 if i - step < 0 else i - step
        right_step = len(signal) if i + step > len(signal) else i + step
        part = signal[step_left: right_step]
        signal[i] = sum(part) / len(part)
    ax.plot(times, signal, 'g', label="Filtered Signal", linestyle='--')
    ax.legend()
    plt.show()


@time_of_function
def task_2_2():
    spectrum = fft(amplitudes)
    freq = np.arange(len(spectrum))

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.stem(freq, abs(spectrum), 'b', markerfmt=" ", basefmt="-b", label="FFT")
    ax1.set_xlabel('Freq (Hz)')
    ax1.set_ylabel('DFT Amplitude |X(freq)|')

    restored_signal = ifft(spectrum)
    ax.plot(times, restored_signal, 'g', label="Restored Signal", linestyle='-.')

    ax.legend()
    ax1.legend()
    plt.show()


@time_of_function
def task_2():
    spectrum = dft(amplitudes)
    freq = np.arange(len(spectrum))

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.stem(freq, abs(spectrum), 'b', markerfmt=" ", basefmt="-b", label="DFT")
    ax1.set_xlabel('Freq (Hz)')
    ax1.set_ylabel('DFT Amplitude |X(freq)|')

    restored_signal = idft(spectrum)
    ax.plot(times, restored_signal, 'g', label="Restored Signal")

    ax.legend()
    ax1.legend()
    plt.show()


@time_of_function
def task_1():
    omegas = np.arange(1e-3, 50, 1e-3)

    # spectrum = ft1(times, amplitudes, omegas)
    # restored_signal = ift1(times, spectrum, omegas)

    spectrum = ft2(times, amplitudes, omegas)
    restored_signal = ift2(times, spectrum, omegas)

    freq = np.arange(len(spectrum))
    ax1 = fig.add_subplot(2, 1, 2)
    ax1.plot(omegas / (2 * np.pi), abs(spectrum), 'b', label="FT")
    ax1.set_xlabel('Freq (Hz)')
    ax1.set_ylabel('DFT Amplitude |X(freq)|')

    ax.plot(times, restored_signal, 'g', label="Restored Signal")

    ax.legend()
    ax1.legend()
    plt.show()


def get_data_from_file(filename):
    count = 5000
    if filename[-3:] == 'wav':
        sr, data = wavfile.read(filename)
        length = data.shape[0] / sr
        data = data[:count]
        times = np.linspace(1e-3, length + 1e-3, data.shape[0])
        amplitudes = data[:, 0]
    else:
        with open(filename) as file:
            lines = file.readlines()
            lines = np.array([[float(num) for num in line.split('\t')] for line in lines if line.find('%') == -1])
            times, amplitudes = lines[:count, 0], lines[:count, 1]
    return times, amplitudes


def get_test_data():
    times = np.arange(-3, 3, 0.05)
    amplitudes = [np.cos(2 * np.pi * 3 * t) * np.exp(-np.pi * (t ** 2)) for t in times]
    return times, amplitudes


if __name__ == "__main__":
    # times, amplitudes = get_test_data()
    # times, amplitudes = get_data_from_file('../Task 1/Signals/small_PWAS1_to_PWAS4(Ch1)_pulse_0.5mus_Filter3MHz.txt')
    times, amplitudes = get_data_from_file('../Task 1/Signals/PWAS1_to_PWAS4(Ch1)_pulse_0.5mus_Filter3MHz.txt')
    # times, amplitudes = get_data_from_file('../Task 1/Signals/myVoice.wav')

    fig = plt.figure(figsize=(30, 20))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(times, amplitudes, 'r', label="Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel('Amplitude')

    # task_1()
    # task_2()
    # task_2_2()
    task_3()
    # task_3_2()
