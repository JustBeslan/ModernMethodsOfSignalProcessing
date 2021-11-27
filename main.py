import time

import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

from FT import *
from filters import *


def time_of_function(function):
    def wrapped(*args):
        start_time = time.perf_counter_ns()
        res = function(*args)
        print(time.perf_counter_ns() - start_time)
        return res

    return wrapped


@time_of_function
def task_3(type_filter):
    if type_filter == 'moving_average_1':
        """
            Фильтр скользящих средних (ключевой элемент - крайний слева)
            N - Ширина окна
        """
        filtered_signal = moving_average(
            signal=np.copy(amplitudes),
            N=21,
            pos_central_elem='left')
        ax.plot(times, filtered_signal, 'g', label="Filtered Signal", linestyle='-')
        ax.legend()
    elif type_filter == 'moving_average_2':
        """
            Фильтр скользящих средних (ключевой элемент - посередине)
            N - Ширина окна (нечетное число)
        """
        filtered_signal = moving_average(
            signal=np.copy(amplitudes),
            N=21,
            pos_central_elem='center')
        ax.plot(times, filtered_signal, 'g', label="Filtered Signal", linestyle='-')
        ax.legend()
    elif type_filter == 'hann':
        """
            Фильтр Низких частот с окном Ханна (Ханнинга)
            signal - Исходный сигнал
            window - тип оконной функции
        """
        filtered_signal = windows_filters(
            signal=np.copy(amplitudes),
            P=5,
            window=type_filter)
        ax.plot(times, filtered_signal, 'g', label="Filtered Signal", linestyle='-')
        ax.legend()
    elif type_filter == 'hamming':
        """
            Фильтр Низких частот с окном Хэмминга
            signal - Исходный сигнал
            window - тип оконной функции
            alpha - коэффициент для оконной функции Хэмминга
        """
        filtered_signal = windows_filters(
            signal=np.copy(amplitudes),
            window=type_filter,
            alpha=0.54)
        ax.plot(times, filtered_signal, 'g', label="Filtered Signal", linestyle='-')
        ax.legend()
    elif type_filter == 'blackman':
        """
            Фильтр Низких частот с окном Блэкмана
            signal - Исходный сигнал
            window - тип оконной функции
            alpha - коэффициент для оконной функции Блэкмана
        """
        filtered_signal = windows_filters(
            signal=np.copy(amplitudes),
            window=type_filter,
            alpha=0.16)
        ax.plot(times, filtered_signal, 'g', label="Filtered Signal", linestyle='-')
        ax.legend()
    elif type_filter == 'raised_cosine':
        """
            Полосовой фильтр с косинусоидальным сглаживанием
            signal - Исходный сигнал
            omega0 - Частота среза
            window - тип оконной функции
            alpha - коэффициент сглаживания
        """
        filtered_signal = windows_filters(
            signal=np.copy(amplitudes),
            window=type_filter,
            alpha=0.5,
            omega0=7)
        ax.plot(times, filtered_signal, 'g', label="Filtered Signal", linestyle='-')
        ax.legend()


@time_of_function
def task_2(type_ft):
    """
        Дискретное преобразование Фурье (ДПФ) и Быстрое преобразование Фурье (БПФ) с пакета numpy
        dft - Прямое ДПФ (результат в spectrum)
        fft - Прямое БПФ (результат в spectrum)
        idft - Обратное ДПФ (результат в restored_signal)
        ifft - Обратное БПФ (результат в restored_signal)
    """

    N = len(amplitudes)
    delta_t = times[1] - times[0]
    delta_omega = 2 * np.pi / (N * delta_t)
    omegas = np.array([k * delta_omega for k in range(N)])

    start_time = time.perf_counter_ns()
    if type_ft == 'dft':
        spectrum = dft(amplitudes)
        restored_signal = idft(spectrum)
    else:
        spectrum = np.fft.fft(amplitudes, norm='ortho')
        restored_signal = np.fft.ifft(spectrum, norm='ortho')
    print(f"time(ns) == {time.perf_counter_ns() - start_time}")

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.plot(omegas / (2 * np.pi), abs(spectrum), 'b', label="Spectrum")
    # freq = np.arange(len(spectrum))
    # ax1.plot(freq, abs(spectrum), 'b', label="Spectrum")
    # ax1.stem(freq, abs(spectrum), 'b', markerfmt=" ", basefmt="-b", label="DFT")
    ax1.set_xlabel('Freq')
    ax1.set_ylabel('Amplitude')

    ax.plot(times, restored_signal, 'g', label="Restored Signal", linestyle='--')

    ax.legend()
    ax1.legend()


@time_of_function
def task_1():
    """
        Преобразование Фурье (ПФ) двумя способами
        1. через прямоугольники
            ft1 - Прямое ПФ (результат в spectrum)
            ift1 - Обратное ПФ (результат в restored_signal)
        2. через интеграл
            ft2 - Прямое ПФ (результат в spectrum)
            ift2 - Обратное ПФ (результат в restored_signal)
    """
    omegas = np.arange(1e-2, 70, 1e-2)

    start_time = time.perf_counter_ns()

    # spectrum = ft1(times, amplitudes, omegas)
    # restored_signal = ift1(times, spectrum, omegas)
    spectrum = ft2(times, amplitudes, omegas)
    restored_signal = ift2(times, spectrum, omegas)

    print(f"time(ns) == {time.perf_counter_ns() - start_time}")

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.plot(omegas / (2 * np.pi), abs(spectrum), 'b', label="Spectrum")
    ax1.set_xlabel('Freq')
    ax1.set_ylabel('Amplitude')

    ax.plot(times, restored_signal, 'g', label="Restored Signal", linestyle='--')

    ax.legend()
    ax1.legend()


def get_data_from_file(filename):
    # start = 6000
    # count = 20000
    if filename[-3:] == 'wav':
        sr, data = wavfile.read(filename)
        length = data.shape[0] / sr
        # data = data[start:count]
        times = np.linspace(1e-3, length + 1e-3, data.shape[0])
        amplitudes = data[:, 0]
    else:
        with open(filename) as file:
            lines = file.readlines()
            lines = np.array([[float(num) for num in line.split('\t')] for line in lines if line.find('%') == -1])
            times, amplitudes = lines[:, 0], lines[:, 1]
            # times, amplitudes = lines[start:count, 0], lines[start:count, 1]
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
    # task_2('dft')
    task_3('hann')
    plt.show()
