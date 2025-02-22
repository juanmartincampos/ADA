from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pyyawt


def notch_filter(x, samplerate, freq, plot=False):
    x = x - np.mean(x)

    high_cutoff_notch = (freq - 1) / (samplerate / 2)
    low_cutoff_notch = (freq + 1) / (samplerate / 2)

    [b, a] = signal.butter(4, [high_cutoff_notch, low_cutoff_notch], btype='stop')

    x_filt = signal.filtfilt(b, a, x.T)

    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt.T, 'k')
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt


def bp_filter(x, low_f, high_f, samplerate, plot=False):
    # x = x - np.mean(x)

    low_cutoff_bp = low_f / (samplerate / 2)
    high_cutoff_bp = high_f / (samplerate / 2)

    [b, a] = signal.cheby1(5, 0.1, [low_cutoff_bp, high_cutoff_bp], btype='bandpass') # should be butter

    '''
    Plot frequency response to check if the filter is stable

    w, h = signal.freqz(b, a)
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.title('Chebyshev Type I frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.show()
    '''

    x_filt = signal.filtfilt(b, a, x)

    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt, 'k')
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt


def plot_signal(x, samplerate, chname):
    t = np.arange(0, len(x) / samplerate, 1 / samplerate)
    plt.plot(t, x)
    plt.autoscale(tight=True)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (mV)')
    plt.title(chname)
    plt.show()


def denoisewavelet(x1, level=5):
    xd, cxd, lxd = pyyawt.wden(x1, 'minimaxi', 'h', 'mln', level, 'db5') # changed from "s" to "h" to avoid spike at low frequencies
    return xd

