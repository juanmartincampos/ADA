import os
import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, iirnotch, detrend, butter, cheby1, lfilter, welch, cheb1ord, freqz
import digital_processing as dp
import feature_extraction as fe


def plot_psd(signal, sample_rate):
    '''
    Plot the power spectral density of a signal
    signal: array-like corresponding to the signal
    sample_rate: int corresponding to the sample rate of the signal in Hz
    '''
    f, Pxx_den = welch(signal, fs=sample_rate, window="hamming", nperseg=0.5*2000, noverlap=0.5*2000/2, nfft=sample_rate)
    plt.plot(f, Pxx_den/np.max(Pxx_den))
    plt.vlines(50, 0, 1, colors='r', linestyles='--')
    # plt.vlines(100, 0.5e-3, 1e3, colors='r', linestyles='--')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

def bandpass_filter(input_signal, sr, lowcut, highcut, order=5):
    
    nyq = 0.5 * sr # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    output_signal = filtfilt(b, a, input_signal)
    
    return output_signal

def NotchF(input, samprate, center, bw):
    wo = center/(samprate/2)
    bw = bw / (samprate/2)
    [b,a] = iirnotch(wo,bw)
    return filtfilt(b,a,input)

def filter_signals(emg_stream):
    # Set variables
    l_bandpass = 10
    h_bandpass = 400
    sample_rate = int(emg_stream['info']['nominal_srate'][0])

    # Get signal information from the XDF file and apply filters
    l_ear_emg = emg_stream["time_series"][:, 0] - emg_stream["time_series"][:, 1] # Left ear EMG
    
    for i in range(50, 250, 50):
        l_ear_emg = NotchF(l_ear_emg, sample_rate, i, 5) # Notch filter at harmonics of 50 Hz
    l_ear_emg = bandpass_filter(detrend(l_ear_emg), sample_rate, l_bandpass, h_bandpass)

    r_ear_emg = emg_stream["time_series"][:, 2] - emg_stream["time_series"][:, 3] # Right ear EMG
    for i in range(50, 250, 50):
        r_ear_emg = NotchF(r_ear_emg, sample_rate, i, 5) # Notch filter at harmonics of 50 Hz
    r_ear_emg = bandpass_filter(detrend(r_ear_emg), sample_rate, l_bandpass, h_bandpass)

    return l_ear_emg, r_ear_emg

def plot_signal(signal, n_ear = ""):
    # Plot signals
    plt.figure(figsize=(10,3))
    plt.plot(signal)
    plt.title(f"{n_ear.capitalize()}")

    plt.show()

def structure_data(emg_stream, EMGinfo_stream, ear_class, ear_emg, sample_rate, time_window=0.5):
    '''
    This function will:
        Check if the classification stream starts earlier, in that case, don't take into account all classification data that starts before
        Create a DataFrame where the class label, the index and the emg data for the time window will be stored, there will be as many columns as needed. If the time window 
        is 500 ms and the sampling rate is 2000 Hz, there will be 1000 columns corresponding to each sEMG recording
    
    :param emg_stream: xdf stream containing the EMG data
    :param EMGinfo_stream: xdf stream containing the classification data
    :param ear_class: array containing the class labels for one ear
    :param ear_emg: array containing the ear EMG data for one ear
    :param sample_rate: int corresponding to the sampling rate of the EMG data recording
    :param time_window: float corresponding to the time window we want to extract in seconds

    '''

    if emg_stream["time_stamps"][0] - EMGinfo_stream["time_stamps"][0] > 0: # This condition is true when EMG signal starts recording AFTER the classification, which makes no sense

        # Find the closest timestamp in the EMGinfo stream to the first timestamp in the EMG stream
        emg_first_timestamp = emg_stream["time_stamps"][0]
        emginfo_timestamps = EMGinfo_stream["time_stamps"]

        closest_index = np.abs(emginfo_timestamps - emg_first_timestamp).argmin()
        print(f"EMG stream starts after classification stream. Closest index: {closest_index}")
    else:
        closest_index = 0
        print(f"EMG stream starts at the same time as the classification stream. Closest index: {closest_index}")
    
    
    # This code will extract a window of EMG data before each class label in the EMGinfo stream and store it in a DataFrame
    # Corresponding number of samples based on frequency and the time window
    samples_window = int(sample_rate * time_window)

    # Create an empty DataFrame
    columns = ['class_label'] + ["stream_idx"] + [f'emg_{i}' for i in range(samples_window)]
    df = pd.DataFrame(columns=columns)

    for class_time, class_label in zip(EMGinfo_stream['time_stamps'][closest_index:], ear_class[closest_index:]): # There was a bug here, the ear_class was considering all the values, not only the ones after the closest_index
        # Find the closest timestamp in the EMG stream
        idx = np.abs(emg_stream['time_stamps'] - class_time).argmin()
        
        # Check if we can extract a full window of data without going out of bounds, append the data only if we're in bounds
        if idx >= samples_window:
            # Extract EMG data for the window
            emg_data_window = ear_emg[idx - samples_window:idx]
            
            # Append to the DataFrame
            row_data = [class_label] + [idx] + emg_data_window.tolist()
            df.loc[len(df)] = row_data
    
    return df

def get_features(ear_df, sample_rate, time_window=0.5, name=""):

    frame = int(time_window*sample_rate) # Number of samples in the time window

    ear_df_values = ear_df.iloc[:, 2:].values # Get the values of the EMG data from the 2nd column onwards

    output_df = pd.DataFrame()  # Create an empty dataframe to store the outputs

    for row in ear_df_values:
        features_df, _ = fe.features_estimation(row, name, sample_rate, frame, step=1, plot=False, verbose=False)  # Call the features_estimation function
        '''
        The step parameter that is passed to the features_estimation function makes reference to the number of samples that will be taken into account to compute the features. This is
        because the code provided is meant to be used as a sliding window, so the step parameter will determine how many samples will be taken into account to compute the features for the
        whole signal. In this case, we are passing 1, which means that we are taking all the samples into account to compute the features because we already split the signal into windows as 
        one row of the ear_df dataframe corresponds to one window of the signal.
        '''
        output_df = pd.concat([output_df, features_df], axis=1)  # Concatenate the features dataframe to the output dataframe column by column

    return pd.concat([ear_df.reset_index(drop=True), output_df.T.reset_index(drop=True)], axis=1) # Concatenate the output dataframe with the original dataframe

def filter_signals_2(emg_stream):

    # Set variables
    l_bandpass = 10
    h_bandpass = 400
    sample_rate = int(emg_stream['info']['nominal_srate'][0])

    l_ear_emg = emg_stream["time_series"][:, 0] - emg_stream["time_series"][:, 1] # Left ear EMG
    r_ear_emg = emg_stream["time_series"][:, 2] - emg_stream["time_series"][:, 3] # Right ear EM

    # Filter the 50Hz armonics (50, 100, 150, 200 Hz)
    for i in range(50, 250, 50):
        l_ear_emg = dp.notch_filter(l_ear_emg, sample_rate, i, False)
        r_ear_emg = dp.notch_filter(r_ear_emg, sample_rate, i, False)

    # Apply bandpass filters 
    l_ear_bandpass = dp.bp_filter(detrend(l_ear_emg), l_bandpass, h_bandpass, sample_rate, False)
    r_ear_bandpass = dp.bp_filter(detrend(r_ear_emg), l_bandpass, h_bandpass, sample_rate, False)

    # Apply wavelet denoising
    l_ear_denoise = dp.denoisewavelet(l_ear_bandpass)
    r_ear_denoise = dp.denoisewavelet(r_ear_bandpass)
    
    return l_ear_denoise, r_ear_denoise

def process_xdf(path):
    
    folders = path.split("/")[:-1]
    folder = "/".join(folders)
    
    csv_file_l = folder + '/l_ear_features_' + folders[-1] + '.csv'
    csv_file_r = folder + '/r_ear_features_' + folders[-1] + '.csv'

    print(f"Data in folder {folder} is being processed.")

    print("Checking if csv files exist.")
    if os.path.exists(csv_file_l) and os.path.exists(csv_file_r):
        print("Files exist. Loading csv files.")
        l_ear_features_df = pd.read_csv(csv_file_l)
        r_ear_features_df = pd.read_csv(csv_file_r)
    
    else:
        print("Files do not exist. Creating csv files.")
        data, _ = pyxdf.load_xdf(path)

        EMGinfo_stream = [stream for stream in data if stream["info"]["name"][0] == "EMGinfo"][0]
        emg_stream = [stream for stream in data if "eegoSports " in stream["info"]["name"][0]][0]

        sample_rate = int(emg_stream['info']['nominal_srate'][0])
        time_window = 0.5

        l_ear_emg, r_ear_emg = filter_signals_2(emg_stream)
        l_ear_class = EMGinfo_stream["time_series"][:, 6]
        r_ear_class = EMGinfo_stream["time_series"][:, 7]

        l_ear_df = structure_data(emg_stream, EMGinfo_stream, ear_class=l_ear_class, ear_emg=l_ear_emg, sample_rate=sample_rate)
        r_ear_df = structure_data(emg_stream, EMGinfo_stream, ear_class=r_ear_class, ear_emg=r_ear_emg, sample_rate=sample_rate)

        l_ear_features_df = get_features(l_ear_df, sample_rate, time_window, "left")
        r_ear_features_df = get_features(r_ear_df, sample_rate, time_window, "right")
        l_ear_features_df.to_csv(folder + '/l_ear_features_' + folders[-1] + '.csv', index=False)
        r_ear_features_df.to_csv(folder + '/r_ear_features_' + folders[-1] + '.csv', index=False)

    return l_ear_features_df, r_ear_features_df

def balance_dataset(df):
    # Pick 70% of the rows with class_label 0

    # Filter the rows where class_label is 0
    class_0_df = df[df['class_label'] == 0]

    # Randomly sample 70% of these rows
    class_0_sampled_df = class_0_df.sample(frac=0.3)

    # Filter the rows where class_label is not 0
    non_class_0_df = df[df['class_label'] != 0]

    # Concatenate the down-sampled class 0 DataFrame with the non-class 0 DataFrame
    balanced_df = pd.concat([class_0_sampled_df, non_class_0_df])

    # Shuffle the rows of the DataFrame
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    return balanced_df