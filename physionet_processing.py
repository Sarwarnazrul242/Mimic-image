#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIMIC ECG classification (adapted for 12-lead ECG data)
"""

# %% Imports

import os
import numpy as np
from scipy import signal
import wfdb  # For reading ECG data files (.dat, .hea)

# %% Functions

def fetch_data(records_df, index_list, sequence_length, base_path):
    """
    Fetch ECG data from the files in the directory using paths from records_filtered.csv.
    `records_df`: DataFrame containing the paths to ECG data.
    `index_list`: List of indices to fetch data for.
    `sequence_length`: Length to which each ECG record will be extended.
    `base_path`: Base directory where the ECG data files are stored.
    
    Returns: array [samples, 12 leads, sequence]
    """
    data = []
    for idx in index_list:
        record_path = records_df.iloc[idx]['path']
        ecg_file_path = os.path.join(base_path, record_path)
        
        # Read ECG data using wfdb's rdsamp function
        ecg_data, _ = wfdb.rdsamp(ecg_file_path)
        
        # Extend each lead to match the desired sequence length
        leads_data = []
        for lead in range(ecg_data.shape[1]):  # Iterate through all leads
            extended = np.zeros(sequence_length)
            siglength = min(sequence_length, ecg_data[:, lead].shape[0])
            extended[:siglength] = ecg_data[:, lead][:siglength]
            leads_data.append(extended)
        
        # Stack all leads together
        data.append(np.stack(leads_data, axis=0))
    
    return np.array(data)

def spectrogram(data, nperseg=64, noverlap=32, log_spectrogram=True):
    """
    Convert ECG data into a spectrogram using a short-time Fourier transform.
    This function handles 12-lead data.
    
    `data`: Input ECG data of shape [12 leads, sequence_length].
    Returns: Spectrograms for each lead as a 3D array [12 leads, time_steps, frequency_bins].
    """
    fs = 500  # Sample frequency for ECG data (500 Hz for MIMIC-IV)
    spectrograms = []
    for lead_data in data:  # Process each lead separately
        f, t, Sxx = signal.spectrogram(lead_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        if log_spectrogram:
            Sxx = abs(Sxx)  # Ensure all values are positive before taking log
            mask = Sxx > 0  # Avoid taking log of zero
            Sxx[mask] = np.log(Sxx[mask])
        spectrograms.append(Sxx)
    
    return np.array(spectrograms)  # Shape: [12 leads, time_steps, frequency_bins]

def transformed_stats(records_df, nperseg, noverlap, sequence_length, base_path):
    """
    Get statistics (min, max, mean, std) for the spectrograms in the entire dataset.
    Used for normalizing the spectrogram data.
    
    Returns: min, max, mean, std of the dataset.
    """
    sample_list = []

    for idx, row in records_df.iterrows():
        ecg_file_path = os.path.join(base_path, row['path'])
        ecg_data, _ = wfdb.rdsamp(ecg_file_path)
        
        # Extend each lead to match the desired sequence length
        leads_data = []
        for lead in range(ecg_data.shape[1]):  # Iterate through all leads
            extended = np.zeros(sequence_length)
            siglength = min(sequence_length, ecg_data[:, lead].shape[0])
            extended[:siglength] = ecg_data[:, lead][:siglength]
            leads_data.append(extended)
        
        # Convert to spectrogram
        leads_data = np.stack(leads_data, axis=0)
        spectrograms = spectrogram(leads_data, nperseg=nperseg, noverlap=noverlap)
        sample_list.append(spectrograms)
    
    # Flatten the array to calculate statistics
    sample_array = np.concatenate([s.flatten() for s in sample_list])
        
    return np.min(sample_array), np.max(sample_array), np.mean(sample_array), np.std(sample_array)

def norm_float(data, data_mean, data_std):
    """
    Normalize the spectrogram data using zero mean and standard deviation.
    `data`: Input spectrogram data of shape [12 leads, time_steps, frequency_bins].
    """
    return (data - data_mean) / data_std

# Run as Script

if __name__ == '__main__':
    pass
























# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# MIMIC ECG classification (adapted from Physionet ECG classification)
# """

# #%% Imports

# import os
# import numpy as np
# import scipy as sc
# from scipy import signal
# import wfdb  # For reading ECG data files (.dat, .hea)
 
# #%% Functions

# # def extend_ts(ts, length):
# #     """
# #     Extend or truncate the time series (ECG) data to match the desired length.
# #     """
# #     extended = np.zeros(length)
# #     siglength = np.min([length, ts.shape[0]])
# #     extended[:siglength] = ts[:siglength]
# #     return extended 

# def fetch_data(records_df, index_list, sequence_length, base_path):
#     """
#     Fetch ECG data from the files in the directory using paths from records_filtered.csv.
#     `records_df`: DataFrame containing the paths to ECG data.
#     `index_list`: List of indices to fetch data for.
#     `sequence_length`: Length to which each ECG record will be extended.
#     `base_path`: Base directory where the ECG data files are stored.
    
#     Returns: array [samples, sequence]
#     """
#     data = []
#     for idx in index_list:
#         record_path = records_df.iloc[idx]['path']
#         ecg_file_path = os.path.join(base_path, record_path)
        
#         # Read ECG data using wfdb's rdsamp function
#         ecg_data, _ = wfdb.rdsamp(ecg_file_path)
        
#         # Use only Lead I (index 0) and extend to the specified sequence length
#         data.append(extend_ts(ecg_data[:, 0], sequence_length))
    
#     return np.vstack(data)

# # Convert ECGs into spectrogram
# def spectrogram(data, nperseg=64, noverlap=32, log_spectrogram=True):
#     """
#     Convert ECG data into a spectrogram using a short-time Fourier transform.
#     """
#     fs = 300  # Sample frequency for ECG data
#     f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
#     Sxx = np.transpose(Sxx, [0, 2, 1])
    
#     if log_spectrogram:
#         Sxx = abs(Sxx)  # Ensure all values are positive before taking log
#         mask = Sxx > 0  # Avoid taking log of zero
#         Sxx[mask] = np.log(Sxx[mask])
    
#     return f, t, Sxx

# # Helper functions needed for data augmentation
# # def stretch_squeeze(source, length):
# #     """
# #     Stretch or squeeze the source signal to match the desired length.
# #     """
# #     target = np.zeros([1, length])
# #     interpol_obj = sc.interpolate.interp1d(np.arange(source.size), source)
# #     grid = np.linspace(0, source.size - 1, target.size)
# #     result = interpol_obj(grid)
# #     return result

# # def fit_tolength(source, length):
# #     """
# #     Fit the source signal to the desired length by truncating or zero-padding it.
# #     """
# #     target = np.zeros([length])
# #     w_l = min(source.size, target.size)
# #     target[0:w_l] = source[0:w_l]
# #     return target

# # Data augmentation scheme: Dropout bursts
# # def zero_filter(input, threshold=2, depth=8):
# #     """
# #     Apply dropout bursts to the input signal for data augmentation.
# #     """
# #     shape = input.shape
# #     noise_shape = [shape[0], shape[1] + depth]  # Add depth to compensate for lost length

# #     # Generate random noise
# #     noise = np.random.normal(0, 1, noise_shape)

# #     # Pick positions where the noise is above a certain threshold
# #     mask = np.greater(noise, threshold)

# #     # Grow a neighbourhood of True values with at least length depth+1
# #     for d in range(depth):
# #         mask = np.logical_or(mask[:, :-1], mask[:, 1:])
    
# #     output = np.where(mask, np.zeros(shape), input)
# #     return output

# # Data augmentation scheme: Random resampling
# # def random_resample(signals, upscale_factor=1):
# #     """
# #     Perform random resampling of the signal for data augmentation.
# #     """
# #     [n_signals, length] = signals.shape
# #     new_length = np.random.randint(
# #         low=int(length * 80 / 120),
# #         high=int(length * 80 / 60),
# #         size=[n_signals, upscale_factor])
    
# #     signals = [np.array(s) for s in signals.tolist()]
# #     new_length = [np.array(nl) for nl in new_length.tolist()]
    
# #     sigs = [stretch_squeeze(s, l) for s, nl in zip(signals, new_length) for l in nl]
# #     sigs = [fit_tolength(s, length) for s in sigs]
# #     sigs = np.array(sigs)
    
#     return sigs

# # Spectrogram statistics needed for normalization of the dataset
# def transformed_stats(records_df, nperseg, noverlap, sequence_length, base_path):
#     """
#     Get statistics (min, max, mean, std) for the spectrograms in the entire dataset.
#     We need this to rescale the data.
#     """
#     sample_list = []

#     for idx, row in records_df.iterrows():
#         ecg_file_path = os.path.join(base_path, row['path'])
#         ecg_data, _ = wfdb.rdsamp(ecg_file_path)
        
#         data = extend_ts(ecg_data[:, 0], sequence_length)  # Use only Lead I
#         data = np.reshape(data, (1, len(data)))
        
#         # Get the spectrogram and append to the list
#         sample_list.append(np.expand_dims(spectrogram(data, nperseg, noverlap)[2], axis=3))
    
#     sample_array = np.vstack(sample_list)
    
#     # Flatten the array to calculate statistics
#     samples = np.ndarray.flatten(sample_array)
        
#     return np.min(samples), np.max(samples), np.mean(samples), np.std(samples)

# # Float types are normalized to zero mean std
# def norm_float(data, data_mean, data_std):
#     """
#     Normalize the float data using zero mean and standard deviation.
#     """
#     scaled = (data - data_mean) / data_std
#     return scaled

# # Run as Script

# if __name__ == '__main__':
#     pass
