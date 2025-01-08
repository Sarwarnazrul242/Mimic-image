#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

MIMIC ECG classification adapted from PhysioNet

"""

#%% Imports

import numpy as np
import scipy as sc
from scipy import signal
import h5py  # Import to work with h5 files for MIMIC

#%% Functions

def special_parameters(h5file):
    ''' Returns unique sampling frequencies, sequence lengths, and recording times for MIMIC.'''

    # Get the list of dataset names
    dataset_list = list(h5file.keys())

    # Find sampling frequencies, sequence lengths and total recording times
    sequence_length = []
    sampling_rates = []
    recording_times = []
    baselines = []
    gains = []
    for fid in dataset_list:
        # Extract Lead I (the first lead) and gather metadata
        sequence_length.append(len(h5file[fid]['ecgdata'][0,: ]))  # Lead I data
        sampling_rates.append(500)
        print(h5file[fid]['ecgdata'].attrs['baseline'])
        recording_times.append(sequence_length[-1] / sampling_rates[-1])
        baselines.append(h5file[fid]['ecgdata'].attrs['baseline'])
        gains.append(h5file[fid]['ecgdata'].attrs['gain'])
        

    return (list(set(sequence_length)), list(set(sampling_rates)), 
            list(set(recording_times)), list(set(baselines)), list(set(gains)))


# Function to extend the time series data
def extend_ts(ts, length):
    extended = np.zeros(length)
    siglength = np.min([length, ts.shape[0]])
    extended[:siglength] = ts[:siglength]
    return extended 


# Fetch some raw sequences from the hdf5 file for MIMIC
def fetch_h5data(h5file, index_list, sequence_length):
    '''Out: array [samples, sequence]'''
    
    dataset_list = list(h5file.keys())
    load_list = [dataset_list[index] for index in index_list]
    
    print(dataset_list)
    data = []
    for dset in load_list:
        # Extract Lead I (first lead) for MIMIC data
        data.append(extend_ts(h5file[dset]['ecgdata'][:, 0], sequence_length))

    return np.vstack(data)


# Convert ECGs into spectrogram
def spectrogram(data, nperseg=64, noverlap=32, log_spectrogram=True):
    fs = 300  # Ensure the correct sampling rate for MIMIC data
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = np.transpose(Sxx, [0, 2, 1])
    if log_spectrogram:
        Sxx = abs(Sxx)  # Ensure all values are positive before taking log
        mask = Sxx > 0  # Avoid taking the log of zero
        Sxx[mask] = np.log(Sxx[mask])
    return f, t, Sxx


# Helper functions needed for data augmentation
def stretch_squeeze(source, length):
    target = np.zeros([1, length])
    interpol_obj = sc.interpolate.interp1d(np.arange(source.size), source)
    grid = np.linspace(0, source.size - 1, target.size)
    result = interpol_obj(grid)
    return result


def fit_tolength(source, length):
    target = np.zeros([length])
    w_l = min(source.size, target.size)
    target[0:w_l] = source[0:w_l]
    return target


# Data augmentation scheme: Dropout bursts
def zero_filter(input, threshold=2, depth=8):
    shape = input.shape
    # compensate for lost length due to mask processing
    noise_shape = [shape[0], shape[1] + depth]
    
    # Generate random noise
    noise = np.random.normal(0, 1, noise_shape)
    
    # Pick positions where the noise is above a certain threshold
    mask = np.greater(noise, threshold)
    
    # Grow a neighborhood of True values with at least length depth+1
    for d in range(depth):
        mask = np.logical_or(mask[:, :-1], mask[:, 1:])
    output = np.where(mask, np.zeros(shape), input)
    return output


# Data augmentation scheme: Random resampling
def random_resample(signals, upscale_factor=1):
    [n_signals, length] = signals.shape
    # pulse variation from 60 bpm to 120 bpm, expected 80 bpm
    new_length = np.random.randint(
        low=int(length * 80 / 120),
        high=int(length * 80 / 60),
        size=[n_signals, upscale_factor])
    signals = [np.array(s) for s in signals.tolist()]
    new_length = [np.array(nl) for nl in new_length.tolist()]
    sigs = [stretch_squeeze(s, l) for s, nl in zip(signals, new_length) for l in nl]
    sigs = [fit_tolength(s, length) for s in sigs]
    sigs = np.array(sigs)
    return sigs


# Spectrogram statistics needed for normalization of the dataset
def transformed_stats(h5file, nperseg, noverlap, sequence_length):
    ''' Gets some important statistics of the spectrograms in the entire dataset.
    We need this to rescale the data'''

    dataset_list = list(h5file.keys())
    sample_list = []

    for dataset in dataset_list:
        # Extract Lead I (first lead) and compute spectrogram
        data = extend_ts(h5file[dataset]['ecgdata'][:, 0], sequence_length)
        data = np.reshape(data, (1, len(data)))
        sample_list.append(np.expand_dims(spectrogram(data, nperseg, noverlap)[2], axis=3))
    
    sample_array = np.vstack(sample_list)
    
    # Flatten the array so that we can do statistics
    samples = np.ndarray.flatten(sample_array)
        
    return np.min(samples), np.max(samples), np.mean(samples), np.std(samples)


# Normalize floating-point ECG data
def norm_float(data, data_mean, data_std):
    scaled = (data - data_mean) / data_std
    return scaled


# Run as Script

if __name__ == '__main__':
    pass
