#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIMIC ECG classification Batch Generator

Custom batch generator to feed spectrograms into a PyTorch model.
Modified from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
"""

#%% Imports

import numpy as np
import torch 
import keras
from torch.utils.data import Dataset, DataLoader
import os
import wfdb  
from physionet_processing import (spectrogram, norm_float)


#%% Batch generator class

class DataGenerator(keras.utils.Sequence):

    def __init__(self, records_df, base_path, list_IDs, labels, batch_size=16, dim=(156, 33),
                 nperseg=64, noverlap=32, data_mean=-9.01, data_std=9.00,
                 n_channels=12, sequence_length=5000, n_classes=3, shuffle=True, augment=False):
        """
        Initialization.
        :param records_df: DataFrame containing metadata for ECG records.
        :param base_path: Path to the directory containing ECG files.
        :param list_IDs: List of study IDs for data generation.
        :param labels: Dictionary mapping study IDs to class labels.
        :param batch_size: Number of samples per batch.
        :param dim: Dimensions of the spectrogram (time_steps, frequency_bins).
        :param nperseg: Number of samples per segment for the spectrogram.
        :param noverlap: Number of overlapping samples for the spectrogram.
        :param data_mean: Mean value for spectrogram normalization.
        :param data_std: Standard deviation for spectrogram normalization.
        :param n_channels: Number of channels (12 leads for ECG).
        :param sequence_length: Length of the ECG signal (5000 for MIMIC-IV).
        :param n_classes: Number of output classes for classification.
        :param shuffle: Whether to shuffle data after each epoch.
        """
        self.records_df = records_df
        self.base_path = base_path
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.data_mean = data_mean
        self.data_std = data_std
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
            
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples.
        """
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=float)
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
       
            record_path = self.records_df[self.records_df['study_id'] == ID]['path'].values[0]
            ecg_file_path = os.path.join(self.base_path, record_path)

            try:
                ecg_data, _ = wfdb.rdsamp(ecg_file_path)
            except FileNotFoundError:
                print(f"File not found: {ecg_file_path}")
                continue


            # Read the ECG data
            ecg_data, _ = wfdb.rdsamp(ecg_file_path)
            data = ecg_data.T # 12 lead


            # Generate spectrogram
            data_spectrogram = np.array([
                spectrogram(lead, nperseg=self.nperseg, noverlap=self.noverlap)[2]
                for lead in data
            ])


            # Normalize spectrogram
            data_transformed = norm_float(data_spectrogram, self.data_mean, self.data_std)

            # Store the spectrogram in X
            X[i,] = np.expand_dims(data_transformed, axis=3).transpose(1, 2, 0, 3) 


            # Store the label for the corresponding patient
            y[i] = self.labels[ID]


        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
      


if __name__ == '__main__':
    pass