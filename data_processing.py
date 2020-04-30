import os
import sys
import time
import copy
import gc

import numpy as np
import random
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.utils.data as utils

sys.path.append('audio_tagging_functions')
from models import *


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


### Data processing functions ###
def load_df2array(df, sample_rate, audio_duration):
    """
    Loads and stores arrays, and also returns a dict of missing files.
    """
    nonValidDict = {}
    waveforms_list = []
    ConstantShape = sample_rate * audio_duration # Zero-padding

    for idx in range(len(df)):
        filename = df['fname'][idx]
        length = df['length'][idx]
        label = df['label'][idx]

        if os.path.isfile(filename):
            waveform = load_waveform2numpy(filename, length, sample_rate=sample_rate, audio_duration=audio_duration)
            length_waveform = len(waveform)

            # Zero-padding
            if length_waveform != ConstantShape:
                waveform = np.pad(waveform, (0, ConstantShape - length_waveform), 'constant')

            waveforms_list.append([waveform, label])

        else:
            iD = df['iD'][idx]
            nonValidDict[iD] = filename

    return (waveforms_list, nonValidDict)


def load_waveform2numpy(filename, length, sample_rate, audio_duration):
    """
    Loads from filename path and returns numpy array.
    """
    # Random crop of a 10 sec segment
    offset = random.randint(0, length-audio_duration)
    waveform, _ = librosa.core.load(filename, sr=sample_rate, mono=True, offset=offset, duration=audio_duration)

    return waveform


def get_dataloaders(x, y, batch_size):
    """
    Converts numpy arrays to Pyrtoch dataloader
    """
    tensor_x = torch.from_numpy(x).float().to(DEVICE)
    tensor_y = torch.from_numpy(y).long().to(DEVICE)

    tensordataset = utils.TensorDataset(tensor_x, tensor_y)
    dataloader_length = len(tensordataset)
    dataloader = utils.DataLoader(tensordataset, batch_size=batch_size, shuffle=True)

    # Free some memory spaces
    del x, y, tensor_x, tensor_y, tensordataset
    gc.collect()

    return (dataloader, dataloader_length)


def plot_distribution(do_plot, y_list):
    """

    """
    if do_plot:
        nb_y = len(y_list)
        labels = ["trainset", "validationset", "testset"]
        fig, axes = plt.subplots(1,nb_y,figsize=(8*nb_y,4))
        axes = axes.ravel()
        for idx, ax in enumerate(axes):
            y_array = y_list[idx]
            label = labels[idx]
            len_y = len(y_array)
            print(y_array)
            ax.hist(y_array, bins=int(len_y/3))
            ax.set_title(f"Distribution of the samples for {label} (n={len_y})")
            ax.set_xlabel('label')
            ax.set_ylabel('Number of samples')

        plt.tight_layout()
        plt.show()


def process_data(df, batch_size, sample_rate, audio_duration, random_state, do_plot=False):
    """
    Process data function, returns all the dataloaders for the training
    """
    print("Processing Data...")
    start_time = time.time()
    waveforms_list, nonValidDict = load_df2array(df, sample_rate, audio_duration)
    print(f"Valid files: {len(waveforms_list)}\nUnvalid files: {len(nonValidDict)}")

    waveforms_arrays = [x[0] for x in waveforms_list]
    X_all = np.vstack(waveforms_arrays)
    y_all = np.array([x[1] for x in waveforms_list])

    # Split Data in train, validation, test
    X_training, X_test, y_training, y_test = train_test_split(X_all, y_all, test_size=0.1, 
                                                              random_state=random_state, stratify=y_all)
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, 
                                                      random_state=random_state, stratify=y_training)
    print(f"X_train:{X_train.shape}, X_val:{X_val.shape}, X_test:{X_test.shape}")
    print(f"y_train:{y_train.shape}, y_val:{y_val.shape}, y_test:{y_test.shape}")

    y_list = [y_train, y_val, y_test]
    plot_distribution(do_plot, y_list)

    print("Attributing arrays to dataloaders...")

    trainloader, train_length = get_dataloaders(X_train, y_train, batch_size)
    validationloader, val_length = get_dataloaders(X_val, y_val, batch_size)
    testloader, test_length = get_dataloaders(X_test, y_test, batch_size)

    # Free some memory spaces
    del X_all, y_all, waveforms_list
    gc.collect()

    now = time.time()-start_time
    print(f"Data processing duration: {int(now//60)}min {int(now%60)}s")

    return ([trainloader, train_length], [validationloader, val_length], [testloader, test_length])



if __name__ == "__main__":

    ### Main parameters ###
    # Path
    LABELS_PATH = "labels/all_data.csv"

    # Audio parameters
    SR = 14000             # Sample Rate
    AUDIO_DURATION = 10    # 10 seconds duration window for all audios

    # Model parameters
    BATCH_SIZE = 8

    # Misc parameters
    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)


    ### Data processing ###
    labels_df = pd.read_csv(LABELS_PATH)
    print('Testing the process_data function with 10% of the dataset')
    (trainloader, validationloader, testloader) = process_data(df=labels_df.take(np.arange(int(len(labels_df)*0.1))), batch_size=BATCH_SIZE,
                                                               sample_rate=SR, audio_duration=AUDIO_DURATION, 
                                                               random_state=RANDOM_STATE, do_plot=True)