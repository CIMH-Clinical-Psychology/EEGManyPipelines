# -*- coding: utf-8 -*-
#%% imports
"""
Created on Tue Feb 22 10:14:41 2022

Perform ERP analysis after preprocessing has occured

All steps outlined in this script are described in the README

@author: Simon Kern
"""
from joblib import Parallel, delayed
import os
import mne
import json
import autoreject
import config
import autoreject
import mne_faster
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from mne.preprocessing import ICA, read_ica
from psutil import virtual_memory as vram
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm # fancy progress bars
from config import md5hash
from natsort import natsorted
from joblib.memory import Memory

memory = Memory(config.cache_dir)

np.random.seed(0)

#%%

#%% load data
# load all subjects from disk either into RAM or into a memmap
subjects = []
data_dir = config.data_dir + '/eeg_brainvision/'
subjects = sorted([f.name for f in os.scandir(data_dir) if f.is_dir() ])


for subj in subjects:
    subj_dir = f'{data_dir}/{subj}/'
    epochs_fif = f'{subj_dir}/{subj}_epochs.fif'
    bads_autoreject = [f for f in os.listdir(subj_dir) if 'autoreject' in f]
    
    assert len(bads_autoreject)==1, 'several or no autoreject annotations found'
    with open(f'{subj_dir}/{bads_autoreject[0]}', 'r') as f:
        bad_epochs = json.load(f)
        
    # load data
    epochs = mne.read_epochs(epochs_fif)
    epochs.drop(bad_epochs)
    
    #%% H1: effect of scene category
    # There is an effect of scene category (i.e., a difference between images showing
    # man-made vs. natural environments) on the amplitude of the N1 component, i.e. the
    # first major negative EEG voltage deflection.
    
    # Find best electrode to detect the N1
    # 1) make avg of trials per subj per electrode
    # 2) Find first negative peak 50-120ms, see which electrode has lowest amplitude
    # 3) use this electrode for further calculations
    
    # calculate component
    # 1) calculate peak position per participant for chosen electrode
    # 2) time windows for component min to max of peak +-5ms
    
    epochs
    stop