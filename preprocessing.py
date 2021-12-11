# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:14:41 2021

Script for data preprocessing and analysis for the EEGManyPipelines project

All steps outlined in this script are described in the README

@author: Simon KErn
"""
#%% imports
import os
import mne
import json
import numpy as np
import autoreject
import config
import mne_bids
from mne.preprocessing import ICA, read_ica
from psutil import virtual_memory as vram
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm # fancy progress bars
import mne_faster

np.random.seed(0)
#%% load data
# load all subjects from disk either into RAM or into a memmap
# memmap will be used if there is not enough memory free on the current system
subjects = []
data_dir = config.data_dir + '/eeg_brainvision/'
subjects = [f[:-5] for f in os.listdir(data_dir) if f.endswith('.vhdr')]

# ram_necessary = 1.2*len(subjects)
# ram_available = vram().available/1024**3
# ram_enough = ram_available<ram_necessary
data = {}

#%% 1.0 Annotate bad channels manually by eye-balling
for subj in tqdm(subjects, desc='annotate bad channels'):
    bads_json = f'{data_dir}{subj}_bads.json'
    if not os.path.exists(bads_json): 
        subj_vhdr = f'{data_dir}{subj}.vhdr'
        eogs = {'VEOG':'eog', 'HEOG':'eog'}
        miscs = {ch:'misc' for ch in ['M1', 'M2', 'IO1', 'IO2', 'Afp9', 'Afp10']}
        raw = mne.io.read_raw_brainvision(subj_vhdr, preload=True, verbose='WARNING')
        raw.set_channel_types({**eogs, **miscs})
        raw.notch_filter(50) # to remove visual noise that can easily be filtered
        print('please manually annotate bad channels by inspecting the plot')
        fig = raw.plot(duration=10, start=1000, n_channels=len(raw.ch_names), scalings={'eeg':2e-5, 'eog':1e-4, 'misc':1e-4}, block=True)
        bads = raw.info['bads']
        with open(bads_json, 'w') as f:
            bads = json.dump(bads, f)
    


report = {}

for subj in tqdm(subjects, desc='Loading participant information'):
    report[subj] = {}
    bads_json = f'{data_dir}{subj}_bads.json'
    subj_vhdr = f'{data_dir}{subj}.vhdr'

    # load data into memory
    raw = mne.io.read_raw_brainvision(subj_vhdr, preload=True, verbose='WARNING')
    
    # set correct channel information and set electrode location
    eogs = {'VEOG':'eog', 'HEOG':'eog'}
    miscs = {ch:'misc' for ch in ['M1', 'M2', 'IO1', 'IO2', 'Afp9', 'Afp10']}
    raw.set_channel_types({**eogs, **miscs})
    
    montage = mne.channels.read_custom_montage('Z:/EEGManyPipelines/channel_locations/chanlocs_besa.txt')
    raw.set_montage(montage)

    # manually annotate bad channels & interpolate them 
    # load previously annotated bad channels
    print('## loading previously manually determined bad channels')
    with open(bads_json, 'r') as f:
        bads = json.load(f)
        if len(bads)>0:
            raw.info['bads'] = bads
        report[subj]['bads_chs'] = bads
    # interpolate bad channels
    print('## Interpolate bad channels')
    raw.interpolate_bads()
    raw_orig = raw.copy() # create copy to use ICA


#%% 1.1 Re-referencing

    # set average reference to approximate electrode reference that
    # is maximally generalizable
    print('## Setting average reference')
    raw.set_eeg_reference(ref_channels='average')

#%% 1.2 Filtering
    
    #%% 1.2.1 FIR filter, bandpass from 0.1 to 100 Hz
    print('## Bandpassfilter 0.1-100 Hz')
    picks_eeg = mne.pick_types(raw.info, eeg=True)
    raw.filter(0.1, 100, method='fir', picks=picks_eeg)

    #%% 1.2.2 downsample to 250 Hz, all relevant ERP frequencies should be below that
    print('## resampling')
    raw.resample(250)
    
    
#%% 1.3 artefact rejection
#%% 1.3.1 ICA: Eyeblinks and horizontal eye movements
    # use picard ICA, most robust method available currently

    ica_method = 'picard'
    n_components = 64
    ica_fif = f'{data_dir}{subj}_ica.fif'
    # reject={'eeg':180e-5}
    # if we have computed this solution already, load it
    if os.path.isfile(ica_fif):
        ica = read_ica(ica_fif)
        assert ica.n_components==n_components, f'n components is not the same, please delete {ica_fif}'
        assert ica.method == ica_method, f'ica method is not the same, please delete {ica_fif}'
    # else compute it
    else:
        print(f'## Apply ICA {ica_method=} {n_components=}')
        ica = ICA(n_components=n_components, method=ica_method)
        # filter data with lfreq 1, as recommended by MNE, to remove slow drifts
        raw_ica = raw.copy().filter(l_freq=1, h_freq=40)
        ica.fit(raw_ica, picks=picks_eeg)
        ica.save(ica_fif) # save ICA to file for later loading
    
    print(f'## Finding EOG components')
    # find best match to EOG to get components that correspond with eye blinks/movements
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    eog_events = mne.preprocessing.find_eog_events(raw) # for later use
    assert len(eog_indices)>1, f'only {len(eog_indices)} eye components found'
    # ica.plot_overlay(raw_ica, exclude=eog_indices)
    # ica.plot_components(eog_indices)
    # ica.plot_sources(raw, eog_indices)
    ica.apply(raw, exclude=eog_indices)
    report[subj]['ica_eog_idx'] = eog_indices


#%% 1.3.2 artefact rejection
#%% 1.2.3 for ERP analysis, apply LP with 35 Hz and Notch with 50 Hz
    print(f'## Epoching')
    raw_erp = raw.copy()
    raw_erp.notch_filter(50)
    events, _ = mne.events_from_annotations(raw_erp, regexp='.*Stimulus')
    reject = {'eeg':100e-3} # +- 100 mV
    epochs = mne.Epochs(raw_erp, events, picks=picks_eeg, preload=True)
    epochs.filter(0, 35, method='fir')
    
    # define functions for threshold rejection
    
    def max_voltage_step_per_ms(epoch, maxvolt=0.05):
        """
        checks the difference of neighbouring values, 
        assuming that sfreq<1000, gives back True if threshold is surpassed
        """
        diff = np.abs(np.diff(epoch, axis=1))
        return np.any(diff>maxvolt)
    
    def max_diff_window(epoch, n_samples=50, maxvolt=0.2):
        """
        within a certain timewindow, difference of values can maximum be X
        """
        for i in range(epoch.shape[-1]-n_samples):
            segment = epoch[:,i:i+n_samples]
            maxdiff = np.ptp(segment, axis=1).max()
            if maxdiff>maxvolt: return True
        return False
    
    
    def max_amplitude(epoch, max_volt=0.1):
         """
         this is the same as max_diff_window in the end, makes no sense imo
         """
         return False
     
    print(f'## Finding bad epochs')
    bad_idx = []
    for i, epoch in enumerate(tqdm(epochs, desc='checking for bad epochs')):
        is_bad = max_voltage_step_per_ms(epoch) | max_diff_window(epoch)
        if is_bad: bad_idx.append(i)
    report[subj]['bad_epochs'] = bad_idx
    
    #%% apply artefact criteria 
    # see also Bublatzky et al 2020  https://doi.org/10.1016/j.cortex.2020.07.009
    
    
    #%% calculate good/bad epochs as percentage per participant
    #%% calculate good/bad epochs percentage per trial type
    #%% calculate good/bad epochs percentage per hypothesis dependency




