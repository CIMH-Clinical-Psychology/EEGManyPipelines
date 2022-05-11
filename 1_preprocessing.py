# -*- coding: utf-8 -*-
#%% imports
"""
Created on Tue Nov 16 10:14:41 2021

Script for data preprocessing and analysis for the EEGManyPipelines project

All steps outlined in this script are described in the README

@author: Simon Kern
"""
import os
import mne
import json
import autoreject
import config
import numpy as np
from mne.preprocessing import ICA, read_ica
from tqdm import tqdm # fancy progress bars

np.random.seed(0)
#%% load data

data_dir = config.data_dir + '/eeg_brainvision/'

# subjects contains a list of all participant ids, i.e. EMP01 to EMP33
subjects = [f[:-5] for f in os.listdir(data_dir) if f.endswith('.vhdr')]
subjects = sorted(subjects)

# data is a dictionary containing a mapping of subjects to their respective 
# preprocessed data
data = {}

#%% PREPROCESSING

report_dir = f'{data_dir}/report/'
os.makedirs(report_dir, exist_ok=True)

resample = False # set to true for ERP analysis, but for report needs to be false

for subj in tqdm(subjects, desc='Loading participant information'):

    # the submission asks us to only report the data used in hypothesis 4b
    # this data is not resampled, so resample is set to "original" here
    # sfreq = 'resampled' if resample else 'original'
    
    sfreq = 'original'  
    
    # these are files that we use to store intermediate results
    # to not having to recompute everything each time
    subj_vhdr = f'{data_dir}/{subj}.vhdr'
    ica_fif   = f'{data_dir}/{subj}_{sfreq}-ica.fif'
    
    # these are the files that are saved for later submission
    subj_dir = f'{report_dir}/{subj.replace("EMP", "Subj")}'
    report_bad_ch = f'{subj_dir}/excluded_chs.csv'
    report_bad_epochs = f'{subj_dir}/excluded_trials.csv'
    report_epochs_fif = f'{subj_dir}/preprocessed_epochs-epo.fif'
    report_ica = f'{subj_dir}/ica_components.txt'
    
    os.makedirs(subj_dir, exist_ok=True)
        
    # load data into memory
    raw = mne.io.read_raw_brainvision(subj_vhdr, preload=True, verbose='WARNING')
    
    # set correct channel information and set electrode location
    eogs = {'VEOG':'eog', 'HEOG':'eog'}
    miscs = {ch:'misc' for ch in ['M1', 'M2', 'IO1', 'IO2', 'Afp9', 'Afp10']}
    
    montage = mne.channels.read_custom_montage(f'{config.data_dir}/channel_locations/chanlocs_besa.txt')
    raw.set_montage(montage)
    raw.set_channel_types({**eogs, **miscs})

    #%% 1.0 Detect bad channels using RANSAC algorithm, implemented by autoreject   
    raw_RANSAC = raw.copy()
    raw_RANSAC.notch_filter(np.arange(50, 251, 50), verbose='WARNING')
    event_func = lambda x: int(x.split('/')[1])
    events, _ = mne.events_from_annotations(raw_RANSAC, event_id=event_func, regexp='.*Stimulus')
    epochs_RANSAC = mne.Epochs(raw_RANSAC, events, tmin=-0.2, tmax=0.5, picks='eeg', preload=True)

    ransac = autoreject.ransac.Ransac(n_jobs=-1)
    ransac.fit(epochs_RANSAC)
    with open(report_bad_ch, 'w') as f:
        f.write('\n'.join(ransac.bad_chs_))

    # interpolate bad channels
    raw.bads = ransac.bad_chs_
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
    raw.filter(0.1, 100, method='fir', picks=picks_eeg, verbose='WARNING')
    
#%% 1.3 artefact rejection
#%% 1.3.1 ICA: Eyeblinks and horizontal eye movements
    # use picard ICA, most robust method available currently
    ica_method = 'picard'
    n_components = 64
    
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
        raw_ica = raw.copy().filter(l_freq=1, h_freq=40, verbose='WARNING')
        ica.fit(raw_ica, picks=picks_eeg)
        ica.save(ica_fif) # save ICA to file for later loading
    
    print(f'## Determining EOG components')
    # find best match to EOG to get components that correspond with eye blinks/movements
    eog_indices, eog_scores = ica.find_bads_eog(raw, verbose='WARNING')
    assert len(eog_indices)>1, f'only {len(eog_indices)} eye components found'

    ica.apply(raw, exclude=eog_indices)
    
    with open(report_ica, 'w') as f:
        ica_str = (f'For {subj.replace("EMP", "Subj")}, our ICA (picard algorithm)'
                   f' decomposition yields {n_components} components.'
                   f' From those, we rejected a total of {len(eog_indices)} components,'
                   f' all of which were related to eye blinks and eye movements')
        f.write(ica_str)
   
#%% 1.3.2 artefact rejection
#%% 1.2.3 for ERP analysis, apply LP with 35 Hz and Notch with 50 Hz
    print(f'## Epoching')
    raw.notch_filter(np.arange(50, 251, 50), verbose='WARNING')
    event_func = lambda x: int(x.split('/')[1])
    events, stim_dict = mne.events_from_annotations(raw, event_id=event_func, 
                                                    regexp='.*Stimulus', verbose='WARNING')
    stim_dict = {config.stim2desc(key):val for key, val in stim_dict.items()}

    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=1,
                        event_id=stim_dict, picks=picks_eeg, preload=True,
                        verbose='WARNING')
    epochs.filter(0, 35, method='fir', verbose='WARNING')
    stop
    stim_dict_inv = {v: k for k, v in stim_dict.items()}
    #%% apply artefact criteria 
    
    print('## finding bad epochs with autoreject')
    
    if os.path.isfile(report_bad_epochs):
        print('## loading bad epochs with autoreject from file')
        with open(report_bad_epochs, 'r') as f:
            bad_epochs_autoreject = [int(x) for x in f.readlines()]
    else:
        print('## finding bad epochs with autoreject')
        ar = autoreject.AutoReject(n_jobs=-1)
        ar.fit(epochs)
        log = ar.get_reject_log(epochs)
        bad_epochs_autoreject = [int(x) for x in np.where(log.bad_epochs)[0]]

               
    # write bad epochs numers in file    
    with open(report_bad_epochs, 'w') as f:
        bad_epochs_str = '\n'.join([str(x) for x in bad_epochs_autoreject])
        f.write(bad_epochs_str)
        
    epochs.save(report_epochs_fif, fmt='double', overwrite=True)
    
    