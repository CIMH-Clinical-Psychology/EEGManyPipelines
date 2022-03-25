# -*- coding: utf-8 -*-
#%% imports
"""
Created on Tue Nov 16 10:14:41 2021

Script for data preprocessing and analysis for the EEGManyPipelines project

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
#%% load data
# load all subjects from disk either into RAM or into a memmap
# memmap will be used if there is not enough memory free on the current system
subjects = []
data_dir = config.data_dir + '/eeg_brainvision/'
subjects = [f[:-5] for f in os.listdir(data_dir) if f.endswith('.vhdr')]
subjects = sorted(subjects)

# ram_necessary = 1.2*len(subjects)
# ram_available = vram().available/1024**3
# ram_enough = ram_available<ram_necessary
data = {}

# dont perform analysis when loading script accidentially
#%% 1.0 Annotate bad channels manually by eye-balling
# for subj in tqdm(subjects, desc='annotate bad channels'):
#     os.makedirs(f'{data_dir}/{subj}/', exist_ok=True)
#     bads_json = f'{data_dir}/{subj}/{subj}_bad_chs_manual.json'   
#     if not os.path.exists(bads_json): 
#         subj_vhdr = f'{data_dir}{subj}.vhdr'
#         eogs = {'VEOG':'eog', 'HEOG':'eog'}
#         miscs = {ch:'misc' for ch in ['M1', 'M2', 'IO1', 'IO2', 'Afp9', 'Afp10']}
#         raw = mne.io.read_raw_brainvision(subj_vhdr, preload=True, verbose='WARNING')
#         raw.set_channel_types({**eogs, **miscs})
#         raw.notch_filter(50) # to remove visual noise that can easily be filtered
#         print('please manually annotate bad channels by inspecting the plot')
#         fig = raw.plot(duration=10, start=1000, n_channels=len(raw.ch_names), scalings={'eeg':2e-5, 'eog':1e-4, 'misc':1e-4}, block=True)
#         bads = raw.info['bads']

report = {}
# for subj in tqdm(subjects, desc='Loading participant information'):
    
def preprocess(subj):
    from config import md5hash

    subj_dir = f'{data_dir}/{subj}/'
    subj_vhdr = f'{data_dir}/{subj}.vhdr'
    report_pkl = f'{subj_dir}/report.pkl'
    epochs_fif = f'{subj_dir}/{subj}-epo.fif'
    ica_fif   = f'{subj_dir}/{subj}_ica.fif'
    ica_json = f'{subj_dir}/{subj}_ica_description.json'
    bads_json = f'{subj_dir}/{subj}_bad_chs_manual.json'   
    trigger_txt = f'{subj_dir}/{subj}_epochs_type.txt'   
    ar_file_template = '{subj_dir}/{subj}_bad_epochs_autoreject_{md5hash(epochs.get_data())}.json'
    os.makedirs(subj_dir, exist_ok=True)
    report = {}
    report[subj] = {}


    # load data into memory
    raw = mne.io.read_raw_brainvision(subj_vhdr, preload=True, verbose='WARNING')
    
    # set correct channel information and set electrode location
    eogs = {'VEOG':'eog', 'HEOG':'eog'}
    miscs = {ch:'misc' for ch in ['M1', 'M2', 'IO1', 'IO2', 'Afp9', 'Afp10']}
    raw.set_channel_types({**eogs, **miscs})
    
    montage = mne.channels.read_custom_montage(f'{config.data_dir}/channel_locations/chanlocs_besa.txt')
    raw.set_montage(montage)

    # manually annotate bad channels & interpolate them 
    # load previously annotated bad channels
    print('## loading previously manually determined bad channels')
    if os.path.isfile(bads_json):
        with open(bads_json, 'r') as f:
            bads = json.load(f)
            if len(bads)>0:
                raw.info['bads'] = bads
            report[subj]['bad_chs'] = bads
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
    # raw.resample(250)
    
    
#%% 1.3 artefact rejection
#%% 1.3.1 ICA: Eyeblinks and horizontal eye movements
    # use picard ICA, most robust method available currently

    ica_method = 'picard'
    n_components = 64
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
    
    with open(ica_json, 'w') as f:
        json_desc = {'n_components':n_components, 
                     'eog components':[int(x) for x in eog_indices]}
        json.dump(json_desc, f)
    
    report[subj]['ica_eog_idx'] = eog_indices

#%% 1.3.2 artefact rejection
#%% 1.2.3 for ERP analysis, apply LP with 35 Hz and Notch with 50 Hz
    print(f'## Epoching')
    raw_erp = raw.copy()
    raw_erp.notch_filter(50)
    event_func = lambda x: int(x.split('/')[1])
    events, stim_dict = mne.events_from_annotations(raw_erp, event_id=event_func, regexp='.*Stimulus')
    stim_dict = {config.stim2desc(key):val for key, val in stim_dict.items()}

    report[subj]['stim_dict'] = stim_dict
    # reject = {'eeg':100e-3} # +- 100 mV
    epochs = mne.Epochs(raw_erp, events, tmin=-0.2, tmax=1, event_id=stim_dict, picks=picks_eeg, preload=True)
    epochs.filter(0, 35, method='fir')
    
    epochs.save(epochs_fif, fmt='double', overwrite=True)
    np.savetxt(trigger_txt, events[:,2], fmt='%d')
    stim_dict_inv = {v: k for k, v in stim_dict.items()}
    report[subj]['epoch_type'] = np.array([stim_dict_inv[e] for e in events[:,2]])
    #%% apply artefact criteria 
    # see also Bublatzky et al 2020  https://doi.org/10.1016/j.cortex.2020.07.009
    
    # define functions for threshold rejection
    
    def max_voltage_step_per_ms(epoch, maxvolt=50e-6):
        """
        checks the difference of neighbouring values, 
        assuming that sfreq<1000, gives back True if threshold is surpassed
        """
        diff = np.abs(np.diff(epoch, axis=1))
        return np.any(diff>maxvolt)
    
    def max_diff_window(epoch, n_samples=50, maxvolt=200e-6):
        """
        within a certain timewindow, difference of values can maximum be X
        """
        for i in range(epoch.shape[-1]-n_samples):
            segment = epoch[:,i:i+n_samples]
            maxdiff = np.ptp(segment, axis=1).max()
            if maxdiff>maxvolt: return True
        return False
    
    
    def max_amplitude(epoch, max_volt=100e-6):
         """
         this is the same as max_diff_window in the end, makes no sense imo
         """
         return False
     
    print(f'## Finding bad epochs with custom function')
    bad_idx = []
    for i, epoch in enumerate(tqdm(epochs, desc='checking for bad epochs')):
        is_bad = max_voltage_step_per_ms(epoch) | max_diff_window(epoch)
        if is_bad: bad_idx.append(i)
    report[subj]['bad_epochs'] = bad_idx

    print('## finding bad epochs with autoreject')
    ar_file = eval(f"f'{ar_file_template}'")
    if os.path.isfile(ar_file):
        print('## loading bad epochs with autoreject from file')
        with open(ar_file, 'r') as f:
            bad_epochs_autoreject = json.load(f)
    else:
        print('## finding bad epochs with autoreject')
        ar = autoreject.AutoReject(n_jobs=-1)
        ar.fit(epochs)
        log = ar.get_reject_log(epochs)
        bad_epochs_autoreject = [int(x) for x in np.where(log.bad_epochs)[0]]
        with open(ar_file, 'w') as f:
            json.dump(bad_epochs_autoreject, f)
    report[subj]['bad_epochs_autoreject'] = bad_epochs_autoreject
    
    print('## finding bad epochs with FASTER')
    report[subj]['bad_epochs_FASTER'] = sorted(mne_faster.find_bad_epochs(epochs))
    
    print('## find bad epochs with RANSAC')
    ransac = autoreject.ransac.Ransac(n_jobs=-1)
    ransac.fit(epochs)
    report[subj]['bad_chs_RANSAC'] = ransac.bad_chs_
    
    # mark bad epochs with autoreject (we decided so)
    
    with open(report_pkl, 'wb') as f:
        pickle.dump(report,f)
     
    
    return report[subj]
 
reports = Parallel(n_jobs=16)(delayed(preprocess)(subj) for subj in tqdm(subjects))
 
stop   

#%% calculate good/bad epochs as percentage per participant
import matplotlib.pyplot as plt
import seaborn as sns

with open(f'{data_dir}/report.pkl', 'rb') as f:
    report = pickle.load(f)

fig, axs = plt.subplots(2,1)
ax = axs[0]
sns.barplot(x=subjects, y=[len(report[s]['bad_chs']) for s in subjects], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('Bad channels by manual inspection')
ax.set_ylabel('n bad channels')
ax.set_ylim(0,5.5)

ax = axs[1]
sns.barplot(x=subjects, y=[len(report[s]['bad_chs_RANSAC']) for s in subjects], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('Bad channels by RANSAC algorithm')
ax.set_ylabel('n bad channels')
ax.set_ylim(0,5.5)

for i, subj in enumerate(subjects):
    text = '\n'.join(report[subj]['bad_chs'])
    axs[0].text(i-0.25, len(report[subj]['bad_chs'])*0.7, text)
    text = '\n'.join(report[subj]['bad_chs_RANSAC'])
    axs[1].text(i-0.25, len(report[subj]['bad_chs_RANSAC'])*0.7, text)
    
plt.pause(0.1)
plt.tight_layout()
plt.savefig('plot_bad_channels.png')

#%% calculate good/bad epochs percentage per participant type

fig, axs = plt.subplots(3,1)
ax = axs[0]
n_epochs = len(report['EMP01']['epoch_type'])
sns.barplot(x=subjects, y=[len(report[s]['bad_epochs'])/n_epochs for s in subjects], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('overall % Bad epochs by custom thresholds')
ax.set_ylabel('% rejected epochs')
ax.set_ylim(0, 0.1)

ax = axs[1]
sns.barplot(x=subjects, y=[len(report[s]['bad_epochs_autoreject'])/n_epochs for s in subjects], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('overall % Bad epochs by AutoReject algorithm')
ax.set_ylabel('% rejected epochs')
ax.set_ylim(0, 0.1)

ax = axs[2]
sns.barplot(x=subjects, y=[len(report[s]['bad_epochs_FASTER'])/n_epochs for s in subjects], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('overall % Bad epochs by FASTER algorithm')
ax.set_ylabel('% rejected epochs')
ax.set_ylim(0, 0.1)

plt.pause(0.1)
plt.tight_layout()
plt.savefig('plot_bad_epochs_per_participant.png')


#%% calculate good/bad epochs per trial type


#%% H1: manmade vs natural trials
fig, axs = plt.subplots(1,3)

algorithms = ['bad_epochs', 'bad_epochs_autoreject', 'bad_epochs_FASTER']
for i, algo in enumerate(algorithms):
    data = pd.DataFrame()

    for subj in subjects:
    
        bad_idx = report[subj][algo]
        epoch_type = np.array([int(x[9]) for x in report[subj]['epoch_type']])
        # we only care about scene effects, encoded in the first int
        bad_types =  epoch_type[bad_idx]
        # bad_percentage = {'bad man-made': sum(bad_types==1)/sum(epoch_type==1), 
        #                   'bad natural': sum(bad_types==2)/sum(epoch_type==2),
        #                   'algorithm': algo}
        data = data.append({'stim_type':'man-made', 
                     'val': sum(bad_types==1)/sum(epoch_type==1),
                     'algorithm':algo}, ignore_index=True)
        data = data.append({'stim_type':'natural', 
                     'val': sum(bad_types==2)/sum(epoch_type==2),
                     'algorithm':algo}, ignore_index=True)
    ax = axs[i]

    sns.boxplot(data=data, x='stim_type', y='val', ax=ax)
    ax.set_title(algo)
    ax.set_ylim(0,0.11)
    ax.set_ylabel('Percentage bad')
    
plt.suptitle('Percentage of bad epochs per trial type for category man-made/natural, minimum/maximum indicated')
plt.pause(0.1)
plt.tight_layout()
plt.savefig('plot_bad_epochs_H1.png')


#%% H2: old vs new
fig, axs = plt.subplots(1,3)

algorithms = ['bad_epochs', 'bad_epochs_autoreject', 'bad_epochs_FASTER']
for i, algo in enumerate(algorithms):
    data = pd.DataFrame()

    for subj in subjects:
    
        bad_idx = report[subj][algo]
        epoch_type = np.array([int(x[10]) for x in report[subj]['epoch_type']])
        # we only care about scene effects, encoded in the first int
        bad_types =  epoch_type[bad_idx]
        # bad_percentage = {'bad man-made': sum(bad_types==1)/sum(epoch_type==1), 
        #                   'bad natural': sum(bad_types==2)/sum(epoch_type==2),
        #                   'algorithm': algo}
        data = data.append({'stim_type':'new', 
                     'val': sum(bad_types==0)/sum(epoch_type==0),
                     'algorithm':algo}, ignore_index=True)
        data = data.append({'stim_type':'old', 
                     'val': sum(bad_types==1)/sum(epoch_type==1),
                     'algorithm':algo}, ignore_index=True)
    ax = axs[i]

    sns.boxplot(data=data, x='stim_type', y='val', ax=ax)
    ax.set_title(algo)
    ax.set_ylim(0,0.11)
    ax.set_ylabel('Percentage bad')
    
plt.suptitle('Percentage of bad epochs per trial type for old/new images, minimum/maximum indicated')
plt.pause(0.1)
plt.tight_layout()
plt.savefig('plot_bad_epochs_H2.png')


#%% H3: hit vs forgotten for old images
fig, axs = plt.subplots(1,3)

algorithms = ['bad_epochs', 'bad_epochs_autoreject', 'bad_epochs_FASTER']
for i, algo in enumerate(algorithms):
    data = pd.DataFrame()

    for subj in subjects:
    
        bad_idx = report[subj][algo]
        epoch_type = np.array([x[10:12] for x in report[subj]['epoch_type']]) 
        # we only care about scene effects, encoded in the first int
        bad_types =  epoch_type[bad_idx]
        bad_types = [int(x[1]) for x in bad_types if x[0]==1] # only old images
        # bad_percentage = {'bad man-made': sum(bad_types==1)/sum(epoch_type==1), 
        #                   'bad natural': sum(bad_types==2)/sum(epoch_type==2),
        #                   'algorithm': algo}
        epoch_type = np.array([int(x[0]) for x in epoch_type])
        bad_types = np.array(bad_types)
        data = data.append({'stim_type':'hit', 
                     'val': np.sum(bad_types==1)/np.sum(epoch_type==1),
                     'algorithm':algo}, ignore_index=True)
        data = data.append({'stim_type':'miss/forgotten', 
                     'val': np.sum(bad_types==2)/np.sum(epoch_type==1),
                     'algorithm':algo}, ignore_index=True)
        data = data.append({'stim_type':'false alarm', 
                     'val': np.sum(bad_types==3)/np.sum(epoch_type==1),
                     'algorithm':algo}, ignore_index=True)
        data = data.append({'stim_type':'correct rejection', 
                     'val': np.sum(bad_types==4)/np.sum(epoch_type==1),
                     'algorithm':algo}, ignore_index=True)
    ax = axs[i]

    sns.boxplot(data=data, x='stim_type', y='val', ax=ax)
    ax.set_title(algo)
    ax.set_ylim(0,0.11)
    ax.set_ylabel('Percentage bad')
    
plt.suptitle('Percentage of bad epochs per trial type for old recognized vs old forgotten images, minimum/maximum indicated')
plt.pause(0.1)
plt.tight_layout()
plt.savefig('plot_bad_epochs_H3.png')


#%% H4: difference between subsequent forgotten and subsequent remembered
fig, axs = plt.subplots(1,3)

algorithms = ['bad_epochs', 'bad_epochs_autoreject', 'bad_epochs_FASTER']
for i, algo in enumerate(algorithms):
    data = pd.DataFrame()

    for subj in subjects:
    
        bad_idx = report[subj][algo]
        epoch_type = np.array([int(x[-1]) for x in report[subj]['epoch_type']]) 
        # we only care about scene effects, encoded in the first int
        bad_types =  epoch_type[bad_idx]
        # bad_percentage = {'bad man-made': sum(bad_types==1)/sum(epoch_type==1), 
        #                   'bad natural': sum(bad_types==2)/sum(epoch_type==2),
        #                   'algorithm': algo}
        data = data.append({'stim_type':'subsequ. forgotten', 
                     'val': np.sum(bad_types==0)/np.sum(epoch_type==0),
                     'algorithm':algo}, ignore_index=True)
        data = data.append({'stim_type':'subsequ. remembered', 
                     'val': np.sum(bad_types==1)/np.sum(epoch_type==1),
                     'algorithm':algo}, ignore_index=True)

    ax = axs[i]

    sns.boxplot(data=data, x='stim_type', y='val', ax=ax)
    ax.set_title(algo)
    ax.set_ylim(0,0.11)
    ax.set_ylabel('Percentage bad')
    
plt.suptitle('Percentage of bad epochs per trial type for subsequent rememberered vs forgotten, minimum/maximum indicated')
plt.pause(0.1)
plt.tight_layout()
plt.savefig('plot_bad_epochs_H4.png')
