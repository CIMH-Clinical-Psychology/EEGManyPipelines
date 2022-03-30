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
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, read_ica
from psutil import virtual_memory as vram
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm # fancy progress bars
from config import md5hash
from natsort import natsorted
from joblib.memory import Memory
import scipy.stats as stats
from mne import EvokedArray, EpochsArray, Evoked
from collections import namedtuple


def get_ch_neighbours(ch_name, n=9, plot=False):
    """retrieve the n neighbours of a given electrode location.
    Count includes the given origin electrode location"""
    montage = mne.channels.read_custom_montage(f'{config.data_dir}/channel_locations/chanlocs_besa.txt')
    positions = montage.get_positions()['ch_pos']
    
    Point = namedtuple('Point', 'name x y z')
    ch = Point(ch_name, *positions[ch_name])   
    chs = [Point(ch, *pos) for ch, pos in positions.items()]
    chs = [ch for ch in chs if not (('EOG' in ch.name) or ('IO' in ch.name))]
 
    dist = lambda p: (p.x - ch.x)**2 + (p.y - ch.y)**2 + (p.z - ch.z)**2
    
    chs_sorted = sorted(chs, key=dist)
    
    chs_out = [ch.name for ch in chs_sorted[:n]]
    
    if plot:
        montage.plot(show_names=chs_out)
    return chs_out
    
memory = Memory(config.cache_dir)

np.random.seed(0)


#%% load data
# load all subjects from disk either into RAM or into a memmap
subjects = []
data_dir = config.data_dir + '/eeg_brainvision/'
plot_dir = data_dir + '/plots/'
subjects = sorted([f.name for f in os.scandir(data_dir) if f.is_dir() and 'EMP' in f.name])
os.makedirs(plot_dir, exist_ok=True)


montage = mne.channels.read_custom_montage(f'{config.data_dir}/channel_locations/chanlocs_besa.txt')


data_erp = {} # this data is resampled (sfreq=250)
data_freq = {} # this data is not resampled (sfreq=512)


for subj in tqdm(subjects, desc='load epochs'):
    
    subj_dir = f'{data_dir}/{subj}/'
    epochs_erp_fif = f'{subj_dir}/{subj}_resampled-epo.fif'
    epochs_freq_fif = f'{subj_dir}/{subj}_original-epo.fif'

    bads_autoreject = [f'{subj_dir}/{f}'for f in os.listdir(subj_dir) if 'autoreject' in f]
    
    assert len(bads_autoreject)>0, 'no autoreject annotations found'
    if len(bads_autoreject)>1:
        warnings.warn('several autorejects found, taking most recent w.r.t fif epochs file')
        mtime = os.path.getmtime(epochs_erp_fif)
        func = lambda x: abs(os.path.getmtime(x)-mtime)
        bads_autoreject = sorted(bads_autoreject, key=func)
        
    with open(bads_autoreject[0], 'r') as f:
        bad_epochs = json.load(f)
        print(subj, bad_epochs)
        
    # load data
    data_erp[subj] = mne.read_epochs(epochs_erp_fif, verbose='ERROR').drop(bad_epochs)
    data_freq[subj] = mne.read_epochs(epochs_freq_fif, verbose='ERROR').drop(bad_epochs)
    
ch_names = data_erp[subj].ch_names
info = data_erp[subj].info
grand_avg = mne.grand_average([epochs.average() for epochs in data_erp.values()])
results = pd.DataFrame()


stop
#%%  

fig_n100_joint = grand_avg.copy().crop(0, 0.2).plot_joint(np.linspace(0.09, 0.15, 7), title='Grand average between 90-150ms')
fig_n400_joint = grand_avg.plot_joint(np.linspace(0.3, 0.5, 11), title='Grand average between 300-500ms')

fig_n100_topo = grand_avg.copy().crop(0.08, 0.15).plot_topo(vline=[0.08, 0.1,  0.12, 0.14], title='Grand average between 90-150ms, lines at 80ms, 100ms, 120ms, 140ms')
fig_n400_topo = grand_avg.copy().crop(0.3, 0.5).plot_topo(vline=[0.3,  0.4, 0.5], title='Grand average between 300-500ms, lines at 300ms, 400ms, 500ms')

plt.pause(0.1)

for name, var in list(locals().items()): # so hacky, lol
    if not isinstance(var, plt.Figure): continue
    print(name, var)
    var.savefig(f'{plot_dir}/grand_avg_{name[4:]}.png')
    
# stop


#%% make plots

# evoked = epochs.average()
# evoked.plot

#%% H1: effect of scene category
# There is an effect of scene category (i.e., a difference between images showing
# man-made vs. natural environments) on the amplitude of the N1 component, i.e. the
# first major negative EEG voltage deflection.

# Find best electrode to detect the N1
# 1) make avg of trials per subj per electrode
# 2) Find first negative peak 50-120ms, see which electrode has lowest amplitude
# 3) use this electrode and all electrodes around it for further calculations

# calculate component
# 1) calculate repeated measures ttest on +-5ms on averages per participant per condition
# 2) run this on data on all sensors on max neg peak +-35ms
# Maris, E., Oostenveld, R., 2007. Nonparametric statistical testing of EEG-and MEG-data.
# J. Neurosci. Methods 164 (1), 177?190

# maximum peak != maximum difference

conditions = ['man-made', 'natural']

tmin, tmax = [0.08, 0.15]
erp_range = 0.005 # 5ms

ERP_ch, ERP_tp = grand_avg.get_peak(tmin=tmin, tmax=tmax, mode='neg')

chs = get_ch_neighbours(ERP_ch, 1)

grands = {}
erps = {} # ERP is a dictionary with conditions as keys and a list of ERP averages for each subject
erps_values = {}

# for ERP_tp in np.arange(120, 130)/1000:
for cond in conditions:
    # erp_range=0.001
    erps[cond] = [epochs[cond].average(chs).crop(0, tmax) for epochs in data_erp.values()]
    erps_values[cond] = [erp.get_data(tmin=ERP_tp-erp_range, tmax=ERP_tp+erp_range).mean() for erp in erps[cond]]
    grands[cond] = mne.grand_average([epochs[cond].average(chs) for epochs in data_erp.values()])

p = stats.ttest_rel(erps_values[conditions[0]], erps_values[conditions[1]])

statistics = pd.Series({**{f'cond{i}':cond for i, cond in enumerate(conditions)},
                   **{f'val{i}':f'{np.mean(erps_values[cond])*1e6:.2f}+-{np.std(erps_values[cond])*1e6:.2f}'for i, cond in enumerate(conditions)},
                   'p':p, 
                   'chs':chs, 
                   't':f'{ERP_tp} +- {erp_range} (mean)',
                   'n_subj':len(data_erp),
                   **{f'n_cond{i}':erps_values[cond] for i, cond in enumerate(conditions)}
                   },
                  name='H1')

results = results.append(statistics)

# plotting
fig_n100 = mne.viz.plot_compare_evokeds(erps, show_sensors=False, vlines=[0, ERP_tp], combine='means',
                                  title=f'N100 ERP for {conditions} on {chs} at {ERP_tp}+-5ms')[0]

plt.fill_between([ERP_tp-erp_range, ERP_tp+erp_range], *plt.ylim(), alpha=0.3, color='green')
plt.legend([conditions[0], '95%', conditions[1], '95%'])


#%% Using cluster analysis
cluster_data = {}
cluster_range = 0.035 # 35ms

tmin = ERP_tp-cluster_range
tmax = ERP_tp+cluster_range

# for ERP_tp in np.arange(120, 130)/1000:
for cond in conditions:
    cluster_data_= [epochs[cond].get_data(tmin=tmin, tmax=tmax).mean(0) for epochs in data_erp.values()]
    cluster_data[cond] =  np.stack(cluster_data_)
    
X = [np.transpose(x, (0, 2, 1)) for x in (cluster_data.values())]

adjacency, _ = mne.channels.find_ch_adjacency(info, 'eeg')
t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(X, adjacency=adjacency, tail=0, 
                                                                         n_permutations=1000, n_jobs=-1, threshold=1.96)
significant_points = np.zeros_like(X[0][0], dtype=bool).T
for _cluster in clusters:
    for x, y in zip(*_cluster):
        significant_points[y, x] = True
        
evoked_diff = mne.combine_evoked(list(grands.values()) , weights=[1, -1]).crop(tmin, tmax)
fig_n100_diff = evoked_diff.plot_joint(times = np.linspace(tmin, tmax, 7), title=f'Difference {" - ".join(conditions)}')

fig_n100_clust = evoked_diff.plot_image(mask=np.pad(significant_points, [1,1]), show_names='all')


#%% H2: effect of novelty
# There are effects of image novelty (i.e., between images shown for the first time/new
# vs. repeated/old images) within the time-range from 300?500 ms ...
#    a. ... on EEG voltage at fronto-central channels.
#    b. ... on theta power at fronto-central channels.
#    c. ... on alpha power at posterior channels

#%% **H2a ERP N400
# 1. make avg of trials per subj per electrode
# 2. plot grand averages from 300-500ms, then choose electrodes to make the calculation
# 3. use this electrode and all frontal electrodes around it for further calculations

# calculate component
# 1) calculate repeated measures ttest on 300-500ms averages per participant per condition, 
#     i.e. two value per participant, for each condition one
# 2) run this on data on all sensors on max neg peak +-35ms
# Maris, E., Oostenveld, R., 2007. Nonparametric statistical testing of EEG-and MEG-data.
# J. Neurosci. Methods 164 (1), 177?190

conditions = ['new', 'old']
tmin, tmax = [0.3, 0.5]

ERP_ch, _ = grand_avg.get_peak(tmin=tmin, tmax=tmax, mode='neg')

chs = get_ch_neighbours(ERP_ch, 7)
if 'Cz' in chs: chs.remove('Cz')

grands = {}
erps = {}
erps_values = {}
for cond in conditions:
    erps[cond] = [epochs[cond].average(chs).crop(0, 0.8) for epochs in data_erp.values()]
    erps_values[cond] = [erp.get_data(tmin=tmin, tmax=tmax).mean() for erp in erps[cond]]
    grands[cond] = mne.grand_average([epochs[cond].average(chs) for epochs in data_erp.values()])

p = stats.ttest_rel(erps_values[conditions[0]], erps_values[conditions[1]])

statistics = pd.Series({**{f'cond{i}':cond for i, cond in enumerate(conditions)},
                   **{f'val{i}':f'{np.mean(erps_values[cond])*1e6:.2f}+-{np.std(erps_values[cond])*1e6:.2f}'for i, cond in enumerate(conditions)},
                   'p':p, 
                   'chs':chs, 
                   't':f'{tmin} - {tmax}',
                   'n_subj':len(data_erp),
                   **{f'n_cond{i}':erps_values[cond] for i, cond in enumerate(conditions)}
                   },
                   name='H2')

results = results.append(statistics)


# plotting
fig_n400 = mne.viz.plot_compare_evokeds(erps, vlines=[0, 0.3, 0.5], show_sensors=True, combine='mean',
                                  title=f'N400 ERP for {conditions} on {chs} at {tmin}s - {tmax}')[0]

ax = fig_n400.axes[0]
ax.fill_between([tmin, tmax], *ax.get_ylim(), alpha=0.3, color='green', )
ax.legend([conditions[0], '95%', conditions[1], '95%'])
evoked_diff = mne.combine_evoked(list(grands.values()) , weights=[1, -1]).crop(tmin, tmax)
fig_n100_diff = evoked_diff.plot_joint(times = np.linspace(tmin, tmax, 7), title=f'Difference {" - ".join(conditions)}')

# stop

#%% **H2b spectra for theta (4-7 Hz)
# 1. calculate spectra for theta (4-7 Hz) for the time range using FFT or Wavelet (depending on what is used later)
# 2. make fft per each trial, average 

from mne.time_frequency import tfr_morlet, psd_welch, psd_multitaper, tfr_stockwell

conditions = ['new', 'old']
fmin, fmax = [4, 7]
tmin, tmax = [0.3, 0.5]
freqs = np.arange(2, 20, .5)
n_cycles = freqs / 2.  # different number of cycle per frequency

ch_fronto_central = [ch for ch in info.ch_names if 'FC' in ch]
chs = ch_fronto_central

df = pd.DataFrame({'type':[], 'subj':[], 'cond':[], 'power':[]})
power_values = {cond:[] for cond in conditions}
spectra = {cond:[] for cond in conditions}
spectra_avgs = {}

for cond in conditions:
    for subj in tqdm(data_freq):
        epochs = data_freq[subj][cond]

        power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, 
                           return_itc=False, n_jobs=-1)
        spectra[cond].append(power.copy())
        
        power.crop(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        power.pick(chs)
        mean_pow = power.data.mean()
        power_values[cond].append(mean_pow)
        
        df.loc[len(df)] = ['morlet', subj, cond, mean_pow]
    spectra_avgs[cond] = np.mean(spectra[cond]).crop(None, 0.7)
    
p = stats.ttest_rel(power_values[conditions[0]], power_values[conditions[1]])

statistics = pd.Series({**{f'cond{i}':cond for i, cond in enumerate(conditions)},
                   **{f'val{i}':f'{np.mean(power_values[cond]):.2e}+-{np.std(power_values[cond]):.2e}'for i, cond in enumerate(conditions)},
                   'p':p, 
                   'chs':chs, 
                   't':f'{tmin} - {tmax}',
                   'freq': f'{fmin} - {fmax}',
                   'n_subj':len(data_erp),
                   },
                   name='H2b')

timefreqs = {(0.4, 5.5):(0.1, 3)}
fig_theta_cond1 = spectra_avgs[conditions[0]].plot_joint(tmin=0, tmax=0.7, timefreqs=timefreqs, picks=chs,
                                                         title=f'Time-Frequency for {conditions[0]}')
fig_theta_cond2 = spectra_avgs[conditions[1]].plot_joint(tmin=0, tmax=0.7, timefreqs=timefreqs, picks=chs,
                                                         title=f'Time-Frequency for {conditions[1]}')
avg_trf_diff = spectra_avgs[conditions[0]] - spectra_avgs[conditions[1]]
fig_avg_theta_diff = avg_trf_diff.plot_joint(tmin=0, tmax=0.7, timefreqs=timefreqs, picks=chs,
                                             title=f'Time-Frequency difference for {conditions[0]}-{conditions[1]}')

results = results.append(statistics)
   
fig, axs = plt.subplots(1, 2); axs=axs.ravel()
sns.violinplot(data=df, x='cond', y='power', ax=axs[0])
plt.title(f'Power at {fmin}-{fmax} between {tmin}-{tmax}')

ax = axs[1]
for cond in conditions:
    ax.scatter(np.arange(len(power_values[cond])), power_values[cond])
plt.title(f'Power for individual participants for {fmin}/{fmax} Hz on \n{chs}')

#%% **H2c spectra for alpha (8-14 Hz) 
# 1. calculate spectra for alpha (8-14 Hz) for the time range using FFT or Wavelet (depending on what is used later)
# 2. 

from mne.time_frequency import tfr_morlet, psd_welch, psd_multitaper, tfr_stockwell

conditions = ['new', 'old']
fmin, fmax = [8, 14]
tmin, tmax = [0.3, 0.5]
freqs = np.arange(2, 20, .5)
n_cycles = freqs / 2.  # different number of cycle per frequency

ch_occipital = [ch for ch in info.ch_names if 'P'==ch[0]]

df = pd.DataFrame({'type':[], 'subj':[], 'cond':[], 'power':[]})
power_values = {cond:[] for cond in conditions}
spectra = {cond:[] for cond in conditions}
spectra_avgs = {}

chs = ch_occipital

for cond in conditions:
    for subj in tqdm(data_freq):
        epochs = data_freq[subj][cond]

        power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, 
                           return_itc=False, n_jobs=-1)
        spectra[cond].append(power.copy())
        
        power.crop(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        power.pick(chs)
        mean_pow = power.data.mean()
        power_values[cond].append(mean_pow)
        
        df.loc[len(df)] = ['morlet', subj, cond, mean_pow]
    spectra_avgs[cond] = np.mean(spectra[cond]).crop(None, 0.7)
    
p = stats.ttest_rel(power_values[conditions[0]], power_values[conditions[1]])

statistics = pd.Series({**{f'cond{i}':cond for i, cond in enumerate(conditions)},
                   **{f'val{i}':f'{np.mean(power_values[cond]):.2e}+-{np.std(power_values[cond]):.2e}'for i, cond in enumerate(conditions)},
                   'p':p, 
                   'chs':chs, 
                   't':f'{tmin} - {tmax}',
                   'freq': f'{fmin} - {fmax}',
                   'n_subj':len(data_erp),
                   },
                   name='H2c')

timefreqs = {(0.4, 11):(0.1, 6)}
fig_alpha_cond1 = spectra_avgs[conditions[0]].plot_joint(tmin=0, tmax=0.7, timefreqs=timefreqs, picks=chs,
                                                         title=f'Time-Frequency for {conditions[0]}')
fig_alpha_cond2 = spectra_avgs[conditions[1]].plot_joint(tmin=0, tmax=0.7, timefreqs=timefreqs, picks=chs,
                                                         title=f'Time-Frequency for {conditions[1]}')
avg_trf_diff = spectra_avgs[conditions[0]] - spectra_avgs[conditions[1]]
fig_avg_theta_diff = avg_trf_diff.plot_joint(tmin=0, tmax=0.7, timefreqs=timefreqs, picks=chs,
                                             title=f'Time-Frequency difference for {conditions[0]}-{conditions[1]}')


results = results.append(statistics)
   
fig, axs = plt.subplots(1, 2); axs=axs.ravel()
sns.violinplot(data=df, x='cond', y='power', ax=axs[0])
plt.title(f'Power at {fmin}-{fmax} between {tmin}-{tmax}')

ax = axs[1]
for cond in conditions:
    ax.scatter(np.arange(len(power_values[cond])), power_values[cond])
plt.title(f'Power for individual participants for {fmin}/{fmax} Hz on \n{chs}')
#%% **H3a hit/miss voltage
# There are effects of successful recognition of old images (i.e., a difference between
# old images correctly recognized as old [hits] vs. old images incorrectly judged as new
# [misses]) on EEG voltage at any channels, at any time

# calculate cluster analysis from time 0ms - 500ms and from 300ms-500ms

conditions = ['hit', 'miss']
tmin, tmax = [0.0, 0.5]
erp_range = 0.005 # 5ms

cluster_data = {}
grands = {}
# for ERP_tp in np.arange(120, 130)/1000:
for cond in conditions:
    cluster_data_ = [epochs[cond].get_data(tmin=tmin, tmax=tmax).mean(0) for epochs in data_erp.values()]
    cluster_data[cond] =  np.stack(cluster_data_)
    grands[cond] = mne.grand_average([epochs[cond].average() for epochs in data_erp.values()])
    
X = [np.transpose(x, (0, 2, 1)) for x in (cluster_data.values())]

adjacency, _ = mne.channels.find_ch_adjacency(info, 'eeg')
t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(X, adjacency=adjacency, tail=0, 
                                                                         n_permutations=1000, n_jobs=-1, threshold=1.96)
significant_points = np.zeros_like(X[0][0], dtype=bool).T
for _cluster in clusters:
    for x, y in zip(*_cluster):
        significant_points[y, x] = True
        
evoked_diff = mne.combine_evoked(list(grands.values()) , weights=[1, -1]).crop(tmin, tmax)
fig_h3_diff = evoked_diff.plot_joint(times = np.linspace(tmin, tmax, 7), title=f'Difference {" - ".join(conditions)}')
fig_h3_clust = evoked_diff.plot_image(mask=np.pad(significant_points, [1,0]), show_names='all')
plt.title(f'Cluster contrast difference {" - ".join(conditions)}')



#%% **H4a remembered/forgotten voltage
# There are effects of subsequent memory (i.e., a difference between images that will
# be successfully remembered vs. forgotten on a subsequent repetition) on EEG voltage at 
# any channels, at any time.
conditions = ['hit/remembered', 'hit/forgotten']
tmin, tmax = [0.0, 0.5]
erp_range = 0.005 # 5ms

cluster_data = {}
grands = {}
# for ERP_tp in np.arange(120, 130)/1000:
for cond in conditions:
    cluster_data_ = [epochs[cond].get_data(tmin=tmin, tmax=tmax).mean(0) for epochs in data_erp.values()]
    cluster_data[cond] =  np.stack(cluster_data_)
    grands[cond] = mne.grand_average([epochs[cond].average() for epochs in data_erp.values()])
    
X = [np.transpose(x, (0, 2, 1)) for x in (cluster_data.values())]

adjacency, _ = mne.channels.find_ch_adjacency(info, 'eeg')
t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(X, adjacency=adjacency, tail=0, 
                                                                         n_permutations=1000, n_jobs=-1, threshold=1.96)
significant_points = np.zeros_like(X[0][0], dtype=bool).T
for _cluster in clusters:
    for x, y in zip(*_cluster):
        significant_points[y, x] = True
        
evoked_diff = mne.combine_evoked(list(grands.values()) , weights=[1, -1]).crop(tmin, tmax)
fig_h3_diff = evoked_diff.plot_joint(times = np.linspace(tmin, tmax, 7), title=f'Difference {" - ".join(conditions)}')
fig_h3_clust = evoked_diff.plot_image(mask=np.pad(significant_points, [1,0]), show_names='all')
plt.title(f'Cluster contrast difference {" - ".join(conditions)}')
