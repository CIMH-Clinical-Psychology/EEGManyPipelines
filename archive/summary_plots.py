# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:25:03 2022

@author: Simon
"""


#%% custom reject function

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
