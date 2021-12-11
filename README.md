# EEGManyPipelines

Description and analysis plan for the EEGManyPipelines project


## 1. Preprocessing
### 1.0 manual data inspection
 detect bad channels by eyeballing, interpolate bads. potentially compare mit autoreject?note down how many channels were interpolated per participant
 
### 1.1 re-referencing
Re-referencing to average electrode (reasoning: to approximat electrode reference that is most generalizable to all electrodes)

### 1.2 filtering

1. Band: 0.1-100 Hz (FIR). Before/after epoching is not relevant, as jitter is only +-4ms
2. downsampling to 250 Hz (as for ERP no relevant frequencies are higher than that, and it's questionable if higher frequencies can even be measured reliably in EEG)
3. For ERPs: 35 LP + Notch 50 Hz
4. For Open questions: leave original LP

### 1.3 Artefact rejection

#### 1.3.1 Eye blinks & horizontal eye movements

1. ICA, find Eyeblink components from 64 components
2. blinks will be easy to detect
3. horizontal eye movements will be more difficult
4. check manually!
5. Only apply ICA correction to time spans where eye blinks / heog are actually observed
6. Apply to all channels
7. Are eyeblinks corrected? Visual random sample check

#### 1.3.2 artefact rejection

2. Epoch data based on trial trigger -200 to 500 ms
3. Apply artefact criteria per trial
    - same as [Bublatzky et al 2020]([https://doi.org/10.1016/j.cortex.2020.07.009](https://doi.org/10.1016/j.cortex.2020.07.009 "Persistent link using digital object identifier")) 
    - maximal allowed voltage step of 50 mV/ms;
    - maximal allowed difference of values in 200 msec intervals of 200 mV; 
    - minimal/maximal allowed amplitude of ± 100 mV

1. good/bad trials measurement, percentage bad per participant
2. calculate good/bad channels percentage per trial type / hypothesis interest

ERP

- AUC for time range


### notes:
EMP20 weird oscillations
EMP21 noisy

max/min amplitude of +-mV, and max diff of 200 mV in 200msec is the same?