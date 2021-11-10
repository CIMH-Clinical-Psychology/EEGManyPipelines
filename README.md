# EEGManyPipelines


Description and analysis plan for the EEGManyPipelines project

## 1. Preprocessing

### 1.1 re-referencing
1. detect bad channels by eyeballing, interpolate bads. potentially compare mit autoreject?
2. note down how many channels were interpolated per participant
3. Re-referencing to average electrode (reasoning: to approximat electrode reference that is most generalizable to all electrodes)

### 1.2 filtering
2. Band: 0.1-100 Hz (FIR)
3. Before/after epoching is not relevant, as jitter is only +-4ms
4. downsampling to 250 Hz (as for ERP no relevant frequencies are higher than that, and it's questionable if higher frequencies can even be measured reliably in EEG)
5. For ERPs: 35 LP + Notch 50 Hz
6. For Open questions: leave original LP

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
3.  [tbd criteria from paper or autoreject]
4. Apply artefact criteria per trial

...

1. good/bad trials measurement, percentage bad per participant
2. calculate good/bad channels percentage per trial type / hypothesis interest

ERP
- AUC for time range
