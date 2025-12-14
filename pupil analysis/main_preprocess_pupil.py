# @author: J.W. de Gee
# @adjusted by: Antreas Vasileiou

import os, glob, datetime
from functools import reduce
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed, Memory # this is used for parallel processing c:
from tqdm import tqdm
from IPython import embed as shell # this is a breakpoint that can be used in the code to make it interactive

import utils_pupil_preprocess

# this a preset of plotting parameters (not important)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()
sns.set_palette("tab10")

# sets the cache directory for joblib (current working directory)
memory = Memory(os.path.expanduser('cache'), verbose=0)

def make_epochs(df, df_meta, locking, start, dur, measure, fs, baseline=False, b_start=-1, b_dur=1):

    # make sure we start with index 0:
    df_meta = df_meta.reset_index(drop=True)

    locking_inds = np.array(df['time'].searchsorted(df_meta[locking]).ravel())

    start_inds = locking_inds + int(start/(1/fs))
    end_inds = start_inds + int(dur/(1/fs)) - 1
    start_inds_b = locking_inds + int(b_start/(1/fs))
    end_inds_b = start_inds_b + int(b_dur/(1/fs))
    
    epochs = []
    for s, e, sb, eb in zip(start_inds, end_inds, start_inds_b, end_inds_b):
        epoch = np.array(df.loc[s:e, measure]) 
        if baseline:
            epoch = epoch - np.array(df.loc[sb:eb,measure]).mean()
        if s < 0:
            epoch = np.concatenate((np.repeat(np.NaN, abs(s)), epoch))
        epochs.append(epoch)
    epochs = pd.DataFrame(epochs)
    epochs.columns = np.arange(start, start+dur, 1/fs).round(5)
    if df_meta[locking].isna().sum() > 0:
        epochs.loc[df_meta[locking].isna(),:] = np.NaN

    return epochs

def make_plot(df, col_names, title, figs_dir, subj, ses, run):
    os.makedirs(figs_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10,5))
    for col in col_names:
        ax.plot(df[col])

    # ax.set_xlim(x_limit[0], x_limit[1])
    ax.set_title(title)
    ax.legend([col for col in col_names])
    fig.savefig(os.path.join(figs_dir, f'{subj}_{ses}_{run}_{title}.png'), dpi = 600)
    plt.close(fig)

# function decorator - the output of the function load_data() will be cached
# if you run the function again with the same arguments, it will return the cached output
@memory.cache
def load_data(filename, figs_dir, subj_dir, plot=True):

    import mne
    import utils_pupil_preprocess

    # needs to be changed to our naming convention!
    subj = os.path.basename(filename).split('_')[0]
    ses = os.path.basename(filename).split('_')[1]
    run = os.path.basename(filename).split('_')[2]

    # shell() # DEBUGGING

    # load pupil data:
    raw_et = mne.io.read_raw_eyelink(filename) # needs to be .asc file
    raw_et_df = raw_et.to_data_frame() # for some reason the pupil values are scaled by e+6
    pupil_column = [c for c in raw_et_df.columns if 'pupil' in c][0] # extract the column name of the pupil values (can be left or rigth pupil)
    raw_et_df[pupil_column] = raw_et.get_data()[2] # replace the scaled pupil values with the raw values

    # load events:
    events = raw_et.annotations.to_data_frame()
    events['onset'] = raw_et.annotations.onset

    # print event counts
    print(f"Total number of events: {len(events)}")
    print(f"Number of phase-1 events: {events['description'].str.match('.*phase-1$').sum()}")

    # interpolate blinks (mne built-in function):
    # works with EyeLink blink triggers
    mne.preprocessing.eyetracking.interpolate_blinks(
        raw_et, buffer=(0.1, 0.1), interpolate_gaze=True)

    # get in right shape:
    df = raw_et.to_data_frame() # same as raw_et_df but this one we overwrite with preprocessed data
    df[pupil_column] = raw_et.get_data()[2] # raw pupil values without scaling
    df = df.rename({pupil_column: 'pupil_int'}, axis=1) # rename the column to 'pupil_int'
    df['pupil'] = raw_et_df[pupil_column] # add the raw pupil values without interpolation for raw_et_df

    # preprocess pupil data:
    # fs = int(1/samples['time'].diff().median()*1000)
    fs = raw_et.info['sfreq']
    params = {'fs':fs, 'lp':10, 'hp':0.01, 'order':3}
    df = utils_pupil_preprocess.preprocess_pupil(samples=df, events=events, params=params)

    # plot interpolated blinks:
    # make_plot(df, ['pupil', 'pupil_int'], [300000, 400000], 'interpolated_blinks', figs_dir, subj, ses, run)
    # plot preprocessed pupil:
    print(f'Plotting...')
    if plot:
        make_plot(df, ['pupil', 'pupil_int', 'pupil_int_lp', 'pupil_int_lp_clean'], 
              'preprocessed', figs_dir, subj, ses, run)

    time_phase_0 = events.loc[events['description'].str.match('.*phase-0$'), 'onset'].astype(float).reset_index(drop=True)
    time_phase_1 = events.loc[events['description'].str.match('.*phase-1$'), 'onset'].astype(float).reset_index(drop=True)
    time_phase_2 = events.loc[events['description'].str.match('.*phase-2$'), 'onset'].astype(float).reset_index(drop=True)

    df_meta = pd.DataFrame({
        'time_phase_0': time_phase_0,  
        'time_phase_1': time_phase_1,
        'time_phase_2': time_phase_2  
        })
    df_meta['subject_id'] = subj
    df_meta['session_id'] = ses
    df_meta['run_id'] = run
    df_meta['trial_id'] = np.arange(df_meta.shape[0]) # maybe change :c

    order = ['subject_id', 'session_id', 'run_id', 'trial_id', 
             'time_phase_0', 'time_phase_1', 'time_phase_2']

    df_meta = df_meta[order]

    # make epochs:
    # this also needs to be changed to the correct pahse and time we want to look at
    columns = ['subject_id', 'session_id', 'run_id', 'trial_id']

    epochs = make_epochs(df=df, df_meta=df_meta, locking='time_phase_1', start=-0.5, dur=0.5, 
                         measure='pupil_int_lp_clean_psc', fs=fs, baseline=False, b_start=-1, b_dur=1)
    epochs[columns] = df_meta[columns]
    epochs_tonic = epochs.set_index(columns)

    epochs = make_epochs(df=df, df_meta=df_meta, locking='time_phase_1', start=-0.5, dur=0.5, 
                         measure='pupil_int_lp_clean', fs=fs, baseline=False, b_start=-1, b_dur=1)
    epochs[columns] = df_meta[columns]
    epochs_tonic_raw = epochs.set_index(columns)
    
    df_meta['nr_blinks'] = [events.loc[(events['description']=='BAD_blink') & 
            (events['onset']>df_meta['time_phase_0'].iloc[i]) &
            (events['onset']<(df_meta['time_phase_1'].iloc[i])),:].shape[0]
                for i in range(df_meta.shape[0])]
    df_meta['nr_sacs'] = [events.loc[(events['description']=='saccade') & 
            (events['onset']>df_meta['time_phase_0'].iloc[i]) &
            (events['onset']<(df_meta['time_phase_1'].iloc[i])),:].shape[0]
                for i in range(df_meta.shape[0])]
    
    # downsample:
    # this is to save some disk space since 1000Hz is more than enough
    # maybe we dont need to downsample since we have only a few participants 
    # epochs_tonic = epochs_tonic.iloc[:,::10]
    # epochs_tonic_raw = epochs_tonic_raw.iloc[:,::10]

    # add parameters:
    df_params = pd.read_csv(os.path.join(subj_dir, f"{subj}_{ses}_{run}_Gabor_events_params.csv"))
    columns = ["signal_present", "signal_orientation", "blank_trial", "target_orientation", 
           "condition", "color_congruency", "gabor_phase", "response", "correct", "RT"]
    df_meta = df_meta.merge(df_params[columns], left_index=True, right_index=True)
    order_f = ['subject_id', 'session_id', 'run_id', 'trial_id', "time_phase_0", "time_phase_1", "time_phase_2", 
               "blank_trial", "signal_present", "signal_orientation", "target_orientation", 
               "condition", "color_congruency", "gabor_phase", "response", "correct", "RT"]
    df_meta = df_meta[order_f]

    return df_meta, epochs_tonic, epochs_tonic_raw

project_dir = os.getcwd()
data_dir = os.path.join(project_dir, 'data')
figs_dir = os.path.join(project_dir, 'figs')

for subj_dir in glob.glob(os.path.join(data_dir, 'raw_pupil', '*')):
    subj = subj_dir.split('/')[-1]
 
    if subj == 'sub-000':
        continue # exclude subj-000 
    else:
        # convert edf to asc:
        edf_filenames = glob.glob(os.path.join(subj_dir, '*.edf'))

        for edf_f in edf_filenames:
            # skip the practice run
            if edf_f.split('_')[3] == 'run-0':
                continue
            else:
                if not os.path.exists(edf_f.split('.')[0] + '.asc'):
                    print(edf_f)
                    os.system('edf2asc {}'.format(edf_f))
                else:
                    print('skipping', edf_f)

        # load:
        n_jobs = 1
        asc_filenames = glob.glob(os.path.join(subj_dir, '*.asc'))
        # asc_filenames = [a for a in asc_filenames if (('sub-01' in a) & ('ses-01' in a) & ('run-2' in a))]
        print(len(asc_filenames))
        res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(load_data)(filename, figs_dir, subj_dir, plot=True) for filename in tqdm(asc_filenames))
        plt.close('all')

        # unpack:
        df = pd.concat([res[i][0] for i in range(len(res))])
        epochs_tonic = pd.concat([res[i][1] for i in range(len(res))])
        epochs_tonic_raw = pd.concat([res[i][2] for i in range(len(res))])

        # sort:
        df = df.sort_values(by=['subject_id', 'session_id', 'run_id', 'trial_id']).reset_index(drop=True)
        epochs_tonic = epochs_tonic.sort_values(by=['subject_id', 'session_id', 'run_id', 'trial_id'])
        epochs_tonic_raw = epochs_tonic_raw.sort_values(by=['subject_id', 'session_id', 'run_id', 'trial_id'])

        # get mean pupil size per trial:
        df["pupil_b_psc"] = epochs_tonic.mean(axis=1).values
        df["pupil_b_raw"] = epochs_tonic_raw.mean(axis=1).values

        # compare with fmri DataFrame: -> NOT RELEVANT NOW
        # df_fmri = pd.read_csv('/Users/jwdegee/Library/CloudStorage/OneDrive-UvA/deGee_eLife_2017/data/deGee_eLife_2017_all2.csv')
        # print(sp.stats.pearsonr(df['rt'], df_fmri['rt']))

        # save:
        df.to_csv(os.path.join(data_dir, f'pupil_meta_{subj}.csv'))
        epochs_tonic.to_hdf(os.path.join(data_dir, f'pupil_epochs_tonic_{subj}.hdf'), key='pupil')