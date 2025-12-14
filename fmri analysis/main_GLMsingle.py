import os, glob
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.masking import apply_mask
from nilearn.image import load_img
from glmsingle.glmsingle import GLM_single

subjects = input('Subject(s):') # multiple subjects need to be separated with commas
subjects = list(subjects.split(','))

home_dir = os.getcwd()
fmri_dir = os.path.join(home_dir, 'derivatives')
pupil_dir = os.path.join(home_dir, 'raw_pupil')
output_dir = os.path.join(home_dir, 'GLMsingle_outputs'); os.makedirs(output_dir, exist_ok=True)
figures_dir = os.path.join(home_dir, 'GLMsingle_figures'); os.makedirs(figures_dir, exist_ok=True)

# GLMsingle arguments
tr = 2
stimdur = 0.5

# default hyperparameters
opt = dict()
opt['wantlibrary'] = 1 # turn on HRF fitting 
opt['wantglmdenoise'] = 1 # turn on GLMdenoise
opt['wantfracridge'] = 1 # turn on fracridge
opt['wantmemoryoutputs'] = [1,1,1,1] # save all intermediate outputs to memory

glmsingle_obj = GLM_single(opt) # initialize object

for subj in subjects:
    subj = 'sub-' + subj
    print(f'\nProcessing subject {subj}')
    
    # necessary files
    events_files = sorted(glob.glob(os.path.join(pupil_dir, subj, '*events.tsv')))
    events_files = events_files[1:] # skip run-0

    params_files = sorted(glob.glob(os.path.join(pupil_dir, subj, '*params.csv')))
    params_files = params_files[1:] # skip run-0

    bold_files = sorted(glob.glob(os.path.join(fmri_dir, subj, 'ses-01', 'func', '*desc-preproc_bold.nii.gz')))
    mask_files = sorted(glob.glob(os.path.join(fmri_dir, subj, 'ses-01', 'func', '*desc-brain_mask.nii.gz')))

    # GLMsingle inputs for each subject
    design = []
    data = []

    for i in range(len(bold_files)):
        print(f'Loading files for run {i+1}...')
        events_df = pd.read_csv(events_files[i], sep='\t')
        params_df = pd.read_csv(params_files[i])
        bold_img = nib.load(bold_files[i])
        bold_data = bold_img.get_fdata()
        mask_data = nib.load(mask_files[i]).get_fdata().astype(bool)
        affine = bold_img.affine
        header = bold_img.header
        original_tr = header.get_zooms()[3]

        bold_masked = np.where(mask_data[..., None], bold_data, 0.0) # mask bold data with fmriprep mask

        bold_volumes = bold_data.shape[3]
        run_time = int(events_df['onset'].iloc[-1] + events_df['duration'].iloc[events_df['duration'].last_valid_index()])
        n_volumes = run_time // tr
        if n_volumes != bold_volumes:
            n_volumes = bold_volumes
            print(f'{subj} has {n_volumes} volumes')

        conditions = sorted(events_df['condition'].unique()) # ['blank', 'empty', 'left', 'right']
        dm = np.zeros((n_volumes, len(conditions) - 1)) # we do not code blank trials

        # create design matrix
        for i, cond in enumerate(params_df['condition']):
            if (cond == conditions[1]):  # empty
                dm[i * tr, 0] = 1
            elif cond == conditions[2]:  # left
                dm[i * tr, 1] = 1
            elif cond == conditions[3]:  # right
                dm[i * tr, 2] = 1
    
        design.append(dm)
        data.append(bold_masked)
    
    print('Dimensions:')
    print('Lists:', len(design), len(data))
    print('Matrix:', design[0].shape) 
    print('Bold:', data[0].shape)
    
    outputdir = os.path.join(output_dir, f'{subj}')
    figuredir = os.path.join(figures_dir, f'{subj}')

    start_time = time.time()

    print('\nRunning GLMsingle...')
    results_glmsingle = glmsingle_obj.fit(
       design,
       data,
       stimdur,
       tr,
       outputdir=outputdir,
       figuredir=figuredir)
    
    elapsed_time = time.time() - start_time
    print(f'\n Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')