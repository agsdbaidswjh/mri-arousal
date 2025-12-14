import os, glob
import argparse
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from utils_nifti_conversion import *
from utils_bids_format import *

def main():
    parser = argparse.ArgumentParser(description="Preprocess fMRI data pipeline")
    parser.add_argument('--nordic', action='store_true',
                        help='If set, apply NORDIC correction after DICOM-to-NIfTI conversion')
    
    args = parser.parse_args()
    nordic = args.nordic
    print(f"NORDIC enabled: {nordic}")

    project_dir = Path.cwd()
    raw_fmri_dir = project_dir / 'raw_fmri'
    dicm2nii_dir = project_dir / 'dicm2nii'
    nii_dir = project_dir / 'nii'
    nordic_dir = nii_dir / 'nordic'
    bids = project_dir / "bids"
    events = project_dir / "raw_pupil"
    mods = ["func", "anat"]
    
    excluded_subjects = ['sub-000']  # subject to exclude
    
    # output directories
    nii_dir.mkdir(exist_ok=True)
    nordic_dir.mkdir(exist_ok=True)
    
    # NIfTI conversion
    subject_dirs = list(raw_fmri_dir.glob('*'))
    proc_subject_dirs = [d for d in subject_dirs if d.name not in excluded_subjects]
    
    print(f"Total number of subjects: {len(subject_dirs)}")
    print(f"{len(excluded_subjects)} subject(s) excluded: {excluded_subjects}")
    print(f"Converting {len(proc_subject_dirs)} subject(s) to NIfTI format.")
    
    for subject in tqdm(proc_subject_dirs, desc='NIfTI conversion', total=len(proc_subject_dirs)):
        process_dicm2nii(subject, nii_dir, dicm2nii_dir)

    print("NIfTI conversion completed! Yeeeeeee (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧")
    print("\n---------------------\n")

    # # NORDIC correction
    if nordic:
        nii_subject_dirs = list(nii_dir.glob('*sub*'))

        print(f"Total number of subjects to be processed with NORDIC: {len(nii_subject_dirs)} \n")
        
        Parallel(n_jobs=1, verbose=1, backend='loky')(
            delayed(process_nordic)(subj_dir, nordic_dir) for subj_dir in tqdm(nii_subject_dirs, 
                                                                            desc='NORDIC correction', 
                                                                            total=len(nii_subject_dirs)))
        
        print("\nNORDIC correction completed! Yeeeeeee (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧")

    # BIDS formatting
    if not (bids / "dataset_description.json").exists():
        create_dataset_description(bids)

    for subject in tqdm(nii_dir.iterdir(), desc="BIDS Formatting", total=len(list(nii_dir.iterdir()))):
        subj = subject.name

        if Path(os.path.join(bids, subj)).exists():
            print(f'Subject {subject.name} has already been BIDS formatted.')
        else:
            if subject.is_dir() and subject.name != "nordic":
                print(f"Processing subject: {subject.name}")
                sub_id = subject.name.replace("sub-", "")
                ses = create_bids_structure(bids, sub_id, mods)
                move_functional_files(subject, ses, sub_id)
                move_anatomical_files(subject, ses, sub_id)
                # move_fieldmap_files(subject, ses, sub_id)

    move_events_files(events, bids)
    update_json_metadata(bids)
    clean_events_tsv(bids)
    fmriprep_folders(project_dir)

if __name__ == "__main__":
    main()