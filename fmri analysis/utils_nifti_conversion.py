import os, glob
import subprocess
from pathlib import Path
from tqdm import tqdm

# convert raw fmri files to nifti files using the MATLAB dicm2nii package
def run_dicm2nii_mat(subj_dir, out_dir, dicm2nii_dir):
    matlab_script_path = Path('run_dicm2nii.m') # creates a temporary matlab file in the working directory 

    # temporary MATLAB script
    matlab_script_path.write_text(f"""
        setpref('dicm2nii_gui_para', 'save_json', true);
        addpath('{dicm2nii_dir}');
        dicm2nii('{subj_dir}', ...
                '{out_dir}', ...
                'nii.gz');
        exit;
        """)
    
    # call matlab
    try:
        subprocess.run(['matlab', '-batch', 'run_dicm2nii'], check=True)
        print("Conversion completed! Yeeeeeee (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧")
    finally:
        # delete temporary matlab script
        if matlab_script_path.exists():
            matlab_script_path.unlink() # way of deleting files in python
            print("Temporary MATLAB script... get the fuck outta here!")
            print("MATLAB SUCKS")
    
    print(f"Processed: {subj_dir.split('/')[-1]}")

# apply NORFIC correction to nifti files
def run_nordic_mat(mag_path, phase_path, out_dir, out_file_name, gz=False):
    matlab_script_path = Path('run_nordic.m') # creates a temporary matlab file in the working directory 
    
    gz_arg = 'ARG.write_gzipped_niftis = 1;' if gz else ''

    # temporary MATLAB script
    matlab_script_path.write_text(f"""
        addpath('/home/c13683446/Desktop/fmri/nordic');
        ARG = struct();
        ARG.temporal_phase = 1;
        ARG.phase_filter_width = 10;
        ARG.DIROUT = '{out_dir}';
        {gz_arg}
        NIFTI_NORDIC('{mag_path}', '{phase_path}', '{out_file_name}', ARG);
        exit;
        """)
    
    # call matlab
    try:
        subprocess.run(['matlab', '-batch', 'run_nordic'], check=True)
        print("Conversion completed! Yeeeeeee (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧")
    finally:
        # delete temporary matlab script
        if matlab_script_path.exists():
            matlab_script_path.unlink() # way of deleting files in python
            print("Temporary MATLAB script... get the fuck outta here!")
            print("MATLAB SUCKS")

    print(f"Processed: {mag_path.split('/')[-1]}")

def process_dicm2nii(subj_dir, nii_base_dir, dicm2nii_dir):
    subj = Path(subj_dir).name # subject id
        
    subj_nii_dir = Path(nii_base_dir) / subj # create path for converted nifti files
    subj_nii_dir.mkdir(parents=True, exist_ok=True) # checks if directory exists; if not creates it
    
    # check if the subject has already been converted
    if subj_nii_dir.exists() and any(subj_nii_dir.glob("*.nii.gz")):
        print(f"Subject {subj} has already been converted to NIfTI format.")
        return
    else:
        print(f"Processing subject {subj}...")
        try:
            run_dicm2nii_mat(subj_dir=str(subj_dir), out_dir=str(subj_nii_dir), dicm2nii_dir=dicm2nii_dir)
        except Exception as e:
            return f"Error processing {subj}: {str(e)}"

def process_nordic(subj_dir, nordic_base_dir):
    subj = Path(subj_dir).name # subject id
    
    # grab and sort magnitude and phase files
    magnitude_files = sorted(glob.glob(os.path.join(subj_dir, '*magnitude.nii.gz')))
    phase_files = sorted(glob.glob(os.path.join(subj_dir, '*phase.nii.gz')))
    
    # sanity check
    if len(magnitude_files) != len(phase_files):
        return f"Warning for {subj}: Unequal number of magnitude and phase files!"
    
    # create path for corrected NORDIC files
    subj_nordic_dir = Path(nordic_base_dir) / subj
    subj_nordic_dir.mkdir(parents=True, exist_ok=True)

    # check if the subject has already been processed
    if subj_nordic_dir.exists() and any(subj_nordic_dir.glob("NORDIC*")):
        print(f"Subject {subj} has already been processed with NORDIC.")
        return
    else:
        print(f"\nProcessing NORDIC correction for subject {subj}...")
        for mag, phase in tqdm(zip(magnitude_files, phase_files), total=len(magnitude_files), desc=f"NORDIC correction for {subj}"):
            mag_path = Path(mag)
            out_dir = str(subj_nordic_dir) + '/'  # NORDIC needs trailing slash
            out_file_name = f"NORDIC_{mag_path.name.split('.')[0]}" # name of output file
            
            print(f"\nProcessing {mag_path.name} and {Path(phase).name}...")
            try:
                run_nordic_mat(mag_path=mag, phase_path=phase, out_dir=out_dir, out_file_name=out_file_name)
                print('\n')
            except Exception as e:
                return f"Error processing {mag_path.name} for subject {subj}: {str(e)}"