import os, re, shutil, json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

SESSION_LABEL = "01"


def create_bids_structure(bids_root, subject_id, modalities):
    ses_folder = Path(bids_root) / f"sub-{subject_id}" / f"ses-{SESSION_LABEL}"
    for mod in modalities:
        (ses_folder / mod).mkdir(parents=True, exist_ok=True)
    return ses_folder


def construct_bids_filename(subject_id, info):
    parts = [f"sub-{subject_id}"]
    parts += [f"ses-{info['ses']}"] if "ses" in info else []
    parts += [f"{key}-{info[key]}" for key in ["task", "acq", "ce", "rec", "dir", "run", "echo", "part"] if key in info and info[key]]
    parts.append(info["suffix"])
    return "_".join(parts)


def create_dataset_description(bids_root):
    description = {
        "Name": "Gabor",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "Authors": ["Antreas Vasileiou", "ConsciousBrain Lab"]
    }

    with open(Path(bids_root) / "dataset_description.json", 'w') as f:
        json.dump(description, f, indent=4)


def move_anatomical_files(src_dir, dest_dir, subject_id):
    for nii_file in Path(src_dir).rglob("*T1w*.nii.gz"):
        info = {
            'modality': 'anat',
            'suffix': 'T1w',
            'ses': SESSION_LABEL
        }
        bids_filename = construct_bids_filename(subject_id, info)
        dest_nii = dest_dir / "anat" / f"{bids_filename}.nii.gz"
        shutil.copy(nii_file, dest_nii)

        json_file = nii_file.with_suffix('').with_suffix('.json')
        if json_file.exists():
            dest_json = dest_dir / "anat" / f"{bids_filename}.json"
            shutil.copy(json_file, dest_json)


def move_functional_files(src_dir, dest_dir, subject_id):
    task = "Gabor"
    for nii_mag in Path(src_dir).rglob("*magnitude*.nii.gz"):
        match = re.search(r'ADOT(\d+)', nii_mag.name)
        if not match:
            continue
        run_number = match.group(1)
        base_info = {
            'modality': 'func',
            'suffix': 'bold',
            'task': task,
            'run': run_number,
            'ses': SESSION_LABEL
        }
        mag_info = base_info.copy()
        mag_info['part'] = 'mag'
        bids_mag = construct_bids_filename(subject_id, mag_info)
        shutil.copy(nii_mag, dest_dir / "func" / f"{bids_mag}.nii.gz")

        phase_candidates = list(Path(src_dir).rglob(f"*ADOT{int(run_number)}*phase*.nii.gz"))
        if phase_candidates:
            nii_phase = phase_candidates[0]
            phase_info = base_info.copy()
            phase_info['part'] = 'phase'
            bids_phase = construct_bids_filename(subject_id, phase_info)
            shutil.copy(nii_phase, dest_dir / "func" / f"{bids_phase}.nii.gz")

        json_candidates = list(Path(src_dir).rglob(f"*ADOT{int(run_number)}*.json"))
        json_file = next((j for j in json_candidates if "magnitude" not in j.name and "phase" not in j.name), None)
        if json_file:
            json_mag_info = base_info.copy()
            bids_json_mag = construct_bids_filename(subject_id, json_mag_info)
            shutil.copy(json_file, dest_dir / "func" / f"{bids_json_mag}.json")


def move_fieldmap_files(src_dir, dest_dir, subject_id):
    for nii_mag in Path(src_dir).rglob("*magnitude*.nii.gz"):
        match = re.search(r'ADOTTopUp(\d+)', nii_mag.name)
        if not match:
            continue
        run_number = match.group(1)
        base_info = {
            'modality': 'fmap',
            'suffix': 'epi',
            'dir': 'SI',
            'run': run_number,
            'ses': SESSION_LABEL
        }
        mag_info = base_info.copy()
        bids_mag = construct_bids_filename(subject_id, mag_info)
        shutil.copy(nii_mag, dest_dir / "fmap" / f"{bids_mag}.nii.gz")
        json_candidates = list(Path(src_dir).rglob(f"*ADOTTopUp{int(run_number)}*.json"))
        json_file = next((j for j in json_candidates if "magnitude" not in j.name and "phase" not in j.name), None)
        if json_file:
            json_info = base_info.copy()
            json_info.pop('part', None)
            bids_json = construct_bids_filename(subject_id, json_info)
            shutil.copy(json_file, dest_dir / "fmap" / f"{bids_json}.json")

def move_events_files(events_src, bids_root):
    for subject_folder in Path(events_src).iterdir():
        if subject_folder.is_dir() and subject_folder.name.startswith("sub-") and subject_folder.name != "sub-000":
            subject_id = subject_folder.name.replace("sub-", "")
            for run_idx in range(1, 5):
                run = str(run_idx)
                src = subject_folder / f"sub-{subject_id}_ses-01_run-{run_idx}_Gabor_events.tsv"
                dest = Path(bids_root) / f"sub-{subject_id}" / "ses-01" / "func" / f"sub-{subject_id}_ses-01_task-Gabor_run-{run}_events.tsv"
                if src.exists():
                    shutil.copy(src, dest)

def update_json_metadata(bids_root):
    for json_file in Path(bids_root).rglob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            changed = False
            name = json_file.name
            subj = name.split("_")[0]
            run = name.split("_")[3] if "run" in name else None

            if name.endswith("_bold.json") or name.endswith("_T1w.json"):
                data["PhaseEncodingDirection"] = "k"
                changed = True
            elif name.endswith("_epi.json"):
                data["PhaseEncodingDirection"] = "k-"
                changed = True
            else:
                continue
            if name.endswith("_T1w.json"):
                if data.get("MRAcquisitionType") != "3D":
                    data["MRAcquisitionType"] = "3D"
                    changed = True
            if name.endswith("_bold.json"):
                if "TaskName" not in data:
                    data["TaskName"] = "Gabor"
                    changed = True
                if data.get("MRAcquisitionType") not in ["2D", "3D"]:
                    data["MRAcquisitionType"] = "2D"
                    changed = True
                if "Units" not in data:
                    data["Units"] = "rad"
                    changed = True
            if name.endswith("_epi.json"):
                if "Units" not in data:
                    data["Units"] = "rad"
                    changed = True
                if data.get("MRAcquisitionType") != "2D":
                    data["MRAcquisitionType"] = "2D"
                    changed = True
                intended_path = f"ses-01/func/{subj}_ses-01_task-Gabor_{run}_part-mag_bold.nii.gz"
                if data.get("IntendedFor") != intended_path:
                    data["IntendedFor"] = intended_path
                    changed = True
            if changed:
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to process {json_file}: {e}")

def clean_events_tsv(bids_root):
    for tsv_path in Path(bids_root).rglob("*_events.tsv"):
        try:
            df = pd.read_csv(tsv_path, sep='\t')
            for col in ['onset', 'duration']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            cols = ['onset', 'duration'] + [c for c in df.columns if c not in ['onset', 'duration']]
            df[cols].to_csv(tsv_path, sep='\t', index=False)
        except Exception as e:
            print(f"Error processing {tsv_path.name}: {e}")

def fmriprep_folders(project_dir):
    output_dir = Path(project_dir) / "derivatives"
    working_dir = Path(project_dir) / "work"
    output_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)