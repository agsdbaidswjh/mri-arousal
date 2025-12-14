#!/bin/bash
# Adapted from Andrew Jahn's script

bids_root_dir="/home/c13683446/Desktop/fmri/bids"
bids_out="/home/c13683446/Desktop/fmri/derivatives"
bids_work="/home/c13683446/Desktop/fmri/work"
FS_LICENSE="/home/c13683446/Desktop/fmri/freesurfer_license.txt"
start_time=$SECONDS # start timer

read -p "Enter subject number (e.g., 001): " subj # prompt for subject ID

docker run -ti --rm \
  -v "$bids_root_dir":/data:ro \
  -v "$bids_out":/out \
  -v "$bids_work":/work \
  -v "$FS_LICENSE":/opt/freesurfer/license.txt \
  nipreps/fmriprep:23.2.1 \
  /data /out participant \
  --participant-label "$subj" \
  --fs-license-file /opt/freesurfer/license.txt \
  --work-dir /work

elapsed=$(( SECONDS - start_time )) # calculate elapsed time
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
