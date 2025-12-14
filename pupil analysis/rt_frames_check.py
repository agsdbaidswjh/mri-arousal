import os
import glob
import re
import pandas as pd

project_dir = os.getcwd()
data_dir = os.path.join(project_dir, 'data')

target_durations = {
    'baseline': 0.500,
    'stimulus and response': 0.500,
    'ITI': 3.000
}

for subj_dir in glob.glob(os.path.join(data_dir, 'raw_pupil', '*')):
    # get subject ID
    subj = subj_dir.split('/')[-1]

    if subj == 'sub-000':
        continue
    else:
        pupil_files = glob.glob(os.path.join(subj_dir, '*events.tsv'))
        
        for pfile in pupil_files:
            print(pfile)
            subj = re.search(r'sub-\d+', pfile)[0]
            ses = re.search(r'ses-\d+', pfile)[0]
            run = re.search(r'run-\d+', pfile)[0]
            
            df = pd.read_csv(pfile, sep='\t')
            first_baseline = df[df['event_type'] == 'baseline'].index.min()
            last_ITI = df[df['event_type'] == 'ITI'].index.max()
            df = df.drop([first_baseline, last_ITI])

            # identifies trials that deviate from the target duration by more than 1%
            for idx, row in df.iterrows():
                event = row['event_type']
                duration = row['duration']
                frames = row['nr_frames']

                # skip events that we don't care about
                if event not in target_durations:
                    continue
                
                # allow for a 1% variance in the target duration
                target = target_durations[event]
                lb = target - (target * 0.01)
                ub = target + (target * 0.01)

                if not (lb <= duration <= ub):
                    trial = int(row['trial'])

                    # ingore the first and last trial
                    # if (trial == 0 and event == 'baseline') or (trial == max(df['trial_nr']) and event == 'ITI'):
                    #     continue
                    
                    print(f"Warning: {subj}/{ses}/{run}/trial-{trial} has a {event} event with duration {duration} seconds")
                
            print('-----------------------------------')
            
            descriptives = []

            for event in target_durations.keys():
                event_df = df[df['event_type'] == event]

                ev_descriptives = {
                    'event_type': event,
                    'min_duration': event_df['duration'].min(),
                    'max_duration': event_df['duration'].max(),
                    'mean_duration': event_df['duration'].mean(),
                    'min_frames': event_df['nr_frames'].min(),
                    'max_frames': event_df['nr_frames'].max(),
                    'mean_frames': event_df['nr_frames'].mean()
                }

                descriptives.append(ev_descriptives)
            
            descriptives_df = pd.DataFrame(descriptives)
            descriptives_df.to_csv(os.path.join(subj_dir, f'{subj}_{ses}_{run}_descriptives.csv'))