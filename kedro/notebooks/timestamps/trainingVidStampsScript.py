import pandas as pd
import os
from datetime import timedelta

def parse_time(t):
    # First, check if t is NaN or None
    if pd.isna(t):
        return timedelta(0)  # Return a zero timedelta if the input is not a valid string

    t = str(t)  # Ensure t is a string to handle cases where it might be read as a different type
    
    if ':' in t:  # Checks if the time is in hh:mm:ss format
        parts = t.split(':')
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    else:  # Handles the (hh, mm, ss) format
        t = t.replace(' ', '')  # Remove any spaces to prevent errors during eval
        parts = t.strip('()').split(',')
        corrected_parts = [str(int(part)) for part in parts]  # Convert parts to integers to remove leading zeros
        corrected_time_str = '(' + ', '.join(corrected_parts) + ')'
        hours, minutes, seconds = eval(corrected_time_str)
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
def convert_time_format(time_str):
    # Strip the parentheses and split by comma
    time_parts = time_str.strip().split(':')
    # Format the parts into hh:mm:ss
    return f"{int(time_parts[0]):02}:{int(time_parts[1]):02}:{int(time_parts[2]):02}"

from pathlib import Path
# Load the offset times
offsets = pd.read_csv(Path(r'D:\Kartik\IIITB\Study\Sem 3\PE\Kedro\kartik-eeg\data\02_intermediate\video_data\16_to_30\offset_times.csv'))
offsets['offset'] = offsets['offset'].apply(parse_time)

# Load the time sheet data
time_data_directory = Path(r'D:\Kartik\IIITB\Study\Sem 3\PE\Kedro\kartik-eeg\data\02_intermediate\video_data\16_to_30\tasksStamps')

# Prepare the directory path to the stamps files
directory_path = Path(r'D:\Kartik\IIITB\Study\Sem 3\PE\Kedro\kartik-eeg\data\02_intermediate\video_data\16_to_30\vid_stamps')

# Initialize the dataframe for the final output
final_df = pd.DataFrame(columns=['participant', 'task', 'file', 'start', 'end'])

# Process each file in the stamps directory
# print(os.listdir(directory_path))
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        # Read the current stamps file
        current_file_path = os.path.join(directory_path, filename)
        current_data = pd.read_csv(current_file_path)

        # print(current_data)

        # Extract subject from filename
        subject = filename[:2]
        # print(subject)
        
        # Determine Task and File from time_data based on adjusted start
        for index, row in current_data.iterrows():
            end_seconds = parse_time(row['end']).total_seconds()

            time_data = pd.read_csv(os.path.join(time_data_directory, filename))

            mask = (end_seconds<=time_data['end'].apply(lambda x: parse_time(x).total_seconds()))
            
            # print(adjusted_start_seconds, time_data[time_data['Participant'] == subject]['Start'].apply(lambda x: parse_time(x).total_seconds()))

            # filtered_times = time_data[time_data['Participant'] == int(subject)]['Start'].apply(lambda x: parse_time(x).total_seconds())
            # print("Filtered times in seconds:", filtered_times)
            # print(subject)

            relevant_rows = time_data[mask]

            # Sort the DataFrame by 'task' and 'file' (assuming 'file' can be compared directly)
            relevant_rows_sorted = relevant_rows.sort_values(by=['task', 'file'])
            selected_row = relevant_rows_sorted.iloc[0]

            if not relevant_rows.empty:
                new_rows = pd.DataFrame({
                    'participant': [subject],
                    'task': selected_row['task'],
                    'file': str(selected_row['file']) + '.py',
                    'start': row['start'],
                    'end': row['end']
                })
                final_df = pd.concat([final_df, new_rows], ignore_index=True)

final_df.sort_values(by=['participant', 'task', 'file'])
final_df['start'] = final_df['start'].apply(convert_time_format)
final_df['end'] = final_df['end'].apply(convert_time_format)
# Save the final dataframe to a new CSV file
final_df.to_csv('trainingVidStamps.csv', index=False)
