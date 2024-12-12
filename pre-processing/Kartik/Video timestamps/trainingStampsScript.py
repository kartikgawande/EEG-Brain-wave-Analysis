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
    time_parts = time_str.strip('()').split(',')
    # Format the parts into hh:mm:ss
    return f"{int(time_parts[0]):02}:{int(time_parts[1]):02}:{int(time_parts[2]):02}"

# Load the offset times
offsets = pd.read_csv(r'D:\\Kartik\\IIITB\\Study\\Sem 3\\PE\\VIdeoTimestampsScript\\Synced_Times_All.csv')
offsets['Offset Time'] = offsets['Offset Time'].apply(parse_time)

# Load the time sheet data
time_data = pd.read_csv(r'D:\\Kartik\\IIITB\\Study\\Sem 3\\PE\\VIdeoTimestampsScript\\time data - Time sheet sorted.csv')

# Prepare the directory path to the stamps files
directory_path = r'D:\\Kartik\\IIITB\\Study\\Sem 3\\PE\\VIdeoTimestampsScript\\stamps'

# Initialize the dataframe for the final output
final_df = pd.DataFrame(columns=['Subject', 'Task', 'File', 'Start', 'End'])

# Process each file in the stamps directory
# print(os.listdir(directory_path))
for filename in os.listdir(directory_path):
    if filename.endswith('.csv') and filename!='Synced_Times_All.csv':
        # Read the current stamps file
        current_file_path = os.path.join(directory_path, filename)
        current_data = pd.read_csv(current_file_path)

        # print(current_data)

        # Extract subject from filename
        subject = filename[:2]
        # print(subject)

        # Find the corresponding offset for the subject
        # print(offsets['File Name'].values)
        if filename in offsets['File Name'].values:
            offset = offsets[offsets['File Name'] == filename]['Offset Time'].iloc[0]
            # print(offsets[offsets['File Name'] == filename]['Offset Time'].iloc[0])
        else:
            continue  # Skip if no offset found

        # Adjust start times in current_data
        # current_data['adjusted_start'] = current_data['start'] + offset
        current_data['adjusted_start'] = current_data['start'].apply(lambda x: parse_time(x) + offset)

        # print(current_data['adjusted_start'])
        
        # Determine Task and File from time_data based on adjusted start
        for index, row in current_data.iterrows():
            adjusted_start_seconds = row['adjusted_start'].total_seconds()


            mask = (time_data['Participant'] == int(subject)) & \
                   (time_data['Start'].apply(lambda x: parse_time(x).total_seconds()) <= adjusted_start_seconds) & \
                   (time_data['End'].apply(lambda x: parse_time(x).total_seconds()) >= adjusted_start_seconds)
            
            # print(adjusted_start_seconds, time_data[time_data['Participant'] == subject]['Start'].apply(lambda x: parse_time(x).total_seconds()))

            # filtered_times = time_data[time_data['Participant'] == int(subject)]['Start'].apply(lambda x: parse_time(x).total_seconds())
            # print("Filtered times in seconds:", filtered_times)
            # print(subject)

            relevant_rows = time_data[mask]
            if not relevant_rows.empty:
                new_rows = pd.DataFrame({
                    'Subject': subject,
                    'Task': relevant_rows['Type'],
                    'File': str(relevant_rows['File'].values[0])+'.py',
                    'Start': row['start'],
                    'End': row['end']
                })
                final_df = pd.concat([final_df, new_rows], ignore_index=True)

final_df.sort_values(by=['Subject', 'Task', 'File'])
final_df['Start'] = final_df['Start'].apply(convert_time_format)
final_df['End'] = final_df['End'].apply(convert_time_format)
# Save the final dataframe to a new CSV file
final_df.to_csv('trainingStamps.csv', index=False)
