import pandas as pd
import os
from datetime import datetime, timedelta

def parse_time(time_str):
    """Parse time in hh:mm:ss format to a datetime object."""
    return datetime.strptime(time_str, "%H:%M:%S")

def format_time_delta(delta):
    """Format time delta to (hh, mm, ss) string format."""
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"({hours:02d}, {minutes:02d}, {seconds:02d})"

# Load the real times data
real_times = pd.read_csv(r'D:\\Kartik\\IIITB\\Study\\Sem 3\\PE\\VIdeoTimestampsScript\\time data - Time sheet sorted.csv')

# Directory containing all participant files (video files)
directory = r'D:\\Kartik\\IIITB\\Study\\Sem 3\\PE\\VIdeoTimestampsScript\\stamps\\'

# Placeholder for the result
results = []

# List all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Load the video times file
        video_times = pd.read_csv(os.path.join(directory, filename))
        
        # Extract participant number from filename
        participant_number = int(filename[:2])
        
        # Get the last end time from video file (start activity time in video)
        video_start_time_tuple = eval(video_times['end'].iloc[-1])
        video_start_time_str = f"({video_start_time_tuple[0]:02d}, {video_start_time_tuple[1]:02d}, {video_start_time_tuple[2]:02d})"
        video_start_time_datetime = datetime.strptime(f"{video_start_time_tuple[0]:02d}:{video_start_time_tuple[1]:02d}:{video_start_time_tuple[2]:02d}", "%H:%M:%S")
        
        # Find corresponding real start time
        real_start_time = real_times[(real_times['Participant'] == participant_number) & (real_times['Type'] == 1) & (real_times['File'] == 1)]['Start'].iloc[0]
        real_start_time_datetime = parse_time(real_start_time)

        # Calculate time offset
        offset_delta = real_start_time_datetime - video_start_time_datetime
        offset_time_str = format_time_delta(offset_delta)

        # Append data to results
        results.append({'File Name': filename, 'Video Time': video_start_time_str, 'Actual Time': real_start_time, 'Offset Time': offset_time_str})

# Convert results to DataFrame
synced_times = pd.DataFrame(results)

# Save the DataFrame to a CSV file
synced_times.to_csv('Synced_Times_All.csv', index=False)
