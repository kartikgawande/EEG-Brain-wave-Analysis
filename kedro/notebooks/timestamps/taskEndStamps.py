# importing vlc module
import vlc, keyboard, csv
import os, sys
from pathlib import Path

VIDEO_FOLDER = Path(r"D:\Kartik\IIITB\Study\Sem 3\PE\Kedro\kartik-eeg\data\01_raw\videos\16_to_30")
FILE = "30.mp4"
STAMP_FOLDER = Path(r'D:\Kartik\IIITB\Study\Sem 3\PE\Kedro\kartik-eeg\data\02_intermediate\video_data\16_to_30\tasksStamps')
CSV_FILE = os.path.join(STAMP_FOLDER,FILE[:-3]+'csv')

if not os.path.exists(CSV_FILE):
    # Create the CSV file and write the headers
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['task', 'file', 'end'])

def add_time_interval(stamp:list):
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(stamp)

# Function to seek forward
def seek_forward(seeksec):
    new_position = media_player.get_time() + seeksec*1000   # seconds 
    # new_position = current_position + seeksec  # Seek forward by 10 seconds
    media_player.set_time(int(new_position))  # seconds to milliseconds

def convert_milliseconds(milliseconds):
    seconds = milliseconds // 1000
    minutes = seconds // 60
    hours = minutes // 60
    
    seconds = seconds % 60
    minutes = minutes % 60
    
    return hours, minutes, seconds

def set_playback_speed(speed):
    media_player.set_rate(speed)

# Initialize the toast notifier
# toaster = ToastNotifier()

stamp = []
task=1
file=0
def on_key_press(event):
    global task
    global file
    global stamp
    key_pressed = event.name
    print(f"You pressed {key_pressed}")
    # toaster.show_toast("Key Pressed", f"You pressed: {key_pressed}", duration=1)
    # getting current media time
    if str(event.name)=='a':
        if file>1:
            file-=1
        elif task>1:
            file=4
            task-=1
        
        if file==1 and task==1:
            task=1
            file=0

        # if file>1:
        #     file-=1
        # elif task>1:
        #     file=4
        #     task-=1

        task_next=task
        if file==4 and task<3:
            task_next=task+1
        file_next=(file)%4+1
        print(f"Will write t{task_next}, f{file_next}")
        
    if str(event.name)=='d':
        if file==4 and task<3:
            task+=1
        file=(file)%4+1

        task_next=task
        if file==4 and task<3:
            task_next=task+1
        file_next=(file)%4+1
        print(f"Will write t{task_next}, f{file_next}")

    if str(event.name)=='up':
        value = media_player.get_time()
        # printing value
        print(f"Current Media time : {stamp}")
        print(convert_milliseconds(value))
        if file==4:
            task+=1
        file=(file)%4+1
        stamp.append(task)
        stamp.append(file)
        stamp.append(convert_milliseconds(value))
        add_time_interval(stamp)
        stamp.clear()
        if(task==3 and file==4):
            media_player.stop()
            keyboard.unhook_all()
            os._exit(0)

    if str(event.name)=='right':
        seek_forward(10)

    if str(event.name)=='left':
        seek_forward(-10)

# creating vlc media player object
media_player = vlc.MediaPlayer()
# media object
print("what: ",os.path.join(VIDEO_FOLDER,FILE))
media = vlc.Media(os.path.join(VIDEO_FOLDER,FILE))
 
try:
    media_player.set_media(media)
    media_player.play()
except Exception as e:
    print(f"Error playing video: {e}")

keyboard.on_press(on_key_press)
keyboard.wait()