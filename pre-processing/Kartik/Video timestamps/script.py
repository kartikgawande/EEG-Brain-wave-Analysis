# importing vlc module
import vlc, keyboard, csv
from win10toast import ToastNotifier
import os

VIDEO_FOLDER = "D:\\Kartik\\IIITB\\Study\\Sem 3\\PE\\VIdeoTimestampsScript\\videos\\Kartik\\"
FILE = "17ank-2019-06-05_14.43.17.mkv"
STAMP_FOLDER = '.\\stamps\\'
CSV_FILE = STAMP_FOLDER+FILE[:-3]+'csv'

if not os.path.exists(CSV_FILE):
    # Create the CSV file and write the headers
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['start', 'end'])

def add_time_interval(start, end):
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([start, end])

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

prev_stamp = 0
def on_key_press(event):
    global prev_stamp
    key_pressed = event.name
    print(f"You pressed {key_pressed}")
    # toaster.show_toast("Key Pressed", f"You pressed: {key_pressed}", duration=1)
    # getting current media time
    if str(event.name)=='down':
        value = media_player.get_time()
        # printing value
        print("Current Media time : ")
        print(convert_milliseconds(value))
        prev_stamp=value

    if str(event.name)=='up':
        value = media_player.get_time()
        # printing value
        print("Current Media time : ")
        print(convert_milliseconds(value))
        add_time_interval(convert_milliseconds(prev_stamp), convert_milliseconds(value))

    if str(event.name)=='right':
        seek_forward(5)

    if str(event.name)=='left':
        seek_forward(-5)

# creating vlc media player object
media_player = vlc.MediaPlayer()
# media object
media = vlc.Media(VIDEO_FOLDER+FILE)
 
# setting media to the media player
media_player.set_media(media)
 
# start playing video
media_player.play()

keyboard.on_press(on_key_press)
keyboard.wait()