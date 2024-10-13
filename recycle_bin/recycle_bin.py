import subprocess
import time
import os
from multiprocessing import Process


def capture_photo_android():
    android_photo_dir = "/sdcard/DCIM/Camera/"
    
    print("Opening the camera app...")
    open_camera_command = 'adb shell am start -a android.media.action.STILL_IMAGE_CAMERA'
    subprocess.run(open_camera_command, shell=True)
    
    time.sleep(3) 
    
    print("Taking a picture...")
    take_photo_command = 'adb shell input keyevent 24'  
    subprocess.run(take_photo_command, shell=True)
    
    time.sleep(3) 

    result = subprocess.run(["adb", "shell", "ls", "-t", android_photo_dir], stdout=subprocess.PIPE, text=True)
    files = result.stdout.splitlines()
    print(files)

    if files:
        latest_photo = files[1]
        android_photo_path = android_photo_dir + latest_photo
        
        local_photo_dir = "./folders"
        if not os.path.exists(local_photo_dir):
            os.makedirs(local_photo_dir)

        local_photo_path = os.path.join(local_photo_dir, latest_photo)
        
        subprocess.run(["adb", "pull", android_photo_path, local_photo_path])
        
        if os.path.exists(local_photo_path):
            print(f"Photo saved at: {local_photo_path}")
        else:
            print("Failed to retrieve the photo.")
    else:
        print("No photo found in the Camera directory.")


def run_flask_app():
    subprocess.run(["python", "app.py"])


if __name__ == "__main__":
    capture_photo_android()

    flask_process = Process(target=run_flask_app)
    flask_process.start()

    flask_process.join()