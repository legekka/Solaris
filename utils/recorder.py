from modules.joysticks import Throttle, Stick
from modules.utils import *
import time
import mss
import os
import keyboard

hotas_throttle = Throttle()
hotas_stick = Stick()

def get_values_dict():
    hotas_throttle.update()
    hotas_stick.update()
    return {
        "throttle": hotas_throttle.throttle,
        "frequency": hotas_throttle.frequency,
        "frameshift_jump": hotas_throttle.frameshift_jump,
        "return_from_surface_scanner": hotas_throttle.return_from_surface_scanner,
        "roll": hotas_stick.roll,
        "pitch": hotas_stick.pitch,
        "fss_yaw": hotas_stick.fss_yaw,
        "fss_pitch": hotas_stick.fss_pitch,
        "yaw": hotas_stick.yaw,
        "primary_fire": hotas_stick.primary_fire,
        "secondary_fire": hotas_stick.secondary_fire,
        "fss_zoom_out": hotas_stick.fss_zoom_out,
        "fss_mode": hotas_stick.fss_mode
    }

def checkForKeypress():
    if keyboard.is_pressed("ctrl+*"):
        print("Ctrl+* pressed...")
        return True

def main():
    dataset_folder = "datasets/fss_1"
    name = "FSSGeneral"
    fps = 1/10
    
    # for convention we are using name_number as folder name
    number = 1
    # check if folder exists, if it does, increment number
    while os.path.exists(dataset_folder + "/" + name + "_" + str(number)):
        number += 1
    folder = dataset_folder + "/" + name + "_" + str(number)
    os.makedirs(folder)
    image_folder = folder + "/images"
    os.makedirs(image_folder)
    
    # create data.csv
    with open(folder + "/data.csv", "w") as f:
        f.write("throttle,frequency,frameshift_jump,return_from_surface_scanner,roll,pitch,fss_yaw,fss_pitch,yaw,primary_fire,secondary_fire,fss_zoom_out,fss_mode\n")

    print("Recording...")
    i = 1
    images = []
    while True:
        if checkForKeypress():
            break
        values = get_values_dict()
        # append values to the folder/data.csv
        start = time.time()
        screenshot = get_elite_screenshot()
        images.append(screenshot)

        with open(folder + "/data.csv", "a") as f:
            f.write(",".join([str(values[key]) for key in values.keys()]) + "\n")

        # wait until fps time elapses
        while time.time() - start < fps:
            time.sleep(0.001)
        i += 1

    print("Saving images...")
    # save images as jpg, 80% quality
    for i, image in enumerate(images):
        image.save(image_folder + "/" + str(i + 1) + ".jpg", "JPEG", quality=80)
    print("Saved " + str(len(images)) + " images .")

if __name__ == "__main__":
    print("Starting in 2...")
    time.sleep(1)
    print("Starting in 1...")
    time.sleep(1)
    main()