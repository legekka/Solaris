from modules.joysticks import Throttle, Stick
from modules.utils import get_elite_screenshot
from modules.visualize import GUI

import torch
import json
from torchvision import transforms
import os
import time
import keyboard

hotas_throttle = Throttle()
hotas_stick = Stick()


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_config(filename):
    with open(filename, "r") as f:
        config = json.load(f)
    return config

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

def format_output(output):
    # output is a tensor, we need to format it to a string
    # the output tensor has shape (1, num_classes)
    # we need to return each key and value in a dictionary
    # keys: throttle,frequency,frameshift_jump,return_from_surface_scanner,roll,pitch,fss_yaw,fss_pitch,yaw,primary_fire,secondary_fire,fss_zoom_out,fss_mode
    output = output.squeeze(0)
    output = output.cpu().detach().numpy()
    output = {
        "primary_fire": round(output[9], 2),
        "secondary_fire": round(output[10], 2),
        "frameshift_jump": round(output[2], 2),
        "throttle": round(output[0], 2),
        "frequency": round(output[1], 2),
        "pitch": round(output[5], 2),
        "yaw": round(output[8], 2),
        "roll": round(output[4], 2),
        "fss_pitch": round(output[7], 2),
        "fss_yaw": round(output[6], 2),
        "fss_zoom_out": round(output[11], 2),
        "fss_mode": round(output[12], 2),    
        "return_from_surface_scanner": round(output[3], 2),
    }
    
    return output

def print_output(output, joystick_values):
    # this function prints the output in a nice format:
    # each key has a slider, and the value is the value of the slider [-1, 1]
    # the slider is 22 characters long (including the brackets)
    # example slider for value 0: [##########----------]
    # also the joystick values are printed next to the sliders also via sliders
    # here's an example of the full output:
    
    # throttle:                    [##########----------] | [##########----------]
    # frequency:                   [##########----------] | [##########----------]
    # frameshift_jump:             [##########----------] | [##########----------]
    # return_from_surface_scanner: [##########----------] | [##########----------]
    # ...

    # we will make sure that the sliders starts at the same position, so we will add spaces to the left of the sliders

    # first we need to get the maximum length of the keys
    max_key_length = max([len(key) for key in output.keys()])
    # now we will print the output
    for key in output.keys():
        # print the key
        print(key + ":" + " " * (max_key_length - len(key)), end=" ")
        # convert the [-1, 1] value to 0-20
        value = int((output[key] + 1) * 10)
        # print the slider
        print("[" + "#" * value + "-" * (20 - value) + "]", end=" | ")
        # print the joystick values
        value = int((joystick_values[key] + 1) * 10)
        print("[" + "#" * value + "-" * (20 - value) + "]")
    print("\n")

def checkForKeypress():
    if keyboard.is_pressed("ctrl+*"):
        print("Ctrl+* pressed...")
        return True
    

# this is a class running on a separate thread that collects screenshots with a given frequency
class ScreenshotCollector:
    def __init__(self, sampling_rate, seq_len):
        self.sampling_rate = sampling_rate
        self.seq_len = seq_len
        self.screenshots = []
        self.running = False
        self.stack = None

    def start(self):
        global image_transforms
        self.running = True
        while self.running:
            start = time.time()
            screenshot = get_elite_screenshot()
            self.screenshots.append(image_transforms(screenshot))
            if len(self.screenshots) > self.seq_len:
                self.screenshots.pop(0)
            if keyboard.is_pressed("ctrl+*"):
                print("Stopping the screenshot collector...")
                self.running = False
            # stack
            self.stack = torch.stack(self.screenshots)
            # wait until the next screenshot
            while time.time() - start < self.sampling_rate:
                time.sleep(0.001)

    def stop(self):
        self.running = False

    def get_screenshots(self):
        return self.screenshots
    
    def get_stack(self):
        return self.stack
    
    def clear(self):
        self.screenshots = []

def load_config(filename):
    with open(filename, "r") as f:
        config = json.load(f)
    return config

def main():

    gui = None
    #gui = GUI()
    config_file = "models/TimeSFormer/EDENFSS/Pretrain-2/training_config.json"
    model_folder = "models/TimeSFormer/EDENFSS/Pretrain-2"
    checkpoint = "10.pth"

    config = load_config(config_file)

    from modules.timesformer import EDEN

    global image_transforms
    image_transforms = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = EDEN(config, num_classes=config["num_classes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model = load_checkpoint(model, os.path.join(model_folder, checkpoint))

    print(model)

    screenshot_fps = 1/10

    # start the screenshot collector on a separate thread
    import threading
    collector = ScreenshotCollector(screenshot_fps, config["num_frames"])
    collector_thread = threading.Thread(target=collector.start)
    collector_thread.start()

    print("Screenshot collector started, waiting for screenshots...")
    # wait until num_frames are collected
    while len(collector.get_screenshots()) < config["num_frames"]:
        time.sleep(0.1)
    print("Done, starting the inference...")
    
    model_fps = 1/10

    model.eval()
    with torch.no_grad():
        while True:
            if checkForKeypress():
                break
            start = time.time()

            # get the latest screenshots
            images = collector.get_stack()
            
            # adding batch dimension
            screenshot = images.unsqueeze(0).to(device)
            
            # get the output
            output = model(screenshot)
            joystick_values = get_values_dict()
            # clear the screen
            print("\033c", end="")
            print_output(format_output(output), joystick_values)
            
            if gui is not None:
                gui.update_and_draw(format_output(output))

            if time.time() - start > model_fps:
                delay = time.time() - start
                print("Slower than the set FPS! Current FPS: ", round(1/delay, 1))

            # wait until fps amount of time has passed
            while time.time() - start < model_fps:
                time.sleep(0.001)
    print("Stopping...")
    collector.stop()
    collector_thread.join()
    if gui is not None:
        gui.close()
           
if __name__ == "__main__":
    main()