from modules.utils import get_elite_screenshot
from modules.gamepad import Gamepad


import torch
import json
from torchvision import transforms
import os
import time
import keyboard

def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_config(filename):
    with open(filename, "r") as f:
        config = json.load(f)
    return config

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

def print_output(output):

    # first we need to get the maximum length of the keys
    max_key_length = max([len(key) for key in output.keys()])
    # now we will print the output
    for key in output.keys():
        # print the key
        print(key + ":" + " " * (max_key_length - len(key)), end=" ")
        # convert the [-1, 1] value to 0-20
        value = int((output[key] + 1) * 10)
        # print the slider
        print("[" + "#" * value + "-" * (20 - value) + "] | " + str(output[key]))
    print("\n")

def checkForKeypress():
    if keyboard.is_pressed("ctrl+*"):
        print("Ctrl+* pressed...")
        return True
    
def main():

    model_folder = "models/TimeSFormer/Small/Final-2"
    checkpoint = "7.pth"


    height, width = 224, 224  # input image size

    image_transforms = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.CenterCrop((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from modules.timesformer import TimeSFormerClassifierU
    model = TimeSFormerClassifierU("facebook/timesformer-base-finetuned-k400", num_classes=13)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model = load_checkpoint(model, os.path.join(model_folder, checkpoint))

    print("Checkpoint loaded...")
    gamepad = Gamepad()
    print("Gamepad initialized...")

    fps = 1/10

    sequence_length = 8
    image_sequence = []

    model.eval()
    with torch.no_grad():
        while True:
            if checkForKeypress():
                break
            start = time.time()
            # get the latest screenshot
            screenshot = get_elite_screenshot()

            image_sequence.append(image_transforms(screenshot))
            if len(image_sequence) > sequence_length:
                image_sequence.pop(0)

            if len(image_sequence) < sequence_length:
                print("Loading sequence...")
                continue

            images = torch.stack(image_sequence)    
            
            # adding batch dimension
            screenshot = images.unsqueeze(0).to(device)
            
            # get the output
            output = model(screenshot)
            # clear the screen
            print("\033c", end="")
            print_output(format_output(output))
            # update the gamepad
            gamepad.update(format_output(output))

            if time.time() - start > fps:
                print("Slower than the set FPS!")

            # wait until fps amount of time has passed
            while time.time() - start < fps:
                time.sleep(0.001)
    print("Stopping...")
           
if __name__ == "__main__":
    main()