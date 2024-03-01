import numpy as np
import PIL.Image as Image
import glob
import torch

from transformers import AutoImageProcessor, TimesformerModel
from torchvision import transforms

from modules.timesformer import TimeSFormerClassifierHR, TimeSFormerClassifierT2E

np.random.seed(0)

config = {
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "num_frames": 32,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0,
    "attention_probs_dropout_prob": 0,
    "initializer_range": 0.02,
    "layer_norm_eps": 0.000001,
    "qkv_bias": True,
    "attention_type": "divided_space_time",
    "drop_path_rate": 0,
}

imagetransforms = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.CenterCrop(config["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image_frames(image_paths):
    frames = []
    for image_path in image_paths:
        frame = Image.open(image_path)
        frame = imagetransforms(frame)
        frames.append(frame)
    
    # Stack all the frames to form a single tensor
    frames = torch.stack(frames)

    return frames

def sample_frame_indices(clip_len, total_frames):
    '''
    Sample frame indices for a given number of frames from the total frames available.
    Args:
        clip_len (`int`): Total number of frames to sample.
        total_frames (`int`): Total available frames in the sequence.
    Returns:
        List[int]: List of sampled frame indices
    '''
    indices = np.linspace(0, total_frames - 1, num=clip_len)
    indices = np.clip(indices, 0, total_frames - 1).astype(np.int64)
    return indices



# Assuming you have a folder 'image_sequence/' with your frames named sequentially (e.g., frame_001.jpg, frame_002.jpg, ...)
image_folder = 'datasets/navigation_1/NavigateGeneral_1/images/'
image_paths = sorted(glob.glob(f'{image_folder}*.jpg'))  # Make sure this matches your file naming pattern

total_frames = len(image_paths)  # Total number of frames available
indices = sample_frame_indices(clip_len=config["num_frames"], total_frames=total_frames)  # Sample 8 frames
sampled_image_paths = [image_paths[i] for i in indices]
frames = load_image_frames(sampled_image_paths)



model = TimeSFormerClassifierHR(config, num_classes=13)
print(model)
print("---")
# number of parameters (all, not just trainable)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")


# model size in memory (MB)
model_size = round(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2), 2)
print(f"Model size: {model_size} MB")

model.to("cuda")
inputs = frames.to("cuda")
inputs = inputs.unsqueeze(0)
print(inputs.shape)
model.eval()
# # forward pass

import tqdm
loop = tqdm.tqdm(range(100))


# for i in loop:
#     print("a")
#     outputs = model(inputs)
#     loop.refresh()


with torch.no_grad():
    for i in loop:
        outputs = model(inputs)
        loop.refresh()


# outputs = model(inputs)
# print(outputs.shape)
