import numpy as np
import PIL.Image as Image
import glob
import torch

from transformers import AutoImageProcessor, TimesformerModel
from torchvision import transforms

from modules.timesformer import EDEN

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

def load_checkpoint(model, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint
    return model


def translate_key(checkpoint_key):
    # Adjust the function to correctly translate keys based on the provided corrections.
    new_key = checkpoint_key.replace('model.', '')

    new_key = new_key.replace('norm.weight', 'layernorm.weight')
    new_key = new_key.replace('norm.bias', 'layernorm.bias')

    new_key = new_key.replace('cls_token', 'embeddings.cls_token')
    new_key = new_key.replace('pos_embed', 'embeddings.position_embeddings')
    new_key = new_key.replace('time_embed', 'embeddings.time_embeddings')
    new_key = new_key.replace('patch_embed.proj', 'embeddings.patch_embeddings.projection')
    new_key = new_key.replace('blocks', 'encoder.layer')

    new_key = new_key.replace('.norm1.', '.layernorm_before.')
    new_key = new_key.replace('.norm2.', '.layernorm_after.')
    new_key = new_key.replace('.attn.qkv.', '.attention.attention.qkv.')
    new_key = new_key.replace('.attn.proj.', '.attention.output.dense.')
    new_key = new_key.replace('.temporal_norm1.', '.temporal_layernorm.')
    new_key = new_key.replace('.temporal_fc.', '.temporal_dense.')
    new_key = new_key.replace('.temporal_attn.qkv.', '.temporal_attention.attention.qkv.')
    new_key = new_key.replace('.temporal_attn.proj.', '.temporal_attention.output.dense.')

    new_key = new_key.replace('.mlp.fc1.', '.intermediate.dense.')
    new_key = new_key.replace('.mlp.fc2.', '.output.dense.')
    # Add more translation rules as necessary
    return new_key

# Assuming you have a folder 'image_sequence/' with your frames named sequentially (e.g., frame_001.jpg, frame_002.jpg, ...)
image_folder = 'datasets/navigation_1/NavigateGeneral_1/images/'
image_paths = sorted(glob.glob(f'{image_folder}*.jpg'))  # Make sure this matches your file naming pattern

total_frames = len(image_paths)  # Total number of frames available
indices = sample_frame_indices(clip_len=config["num_frames"], total_frames=total_frames)  # Sample 8 frames
sampled_image_paths = [image_paths[i] for i in indices]
frames = load_image_frames(sampled_image_paths)



model = EDEN(config, num_classes=13)
print(model)
print("---")

# checkpoint
checkpoint_path = "models/checkpoint.pyth"
checkpoint = torch.load(checkpoint_path)

checkpoint = checkpoint["model_state"]

# Prepare the new state dict
new_state_dict = {}

for key in checkpoint.keys():
    new_key = translate_key(key)
    if new_key in model.timesformer.state_dict().keys():
        print(f"Mapping {key} to {new_key}")
        new_state_dict[new_key] = checkpoint[key]
    else:
        print(f"Skipping {key} as {new_key} not in model")

# Load the new state dict
model.timesformer.load_state_dict(new_state_dict)


# save the model state dict
model_state_dict = model.state_dict()
checkpoint = {
    "state_dict": model_state_dict
}
torch.save(checkpoint, "models/EDEN_base.pth")
exit()

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
