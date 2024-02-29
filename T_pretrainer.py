from modules.convlstm import ConvLSTM
import torch
import os 
import pandas as pd
from PIL import Image
import tqdm
from torchvision import transforms

import wandb
use_wandb = True


import json

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader, val_loader, num_epochs):
        # train the model using tqdm
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0            

            loop2 = tqdm.tqdm(train_loader, total=len(train_loader), position=0, leave=True)
            loop2.set_description(f"Epoch [{epoch}/{num_epochs}]")

            self.model.reset_last_hidden_states(batch_size=train_loader.batch_size)
            i = 0   
            for inputs, labels, new_sequence in loop2:
                # if new_sequence.any():
                #     self.model.reset_last_hidden_states(batch_size=inputs.size(0))

                self.model.reset_last_hidden_states(batch_size=inputs.size(0))

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                # show the current loss in the progress bar
                loop2.set_postfix(loss=loss.item())

                # DLR adjusts the learning rate based on the current loss
                DLR_adjust(self.optimizer, loss.item())

                # log the loss to wandb
                if use_wandb:
                    wandb.log({"train_loss": loss.item()})
                    # log learning rate
                    wandb.log({"learning_rate": self.optimizer.param_groups[0]['lr']})

                # save every 1000 steps as well
                if i % 1000 == 0 and i > 0:
                    save_checkpoint(self.model, self.optimizer, filename=f"checkpoints/{epoch}_{i}.pth")
                
                i += 1

            train_loss /= len(train_loader)
            print("Epoch", epoch, "train_loss", train_loss)
            val_loss = self.evaluate(val_loader)
            print("Epoch", epoch, "val_loss", val_loss)

            # save the model
            save_checkpoint(self.model, self.optimizer, filename=f"checkpoints/{epoch}.pth")
            save_config(self.model, filename="checkpoints/config.json")

    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0.0

        self.model.reset_last_hidden_states(batch_size=val_loader.batch_size)

        with torch.no_grad():
            loop = tqdm.tqdm(val_loader, total=len(val_loader), position=0, leave=True)
            for inputs, labels, new_sequence in loop:

                if new_sequence.any():
                    self.model.reset_last_hidden_states(batch_size=inputs.size(0))

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)

                

                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                loop.set_postfix(loss=loss.item())

                if use_wandb:
                    # log the loss to wandb
                    wandb.log({"val_loss": loss.item()})
                    # log the formatted output to wandb
                    output = val_loader.dataset.format_output(outputs)
                    # log each output to wandb
                    for i in range(len(output)):
                        wandb.log(output[i])
            

        val_loss /= len(val_loader)
        return val_loss

class TDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, data_folders, transform=None, sequence_length=10):
        self.dataset_folder = dataset_folder
        self.data_folders = data_folders
        self.transform = transform
        self.sequence_length = sequence_length

        # Data folders are a list of folders, each folder contains images and a data.csv file for a given sequence
        # For easy handling, we will concatenate all the data.csv files into a single dataframe, and store the starting indexes of each folder
        self.labels = pd.DataFrame()
        self.start_indexes = [0]
        for folder in data_folders:
            data_file = os.path.join(self.dataset_folder, folder, "data.csv")
            data = pd.read_csv(data_file)
            self.labels = pd.concat([self.labels, data])
            self.start_indexes.append(len(self.labels))
        
        self.image_folders = [os.path.join(self.dataset_folder, folder, "images") for folder in data_folders]
        self.images = []
        for folder in self.image_folders:
            images = os.listdir(folder)
            for i in range(len(images)):
                self.images.append(os.path.join(folder, f"{i+1}.jpg"))

        self.labels_columns = self.labels.columns
    

    def __len__(self):
        return len(self.labels) - self.sequence_length

    def __getitem__(self, idx):
        images = []
        for i in range(self.sequence_length):
            img_name = self.images[idx + i]
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images)

        # for the labels, now the ConvLSTM model supports multiple labels, so we will include all the labels in the sequence
        labels = torch.tensor(self.labels.iloc[idx:idx+self.sequence_length].values, dtype=torch.float32)

        
        # if idx is one of the start indexes
        if idx in self.start_indexes:
            # Indicate that this is a new sequence
            new_sequence = True
        else:
            new_sequence = False

        return images, labels, new_sequence
    
    def format_output(self, output):
        output = output.squeeze(0)
        output = output.cpu().detach().numpy()
        formatted_output = []
        for i in range(len(output)):
            single_formatted_output = {}
            for j in range(len(output[i])):
                single_formatted_output[self.labels_columns[j]] = round(output[i][j], 2)
            formatted_output.append(single_formatted_output)
        return formatted_output

class TDataset_Fast(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, data_folders, transform=None, sequence_length=10):
        self.dataset_folder = dataset_folder
        self.data_folders = data_folders
        self.transform = transform
        self.sequence_length = sequence_length
        self.step_length = sequence_length // 3

        self.labels = pd.DataFrame()
        self.start_indexes = [0]
        for folder in data_folders:
            data_file = os.path.join(self.dataset_folder, folder, "data.csv")
            data = pd.read_csv(data_file)
            self.labels = pd.concat([self.labels, data])
            self.start_indexes.append(len(self.labels))
        
        self.image_folders = [os.path.join(self.dataset_folder, folder, "images") for folder in data_folders]
        self.images = []
        for folder in self.image_folders:
            images = os.listdir(folder)
            for i in range(len(images)):
                self.images.append(os.path.join(folder, f"{i+1}.jpg"))

        self.windows = []
        for i in range(0, len(self.images) - self.sequence_length, self.step_length):
            # we will also have to check if there's any frame between the current index and the index + sequence_length is a new sequence
            new_sequence = False
            for j in range(i, i + self.sequence_length):
                if j in self.start_indexes:
                    new_sequence = True
                    break

            labels = self.labels.iloc[i:i+self.sequence_length].values

            window = {
                "images": self.images[i:i+self.sequence_length],
                "labels": labels,
                "new_sequence": new_sequence
            }
            self.windows.append(window)

        self.labels_columns = self.labels.columns

        # cleanup some memory
        del self.images
        del self.labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        images = []
        for img_name in window["images"]:
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images)

        labels = torch.tensor(window["labels"], dtype=torch.float32)
        
        new_sequence = window["new_sequence"]

        return images, labels, new_sequence
    
    def format_output(self, output):
        output = output.squeeze(0)
        output = output.cpu().detach().numpy()
        formatted_output = []
        for i in range(len(output)):
            single_formatted_output = {}
            for j in range(len(output[i])):
                single_formatted_output[self.labels_columns[j]] = round(output[i][j], 2)
            formatted_output.append(single_formatted_output)
        return formatted_output


def save_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def save_config(model, filename="config.json"):
    config = {
        "input_dim": model.input_dim,
        "hidden_dims": model.hidden_dims,
        "kernel_size": model.kernel_size,
        "num_classes": model.num_classes,
        "num_layers": model.num_layers
    }
    with open(filename, "w") as f:
        json.dump(config, f)



def DLR_adjust(optimizer, loss):
    # dynamic learning rate parameters
    # DLR adjusts the learning rate based on the current loss
    # it maps the loss to a learning rate between learning_rate_lower and learning_rate_upper on a linear scale

    loss_treshold_lower = 0.01
    loss_treshold_upper = 0.1
    learning_rate_lower = 1e-4
    learning_rate_upper = 1e-2

    if loss < loss_treshold_lower:
        lr = learning_rate_lower
    elif loss > loss_treshold_upper:
        lr = learning_rate_upper
    else:
        lr = learning_rate_lower + (loss - loss_treshold_lower) * (learning_rate_upper - learning_rate_lower) / (loss_treshold_upper - loss_treshold_lower)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    del checkpoint
    return model, optimizer

def main():
    num_epochs = 20

    checkpoint_path = "checkpoints/1.pth"
    dataset_folder = "dataset_1"
    dataset_json = "dataset.json"

    wdb_name = "Pretraining-4"
    wdb_notes = "Using Batches, Shuffle, and DLR, also resetting hidden states for every step."

    # init the model
    input_dim = 3 # we have 3 channels for RGB
    kernel_size = 3 # kernel size
    num_classes = 13 # number of output classes
    num_layers = 4 # number of ConvLSTM layers
    
    hidden_dims = [8, 32, 64, 256] # hidden dimensions for each ConvLSTM layer
    
    height, width = 512, 288  # input image size
    sequence_length = 30

    batch_size = 4
    learning_rate = 1e-3


    model = ConvLSTM(input_dim, hidden_dims, kernel_size, num_classes, num_layers)

    # count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Number of parameters: {num_params}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_transforms = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # open dataset/dataset.json
    with open(os.path.join(dataset_folder, dataset_json), "r") as f:
        dataset = json.load(f)
    # init the trainset and valset
    train_folders = dataset["train"]
    val_folders = dataset["val"]
    trainset = TDataset(dataset_folder, train_folders, transform=image_transforms, sequence_length=sequence_length)
    valset = TDataset(dataset_folder, val_folders, transform=image_transforms, sequence_length=1)

    # init the dataloaders (these are sequential datasets)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4, drop_last=True, shuffle=True) # Shuffle is on !!
    val_loader = torch.utils.data.DataLoader(valset)

    # init the trainer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # load the checkpoint
    if os.path.exists(checkpoint_path):
        model, optimizer = load_checkpoint(model, optimizer, filename=checkpoint_path)
        print("Checkpoint loaded")

    # init wandb
    if use_wandb:
        wandb.init(project="Solaris", name=wdb_name, notes=wdb_notes)

    trainer = Trainer(model, criterion, optimizer, device)

    # train the model
    trainer.train(train_loader, val_loader, num_epochs)

if __name__ == "__main__":
    main()