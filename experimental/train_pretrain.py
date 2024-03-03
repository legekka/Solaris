from modules.timesformer import TimeSFormerClassifierT1
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
    def __init__(self, model, criterion, optimizer, lr_scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        

    def train(self, train_loader, val_loader, learning_rate, num_epochs):
        self.learning_rate = learning_rate
        step = 0
        # train the model using tqdm
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0            

            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")

            for i, (inputs, labels) in loop:
               
                # move the frames and labels to the device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                # update the learning rate
                self.lr_scheduler.step()

                # show the current loss in the progress bar
                loop.set_postfix(loss=loss.item())

                # log the loss to wandb
                if use_wandb:
                    wandb.log({"train_loss": loss.item(), "learning_rate": self.optimizer.param_groups[0]['lr']})

                # save every 1000 steps as well
                if i % 1000 == 0 and i > 0:
                    save_checkpoint(self.model, self.optimizer, filename=f"checkpoints/{epoch}_{i}.pth")


            train_loss /= len(train_loader)
            print("Epoch", epoch, "train_loss", train_loss)
            val_loss = self.evaluate(val_loader)
            print("Epoch", epoch, "val_loss", val_loss)

            # save the model
            save_checkpoint(self.model, self.optimizer, filename=f"checkpoints/{epoch+1}.pth")

    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            loop = tqdm.tqdm(val_loader, total=len(val_loader), position=0, leave=True)
            for inputs, labels in loop:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                loop.set_postfix(loss=loss.item())

                if use_wandb:
                    output = val_loader.dataset.format_output(outputs)
                    # log loss, and the output to wandb
                    wandb.log({"val_loss": loss.item(), **output})
            

        val_loss /= len(val_loader)
        return val_loss

class TDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, data_folders, transforms=None, sequence_length=10):
        self.dataset_folder = dataset_folder
        self.data_folders = data_folders
        self.sequence_length = sequence_length
        self.transforms = transforms

        
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
        frames = []
        for i in range(self.sequence_length):
            frame = Image.open(self.images[idx + i])
            if self.transforms:
                frame = self.transforms(frame)
            frames.append(frame)
        
        frames = torch.stack(frames)


        # for the labels, we will return the last label in the sequence
        labels = self.labels.iloc[idx + self.sequence_length - 1].values
        labels = torch.tensor(labels, dtype=torch.float32)


        return frames, labels
    
    def format_output(self, output):
        output = output.squeeze(0)
        output = output.cpu().detach().numpy()
        formatted_output = {}
        for i in range(len(output)):
            formatted_output[self.labels_columns[i]] = round(output[i], 2)
        return formatted_output

    
def save_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    del checkpoint
    return model, optimizer

def main():
    num_epochs = 10

    checkpoint_path = "models/TimeSFormer/Pretrain-3/2.pth"
    dataset_folder = "dataset_1"
    dataset_json = "dataset.json"

    wdb_name = "T-Pretraining-5"
    wdb_notes = "Last transformer block unfreezed, FC, unfreezed"

    batch_size = 8
    learning_rate = 6e-6

    sequence_length = 8

    model = TimeSFormerClassifierT1("facebook/timesformer-base-finetuned-k400", num_classes=13)

    # count the trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {num_params}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    t_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # open dataset/dataset.json
    with open(os.path.join(dataset_folder, dataset_json), "r") as f:
        dataset = json.load(f)
    # init the trainset and valset
    train_folders = dataset["train"]
    val_folders = dataset["val"]
    trainset = TDataset(dataset_folder, train_folders, t_transforms, sequence_length=sequence_length)
    valset = TDataset(dataset_folder, val_folders, t_transforms, sequence_length=sequence_length)

    # init the dataloaders (these are sequential datasets)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4, drop_last=True, shuffle=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(valset, persistent_workers=True, num_workers=4)


    

    # init the trainer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # calculating max steps based on the length of the train_loader
    max_steps = len(train_loader) * num_epochs 
    
    # init learning rate scheduler (CosineAnnealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-8)

    
    # load the checkpoint
    if os.path.exists(checkpoint_path):
        model, optimizer = load_checkpoint(model, optimizer, filename=checkpoint_path)
        print("Checkpoint loaded")
        # set learning rate back to the original value
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    
    # init wandb
    if use_wandb:
        wandb.init(project="Solaris", name=wdb_name, notes=wdb_notes)

    trainer = Trainer(model, criterion, optimizer, scheduler, device)

    # train the model
    trainer.train(train_loader, val_loader, learning_rate, num_epochs)

if __name__ == "__main__":
    main()