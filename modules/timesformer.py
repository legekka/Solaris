import torch.nn as nn
from transformers import TimesformerModel, TimesformerConfig

class TimeSFormerClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(TimeSFormerClassifier, self).__init__()
        # Load the pretrained TimeSFormer model
        self.timesformer = TimesformerModel.from_pretrained(pretrained_model_name)
        # freeze the parameters
        for param in self.timesformer.parameters():
            param.requires_grad = False
        hidden_size = self.timesformer.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//8),
            nn.BatchNorm1d(hidden_size//8),
            nn.ReLU(),
            nn.Linear(hidden_size//8, hidden_size//16),
            nn.BatchNorm1d(hidden_size//16),
            nn.ReLU(),
            nn.Linear(hidden_size//16, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.timesformer(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.fc(pooled_output)
        return logits
    

class TimeSFormerClassifierU(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(TimeSFormerClassifierU, self).__init__()
        # Load the pretrained TimeSFormer model
        self.timesformer = TimesformerModel.from_pretrained(pretrained_model_name)
        hidden_size = self.timesformer.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//8),
            nn.BatchNorm1d(hidden_size//8),
            nn.ReLU(),
            nn.Linear(hidden_size//8, hidden_size//16),
            nn.BatchNorm1d(hidden_size//16),
            nn.ReLU(),
            nn.Linear(hidden_size//16, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.timesformer(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.fc(pooled_output)
        return logits
    


class TimeSFormerClassifierT1(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(TimeSFormerClassifierT1, self).__init__()
        # Load the pretrained TimeSFormer model
        self.timesformer = TimesformerModel.from_pretrained(pretrained_model_name)

        # Initially freeze all the parameters
        for param in self.timesformer.parameters():
            param.requires_grad = False

        # Unfreeze the last layer of the encoder
        # Note: The exact path to the last layer might need adjustment based on the model's definition
        for param in self.timesformer.encoder.layer[-1].parameters():
            param.requires_grad = True

        hidden_size = self.timesformer.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//8),
            nn.BatchNorm1d(hidden_size//8),
            nn.ReLU(),
            nn.Linear(hidden_size//8, hidden_size//16),
            nn.BatchNorm1d(hidden_size//16),
            nn.ReLU(),
            nn.Linear(hidden_size//16, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.timesformer(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.fc(pooled_output)
        return logits
    

class TimeSFormerClassifierT2(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(TimeSFormerClassifierT2, self).__init__()
        # Load the pretrained TimeSFormer model
        self.timesformer = TimesformerModel.from_pretrained(pretrained_model_name)

        # Initially freeze all the parameters
        for param in self.timesformer.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 layer of the encoder
        for param in self.timesformer.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.timesformer.encoder.layer[-2].parameters():
            param.requires_grad = True

        hidden_size = self.timesformer.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//8),
            nn.BatchNorm1d(hidden_size//8),
            nn.ReLU(),
            nn.Linear(hidden_size//8, hidden_size//16),
            nn.BatchNorm1d(hidden_size//16),
            nn.ReLU(),
            nn.Linear(hidden_size//16, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.timesformer(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.fc(pooled_output)
        return logits
    
    
class TimeSFormerClassifierT2E(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(TimeSFormerClassifierT2E, self).__init__()
        # Load the pretrained TimeSFormer model
        self.timesformer = TimesformerModel.from_pretrained(pretrained_model_name)

        # Initially freeze all the parameters
        for param in self.timesformer.parameters():
            param.requires_grad = False

        # Unfreeze embedding layers
        for param in self.timesformer.embeddings.parameters():
            param.requires_grad = True

        # Unfreeze the last 2 layer of the encoder
        for param in self.timesformer.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.timesformer.encoder.layer[-2].parameters():
            param.requires_grad = True

        hidden_size = self.timesformer.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//8),
            nn.BatchNorm1d(hidden_size//8),
            nn.ReLU(),
            nn.Linear(hidden_size//8, hidden_size//16),
            nn.BatchNorm1d(hidden_size//16),
            nn.ReLU(),
            nn.Linear(hidden_size//16, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.timesformer(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.fc(pooled_output)
        return logits
    

class TimeSFormerClassifierHR(nn.Module):
    def __init__(self, config, num_classes=13):
        super(TimeSFormerClassifierHR, self).__init__()
        # Load the pretrained TimeSFormer model
        config = TimesformerConfig(**config)
        
        self.timesformer = TimesformerModel(config=config)
        # freeze the parameters
        # for param in self.timesformer.parameters():
        #     param.requires_grad = False
        # unfreeze everything
        for param in self.timesformer.parameters():
            param.requires_grad = True

        hidden_size = self.timesformer.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//8),
            nn.BatchNorm1d(hidden_size//8),
            nn.ReLU(),
            nn.Linear(hidden_size//8, hidden_size//16),
            nn.BatchNorm1d(hidden_size//16),
            nn.ReLU(),
            nn.Linear(hidden_size//16, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.timesformer(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.fc(pooled_output)
        return logits
    