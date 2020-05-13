import os
import sys

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('audio_tagging_functions')
from models import *


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


### Load model function and Models class ###

def load_model(model_type, sample_rate, nb_species, model_path):
    """
    Loads and returns the choosen model.
    """
    # Initialize Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
        classes_num=nb_species, freeze_base=True)

    print(model)

    # Load pretrained model
    model.load_from_pretrain(model_path)

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if DEVICE == 'cuda':
        model.to(DEVICE)

    print('Load pretrained model successfully!')

    return model


class Transfer_Cnn6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """
        Classifier for bird species using pretrained Cnn6 as a sub module.
        """
        super(Transfer_Cnn6, self).__init__()
        audioset_classes_num = 527

        self.base = Cnn6(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer1 = nn.Linear(512, 256, bias=True)
        self.fc_transfer2 = nn.Linear(256, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer1)
        init_layer(self.fc_transfer2)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=torch.device(DEVICE))
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        x = F.relu(self.fc_transfer1(embedding))
        clipwise_output = self.fc_transfer2(x)
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict


class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527

        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer1 = nn.Linear(2048, 256, bias=True)
        self.fc_transfer2 = nn.Linear(256, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer1)
        init_layer(self.fc_transfer2)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=torch.device(DEVICE))
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        x = F.relu(self.fc_transfer1(embedding))
        clipwise_output =  self.fc_transfer2(x)
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict


class Transfer_ResNet22(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_ResNet22, self).__init__()
        audioset_classes_num = 527

        self.base = ResNet22(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer1 = nn.Linear(2048, 256, bias=True)
        self.fc_transfer2 = nn.Linear(256, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer1)
        init_layer(self.fc_transfer2)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=torch.device(DEVICE))
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        with torch.no_grad():
            output_dict = self.base(input, mixup_lambda)
            embedding = output_dict['embedding']

        x = F.relu(self.fc_transfer1(embedding))
        clipwise_output =  self.fc_transfer2(x)
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict



if __name__ == "__main__":

    ### Main parameters ###
    # Path
    MODEL_PATH = "pretrained_models/Cnn6_mAP=0.343.pth"

    # Audio parameters
    SR = 14000             # Sample Rate

    # Model parameters
    MODEL_TYPE = "Transfer_Cnn6"
    NB_SPECIES = 13        # Number of classes

    ### Load Model ###
    model = load_model(model_type=MODEL_TYPE, sample_rate=SR, nb_species=NB_SPECIES, model_path=MODEL_PATH)

