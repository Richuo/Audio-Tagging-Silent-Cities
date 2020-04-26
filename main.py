import os
import sys
import time
import copy
import gc

import numpy as np
import random
import pandas as pd
import librosa
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as utils

sys.path.append('audio_tagging_functions')
from models import *
from data_processing import process_data


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {DEVICE}")


### Training function and class ###
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


def train_model(model, criterion, optimizer, dataloaders, scheduler=None, num_epochs=25):
    """
    Training function. 
    Returns the trained model and a dictionary for plotting purposes.
    """
    history_training = {'epochs': np.arange(num_epochs),
                        'train_loss': [],
                        'val_loss': [],
                        'train_acc': [],
                        'val_acc': []}

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        lasttime = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)['clipwise_output']
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))            

            history_training[f'{phase}_loss'].append(epoch_loss)
            history_training[f'{phase}_acc'].append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print("Epoch complete in {:.1f}s\n".format(time.time() - lasttime))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history_training


def plot_training(hist, graphs_path, model_type, do_save):
    hist = history_training
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('Cnn6 Finetunning - loss')
    ax[0].plot(hist["epochs"], hist["train_loss"], label="Train loss")
    ax[0].plot(hist["epochs"], hist["val_loss"], label="Validation loss")
    ax[1].set_title('Cnn6 Finetunning - accuracy')
    ax[1].plot(hist["epochs"], hist["train_acc"], label="Train accuracy")
    ax[1].plot(hist["epochs"], hist["val_acc"], label="Validation accuracy")
    ax[0].legend()
    ax[1].legend()
    if do_save:
        save_graph_path = f"{graphs_path}/{model_type}_training.jpg"
        plt.savefig(save_graph_path)
        print(f"Training graph saved at {save_graph_path}")



if __name__ == "__main__":

    ### Main parameters ###
    # Path
    MODEL_PATH = "pretrained_models/Cnn6_mAP=0.343.pth"
    LABELS_PATH = "labels/all_data.csv"

    TRAINED_MODELS_PATH = "models"
    GRAPHS_PATH = "graphs"
    SAVING = True

    # Audio parameters
    SR = 14000             # Sample Rate
    AUDIO_DURATION = 10    # 10 seconds duration window for all audios

    # Model parameters
    MODEL_TYPE = "Transfer_Cnn6"
    NB_SPECIES = 13        # Number of classes
    LR = 1e-2              # Learning Rate
    BATCH_SIZE = 16
    EPOCHS = 20

    # Misc parameters
    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)


    ### Data processing ###
    labels_df = pd.read_csv(LABELS_PATH)
    (trainloader, validationloader, testloader) = process_data(df=labels_df, batch_size=BATCH_SIZE,
                                                               sample_rate=SR, audio_duration=AUDIO_DURATION, 
                                                               random_state=RANDOM_STATE)
    dataloaders = {"train": trainloader,
                   "val": validationloader}
    dataset_sizes = {"train": len(trainloader)*BATCH_SIZE,
                     "val": len(validationloader)*BATCH_SIZE}


    ### Load Model ###
    model = load_model(model_type=MODEL_TYPE, sample_rate=SR, nb_species=NB_SPECIES, model_path=MODEL_PATH)


    ###  Define loss function and optimizer ### 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    ### Training ###
    model, history_training = train_model(model=model, criterion=criterion, optimizer=optimizer, 
                                          dataloaders=dataloaders, scheduler=None, num_epochs=EPOCHS)


    ### Save the model ###
    if SAVING:
        save_model_path = f"{TRAINED_MODELS_PATH}/{MODEL_TYPE}_trained.pth"
        torch.save(model.module.state_dict(), save_model_path)
        print(f"Model saved at {save_model_path}")


    ### Plotting the losses ###
    plot_training(hist=history_training, graphs_path=GRAPHS_PATH, model_type=MODEL_TYPE, do_save=SAVING)
