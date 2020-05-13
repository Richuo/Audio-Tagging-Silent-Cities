import os
import sys
import time
import copy
import argparse

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

sys.path.append('audio_tagging_functions')
from models import *
from transflearn_models import *
from create_birds_dataset import FINAL_LABELS_PATH
from data_processing import process_data
from mp3towav import SAMPLE_RATE


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {DEVICE}")


### Training function ###

def train_model(model, criterion, optimizer, dataloaders, scheduler=None, num_epochs=25):
    """
    Training function. 
    Returns the trained model and a dictionary for plotting purposes.
    """
    history_training = {'epochs': np.arange(num_epochs),
                        'train_loss': [],
                        'val_loss': [],
                        'train_acc': [],
                        'val_acc': [], 
                        'best_val_acc': 0}

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

            print(running_corrects.double(), dataset_sizes[phase])

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
    best_acc = round(float(best_acc), 4)
    print('Best val Acc: {:4f}'.format(best_acc))
    history_training['best_val_acc'] = best_acc

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history_training


def test_model(model, criterion, dataloaders):
    """
    Testing function. 
    Print the loss and accuracy after the inference on the testset.
    """
    sincetime = time.time()

    phase = "test"
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)['clipwise_output']
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / dataset_sizes[phase]
    test_acc = running_corrects.double() / dataset_sizes[phase]

    print('\n**TESTING**\nTest stats -  Loss: {:.4f} Acc: {:.2f}%'.format(test_loss, test_acc*100))            

    print("Inference on Testset complete in {:.1f}s\n".format(time.time() - sincetime))


def save_model(model, hist, trained_models_path, model_type, do_save):
    """
    Saves the trained model.
    """
    if do_save:
        saved_model_path = f"{trained_models_path}/{model_type}_trained_bestValAcc={hist['best_val_acc']}.pth"
        torch.save(model.module.state_dict(), saved_model_path)
        print(f"Model saved at {saved_model_path}")


def plot_training(hist, graphs_path, model_type, do_save, do_plot=False):
    """
    Plots the training and validation loss/accuracy.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title(f'{model_type} - loss')
    ax[0].plot(hist["epochs"], hist["train_loss"], label="Train loss")
    ax[0].plot(hist["epochs"], hist["val_loss"], label="Validation loss")
    ax[1].set_title(f'{model_type} - accuracy')
    ax[1].plot(hist["epochs"], hist["train_acc"], label="Train accuracy")
    ax[1].plot(hist["epochs"], hist["val_acc"], label="Validation accuracy")
    ax[0].legend()
    ax[1].legend()
    if do_save:
        save_graph_path = f"{graphs_path}/{model_type}_training_bestValAcc={hist['best_val_acc']}.jpg"
        plt.savefig(save_graph_path)
        print(f"Training graph saved at {save_graph_path}")
    if do_plot: plt.show()


if __name__ == "__main__":

    ### Main parameters ###
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--trained_models_path', type=str, required=True)
    parser.add_argument('--graphs_path', type=str, required=True)
    parser.add_argument('--saving', type=int, required=True)

    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--frac_data', type=float, required=False, default=1.)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True) 

    args = parser.parse_args()


    # Path
    MODEL_PATH = args.model_path                    #"pretrained_models/ResNet22_mAP=0.430.pth"
    TRAINED_MODELS_PATH = args.trained_models_path  # "models"
    GRAPHS_PATH = args.graphs_path                  # "graphs"
    SAVING = args.saving                            # int 0: no, 1: yes

    # Audio parameters
    SR = SAMPLE_RATE             # Sample Rate
    AUDIO_DURATION = 10          # 10 seconds duration for all audios

    # Model parameters
    MODEL_TYPE = args.model_type      # "Transfer_ResNet22"
    LR = args.lr                      # Learning Rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    # Misc parameters
    FRAC_DATA = args.frac_data        # Takes a {FRAC_DATA}% of the dataset
    RANDOM_STATE = 17
    random.seed(RANDOM_STATE)


    ### Data processing ###
    labels_df = pd.read_csv(FINAL_LABELS_PATH)
    labels_df = labels_df.sample(frac=FRAC_DATA, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Using {int(FRAC_DATA*100)}% of the dataset.")

    NB_SPECIES = len(set(labels_df['label']))      # Number of classes
    print("NB_SPECIES: ", NB_SPECIES)

    (trainloader, validationloader, testloader) = process_data(df=labels_df, batch_size=BATCH_SIZE,                 
                                                               sample_rate=SR, audio_duration=AUDIO_DURATION, 
                                                               random_state=RANDOM_STATE, do_plot=False)
    dataloaders = {"train": trainloader[0],
                   "val": validationloader[0],
                   "test": testloader[0]}
    dataset_sizes = {"train": trainloader[1],
                     "val": validationloader[1],
                     "test": testloader[1]}
    print(dataset_sizes)


    ### Load Model ###
    model = load_model(model_type=MODEL_TYPE, sample_rate=SR, nb_species=NB_SPECIES, model_path=MODEL_PATH)


    ###  Define loss function and optimizer ### 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)


    ### Training ###
    model, history_training = train_model(model=model, criterion=criterion, optimizer=optimizer, 
                                          dataloaders=dataloaders, scheduler=None, num_epochs=EPOCHS)


    ### Testing ###
    test_model(model=model, criterion=criterion, dataloaders=dataloaders)


    ### Save the model ###
    save_model(model=model, hist=history_training, 
               trained_models_path=TRAINED_MODELS_PATH, model_type=MODEL_TYPE, do_save=SAVING)


    ### Plotting the losses ###
    plot_training(hist=history_training, graphs_path=GRAPHS_PATH, 
                  model_type=MODEL_TYPE, do_save=SAVING)
