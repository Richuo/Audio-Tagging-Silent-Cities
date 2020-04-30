import os
import sys
import time
import urllib.request, json

import re
import pandas as pd

from create_birds_dataset import DATA_DIR, LABELS_PATH, DATA_PATH_FILTERED


def download_birds_audio(df, birdName, do_print=True):
    """
    Download the audio files from the dataframe, by birdName.
    Returns IDs that have invalid url in a list.
    """
    if do_print: print(f"Downloading for {birdName}")
    nonValidIDList = []
    newbirdName = birdName.replace(" ", "_").lower()
    path = DATA_DIR + "/" + newbirdName
    
    # Create directory if it does not exist.
    if not os.path.exists(path):
        os.makedirs(path)
    birdf = df.loc[df['birdName'] == birdName]
    
    # Downloading the file as .mp3
    for idx in range(len(birdf)):
        iD = birdf.iloc[idx]['iD']
        filename = f"{path}/{newbirdName}_{iD}.mp3"
        if not os.path.isfile(filename):
            url = f"http://{birdf.iloc[idx]['url']}"
            try:
                urllib.request.urlretrieve(url, filename)
            except:
                # Add unretrievable files' id
                nonValidIDList.append(iD)
            time.sleep(0.1) # delay between API requests

    if do_print: print("Downloading finished.")
    return nonValidIDList


def download_all_birds_audio(df, birdslist):
    """
    Download all the audio files from the dataframe.
    Returns IDs that have invalid url in a dictionary where birdName are the dict keys.
    """
    nonValidFiles = {}
    for birdName in birdslist:
        nonValidIDList = download_birds_audio(df, birdName)
        nonValidFiles[birdName] = nonValidIDList
    
    return nonValidFiles



if __name__ == "__main__":

    df = pd.read_csv(DATA_PATH_FILTERED)

    birdsList = list(set(df['birdName']))

    # Sanity check of the urls
    for url in df['url']:
        if url[:4] != "www.":
            print("WRONG URL IN THE DATASET !!")
            break;

    # Start downloading (approximately 7.25 Go with the original birdList)
    nonValidFiles = download_all_birds_audio(df, birdsList)

    nonValidList = []
    for IDList in nonValidFiles.values():
        for ID in IDList:
            nonValidList.append(ID)

    updatedf = df.loc[~df['iD'].isin(nonValidList)].reset_index(drop=True)
    print(f"{len(df)-len(updatedf)} non valid elements removed")
    print("Updated df shape:",updatedf.shape)
    
    # TODO 
    # (Those unvalid files won't affect the training because it won't be added in the Dataloader)
    # Create new Dataframe for ValidAudio only.
