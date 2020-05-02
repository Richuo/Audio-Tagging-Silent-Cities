import os
import sys
import time
import urllib.request, json

import re
import pandas as pd
import matplotlib.pyplot as plt


# Birds list
BIRDSLIST = ['Sturnus vulgaris',
             'Linaria cannabina',
             'Regulus regulus',
             'Cyanistes caeruleus',
             'Phylloscopus collybita',
             'Carduelis carduelis',
             
             'Parus major',
             'Fringilla coelebs',
             'Erithacus rubecula',
             'Passer domesticus',
             'Hirundo rustica',
             'Columba palumbus',
             'Turdus merula',
            ]

BANNEDIDS = [465779, 456971] # Audio that has an issue when converting mp3 to wav.


# Dataset paths
LABELS_PATH = "labels"
DATA_PATH_UNFILTERED = f"{LABELS_PATH}/unfiltered_birds_df.csv"
DATA_PATH_FILTERED = f"{LABELS_PATH}/filtered_birds_df.csv"
FINAL_LABELS_PATH = f"{LABELS_PATH}/all_data.csv"
AUDIO_TYPE = ' type:song'

DATA_DIR = "audiodata/mp3"
DATA_DIR_WAV = "audiodata/wav"


# Create dir if it doesn't exist
for path in [LABELS_PATH, DATA_DIR, DATA_DIR_WAV]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{DATA_DIR_WAV} folder created.")



def retrieve_jsondata(birdName, page=1):
    """
    Returns the json data of the specific bird specie.
    """
    searchTerms = birdName + AUDIO_TYPE
    url = 'https://www.xeno-canto.org/api/2/recordings?query={0}&page={1}'.format(
            searchTerms.replace(' ', '%20'), page)
    
    jsonPage = urllib.request.urlopen(url)
    jsondata = json.loads(jsonPage.read().decode('utf-8'))
    
    return jsondata


def get_birdsdict(birdName):
    """
    Returns a dictionary (key is an ID given by xeno-canto) of the quality, length, country and url of the audio files. 
    """
    main_keys = ['q', 'length', 'cnt', 'file']
    jsondata = retrieve_jsondata(birdName, page=1)
    
    # Store all of the other pages in one list
    numPages = jsondata['numPages']
    jsonlist = jsondata['recordings']
    for page in range(2,numPages+1):
        new_jsondata = retrieve_jsondata(birdName, page)
        jsonlist = jsonlist + new_jsondata['recordings']
        time.sleep(0.1) # small delay between API requests
    
    # Store the data in a dictionary
    dict_recordings = {}
    for listdata in jsonlist:
        birdID = listdata['id']
        birdata = []
        add_data = True # Don't add any non valid data
        
        for key in main_keys:
            data = listdata[key]
            
            # Convert "min:seconds" time format to s seconds.
            if key == 'length':
                # Don't add any audio file that has its duration higher than an hour
                if len(data) >= 6:
                    add_data = False
                    break;

                list_time = re.split(':', data)
                data = int(list_time[0]) * 60 + int(list_time[1])

            # Remove the first 2 elements of the string ('//www.xeno-canto.org/id/download')
            if key == 'file':
                data = data[2:]

            birdata.append(data)

        if add_data:
            dict_recordings[birdID] = birdata

    return dict_recordings


def convert_dict2df(dicto, birdName):
    """
    Returns the dataframe version of the specific bird specie data dictionary.
    """
    df = pd.DataFrame.from_dict(dicto, orient='index', 
                                columns=['quality', 'length', 'country','url']).reset_index()
    df = df.rename(columns={'index':'iD'})
    df["birdName"] = birdName
    
    return df


def get_all_df(birdslist):
    """
    Returns the dataframe of all the bird species in birdlist.
    """
    dataframes = []
    for birdName in birdslist:
        print(f"Retrieving data for {birdName}")
        birdict = get_birdsdict(birdName)
        birdf = convert_dict2df(birdict, birdName)
        dataframes.append(birdf)
    
    final_df = pd.concat(dataframes).reset_index(drop=True)
    
    return final_df


def get_filtered_df(df, bannedList):
    """
    Returns a filtered dataframe with audio quality == 'A' or 'B', duration between 10 and 100 seconds
    and filter out the bannedIDs.
    """
    filtered_df = df.loc[(~df['birdName'].isin(bannedList))
                         & ((df['quality'] == 'A') | (df['quality'] == 'B'))
                         & ((df['length'] >= 10) & (df['length'] < 100))]
    
    return filtered_df



if __name__ == "__main__":

    BirdsList = BIRDSLIST
    BannedIDs = BANNEDIDS

    df = get_all_df(BirdsList)
    print("Unfiltered Dataframe Shape:", df.shape)

    # Save to csv
    df.to_csv(DATA_PATH_UNFILTERED, index=False)

    filtered_df = get_filtered_df(df, BannedIDs)
    print("Filtered Dataframe Shape:", filtered_df.shape)

    # Save to csv
    filtered_df.to_csv(DATA_PATH_FILTERED, index=False)


