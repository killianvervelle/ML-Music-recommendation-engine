import numpy as np
import pandas as pd
import pyarrow as pa
import time
import os
import glob
import csv
import requests
import urllib
import tarfile
from hdf5_getters import *
import pyarrow.parquet as pq
from geopy.geocoders import Nominatim
import urllib.request

pd.set_option('display.max_columns', None)  # Display all columns

# *****************************************************************************
#                  GLOBAL VARIABLES
# *****************************************************************************

csv_path = 'data/data.csv'
api_key = "059e637024c2da6d558a09dfa118a79a"
i = 0

# *****************************************************************************
#                  FUNCTIONS
# *****************************************************************************

# Get the country of an artist based on his Long and Lat coordinates
def get_country(latitude, longitude, df):
    global i
    if i == len(df):
        print(i)
        i = 0
        return
    if latitude is None or longitude is None or np.isnan(latitude) or np.isnan(longitude):
        df['artist_location'][i] = np.nan
        print(i)
        i += 1
        return np.nan
    geolocator = Nominatim(user_agent="geo_app")
    location = geolocator.reverse(f"{latitude}, {longitude}", exactly_one=True)
    if location is not None and 'address' in location.raw:
        df['artist_location'][i] = location.raw['address'].get('country', '')
        print(i)
        i += 1
        return location.raw['address'].get('country', '')

# Get the music genre of an artiste based on his name and by using Lastfm's API
def get_artist_genre(artist_name, df):
    global i
    base_url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "artist.getinfo",
        "artist": artist_name,
        "api_key": api_key,
        "format": "json"
    }
    if i == len(df):
        print(i)
        i = 0
        return
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        try:
            artist = data["artist"]
            if "tags" in artist and "tag" in artist["tags"]:
                df['artist_genre'][i] = [tag["name"] for tag in artist["tags"]["tag"]]
                print(i)
                i += 1
                return [tag["name"] for tag in artist["tags"]["tag"]]
        except:
            df['artist_genre'][i] = np.nan
            print(i)
            i += 1
            return np.nan
    else:
        print('Error 6: Failed request')
        return

def get_all_files(basedir,ext='.h5') :
    """
    From a root directory, go through all subdirectories
    and find all files with the given extension.
    Return all absolute paths in a list.
    """
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files :
            allfiles.append( os.path.abspath(f) )
    return allfiles

# Multiprocessing function
def process_chunk(chunk):
    global i
    results = []
    for row in chunk.itertuples(index=False):
        time.sleep(3)
        result = get_country(row.artist_latitude, row.artist_longitude)
        if result is np.nan or None:
            results.append(np.nan)
            i+=1
        else:
            results.append(result)
            i+=1
    return results

# Data augmentation functions
def data_augmentation_location(df):
    return df.apply(lambda row: get_country(row['artist_latitude'], row['artist_longitude'], df), axis=1)

def data_augmentation_genre(df):
    df["artist_genre"] = None
    return df.apply(lambda row: get_artist_genre(row['artist_name'], df), axis=1)


# *****************************************************************************
#                  DATA AUGMENTATION - ANALYTICS
# *****************************************************************************


def get_data ():
    # Download MillionSong subset from url
    urllib.request.urlretrieve("http://labrosa.ee.columbia.edu/~dpwe/tmp/millionsongsubset.tar.gz", "data/subset.gz")
    
    # Specify the path of the .gz file
    gz_file_path = 'data/subset.gz'

    # Specify the output file path after unzipping
    output_directory_path = 'data/'

    # Open the .gz file in read mode and extract the contents to the output directory
    with tarfile.open(gz_file_path, 'r:gz') as gz_file:
        gz_file.extractall(output_directory_path)
    
    # Retrieve all .h5 files 
    files = get_all_files('data/MillionSongSubset', ext='.h5')
    
    # Create a list of variable names
    header = [
    'artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude',
    'song_id', 'title', 'song_hotttnesss', 'similar_artists', 'artist_terms', 'artist_terms_freq',
    'artist_terms_weight', 'duration', 'time_signature', 'time_signature_confidence',
    'beats_start', 'beats_confidence', 'key', 'key_confidence', 'loudness', 'energy',
    'mode', 'mode_confidence', 'tempo', 'year'
    ]

    # Open the CSV file in write mode
    with open(csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()  # Write the header row
        # Iterate over the .h5 files
        for file in files:
            # Open the .h5 file using a context manager
            with open_h5_file_read(file) as data:
                # Select only the relevant features
                row = {
                    'artist_id': get_artist_id(data),
                    'artist_name': get_artist_name(data),
                    'artist_location': get_artist_location(data),
                    'artist_latitude': get_artist_latitude(data),
                    'artist_longitude': get_artist_longitude(data),
                    'song_id': get_song_id(data),
                    'title': get_title(data),
                    'song_hotttnesss': get_song_hotttnesss(data),
                    'similar_artists': get_similar_artists(data),
                    'artist_terms': get_artist_terms(data),
                    'artist_terms_freq': get_artist_terms_freq(data),
                    'artist_terms_weight': get_artist_terms_weight(data),
                    'duration': get_duration(data),
                    'time_signature': get_time_signature(data),
                    'time_signature_confidence': get_time_signature_confidence(data),
                    'beats_start': get_beats_start(data),
                    'beats_confidence': get_beats_confidence(data),
                    'key': get_key(data),
                    'key_confidence': get_key_confidence(data),
                    'loudness': get_loudness(data),
                    'energy': get_energy(data),
                    'mode': get_mode(data),
                    'mode_confidence': get_mode_confidence(data),
                    'tempo': get_tempo(data),
                    'year': get_year(data)
                }          
                for key, value in row.items():
                    if isinstance(value, bytes):
                        row[key] = value.decode()
                    elif isinstance(value, list):
                        row[key] = [item.decode() for item in value]
                    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.bytes_):
                        row[key] = [val.decode() for val in value]
                    else:
                        row[key] = value
                writer.writerow(row)

    # Convert the DataFrame to a pyarrow Table
    df = pd.read_csv(csv_path)
    
    # Data augmentation: get artiste location for each music from longitude and latitude coordinates
    data_augmentation_location(df)
    
    # Data augmentation: get artiste genre for each music
    data_augmentation_genre(df)
    
    # Write the pyarrow Table to a Parquet file
    table = pa.Table.from_pandas(df)
    pq.write_table(table, "Scala/src/main/data/data.parquet")
        
get_data()