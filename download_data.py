import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
def check_if_data_exists():
    #https://drive.google.com/open?id=1KUaeIqIQh8ECjNdIFieVa6MW7DYgVrwg
    required_files = [['1azkG7nXZhsVbBD04ka8D8S-YQyHmT_6W',  'data/world.h5'],
                      ['1yPmdlxo-Uhc7jrXnOtx-E6_89FFKVofM', 'data/recorded_states.pkl'],
                      ['1KUaeIqIQh8ECjNdIFieVa6MW7DYgVrwg', 'data/ChauffeurNet.pt']]
    if not os.path.exists("data"):
        os.mkdir("data")
    for pair in required_files:
        if not os.path.exists(pair[1]):
            print ("downloading file from drive", pair[1])
            download_file_from_google_drive(pair[0], pair[1])
