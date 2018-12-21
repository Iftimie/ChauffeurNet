import cv2
import os
import numpy as np
import h5py
from util.Vehicle import Vehicle
from GUI import GUI
from GUI import EventBag

class Simulator(GUI):

    def __init__(self, ):
        super(Simulator, self).__init__("Simulator")
        self.camera.set_transform(y=-1500)
        self.vehicle = Vehicle(self.camera)
        self.vehicle.set_transform(x = 100)
        self.world.actors.append(self.vehicle)

        self.event_bag = EventBag("data/recording.h5", record=True)

    # @Override
    def interpret_key(self):
        key = self.pressed_key
        if key == 27:
            self.running = False

    def run(self):

        while self.running:
            self.interact()
            self.event_bag.append([self.pressed_key, GUI.mouse[0], GUI.mouse[1]])

        print ("Game over")

import requests
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
    required_files = [['1V8T8L_UuP4CXUkVjm9dAVfMJsm6Q3NlJ',  'data/world__.h5'],
                      ['17WkZHVgzQEcmwJspR-oJJPOv4mt8GjgX', 'data/recording.h5'],
                      ['1GBdon5rGyjGZm24MaeDjgYI1gHAkYEuj', 'data/pytorch_data.h5'],
                      ['1osWDfzGCSxHtE9mYlF4UsC6II2rzdoT_', '../network/ChauffeurNet.pt']]
    for pair in required_files:
        if not os.path.exists(pair[1]):
            download_file_from_google_drive(pair[0], pair[1])

if __name__ =="__main__":
    check_if_data_exists()
    simulator = Simulator()
    simulator.run()
