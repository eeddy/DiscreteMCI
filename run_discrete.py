import sys
import os
import torch 

# Add the parent directory of 'scripts' (which contains config.py) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libemg.streamers import myo_streamer
from libemg.data_handler import OnlineDataHandler
from libemg.discrete import DiscreteControl

if __name__ == "__main__":
    _, sm = myo_streamer()
    odh = OnlineDataHandler(sm)
    model = torch.load('Other/Discrete.model', map_location=torch.device('cpu'))
    discrete = DiscreteControl(odh, 10, 5, model)
    discrete.run()