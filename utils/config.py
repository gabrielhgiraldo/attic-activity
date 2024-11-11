import os

from dotenv import load_dotenv


load_dotenv()

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
VIDEO_INPUT = os.getenv('ATTIC_VIDEO_INPUT', './data/input.mp4')
FOX_SOUNDS_PATH = os.getenv('FOX_SOUNDS_PATH', './data/fox_sounds.mp3')


NOISE_DURATION_S = 30 # how long to play noise for
TRAP_PLACEMENT_DELAY_S = 5 # how often to update trap placement
ACCESSWAY_DELAY_S = 5 # how long a rodent needs to be missing before marking a spot as an access