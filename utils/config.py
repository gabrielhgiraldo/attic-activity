import os

from dotenv import load_dotenv


load_dotenv()

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
VIDEO_INPUT = os.getenv('ATTIC_VIDEO_INPUT', './data/input.mp4')
FOX_SOUNDS_PATH = os.getenv('FOX_SOUNDS_PATH', './data/fox_sounds.mp3')