from playsound import playsound
from utils.config import FOX_SOUNDS_PATH

def trigger_fox_sounds():
    playsound(FOX_SOUNDS_PATH, block=False)