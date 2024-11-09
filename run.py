from collections import deque
from functools import partial
import time

from utils.config import ROBOFLOW_API_KEY, VIDEO_INPUT
from utils.audio import trigger_fox_sounds

import cv2
from inference import get_roboflow_model, InferencePipeline
from inference.core.interfaces.stream.sinks import (
    render_boxes, DEFAULT_BBOX_ANNOTATOR, DEFAULT_LABEL_ANNOTATOR
)
from inference.core.interfaces.camera.entities import VideoFrame
import supervision as sv

def get_box_center(x, y, w, h):
    print(x,y,w,h)
    return int(x-w//2), int(y-h//2)




class AtticSupervisor:

    detections = deque(maxlen=100)
    last_sound_trigger = 0

    def __init__(self, video_reference, video_output):
        self.video_reference = video_reference
        # TODO: allow input directly from camera
        self.video_info = sv.VideoInfo.from_video_path(self.video_reference)
        self.video_sink = sv.VideoSink(video_output, self.video_info)
        self.annotator = [
            DEFAULT_LABEL_ANNOTATOR,
            DEFAULT_BBOX_ANNOTATOR,
            sv.DotAnnotator()
        ]
        self.pipeline = InferencePipeline.init(
            video_reference=[self.video_reference],
            api_key=ROBOFLOW_API_KEY,
            model_id='attic-activity/2',
            on_prediction=self._on_prediction
        )

    def _on_prediction(self, prediction: dict, video_frame:VideoFrame):
        detections:sv.Detections = sv.Detections.from_inference(prediction)
        if len(detections) > 0 and (time.time() - self.last_sound_trigger) > 30:
            self.last_sound_trigger = time.time()
            trigger_fox_sounds()
        # detections = detections[detections.area > 1000]
        render_boxes(
            predictions=prediction,
            video_frame=video_frame,
            annotator=self.annotator
        )

    def start(self):
        self.pipeline.start()
        self.pipeline.join()
        self.video_sink.release()

if __name__ == "__main__":
    AtticSupervisor(
        video_reference=VIDEO_INPUT,
        video_output='./data/output.mp4'
    ).start()

        


# # optionally, change the confidence and overlap thresholds
# # values are percentages
# model.confidence = 50
# model.overlap = 25



# get roboflow model

# load in target video/camera stream

# run detections on target video/camera stream

# add boxes around detections

# for each detection, determine the center point of the bounding box, and store in history

# if sufficient centroids, calculate coordinates with highest traffic/frequency to determine suggested trap locations

# calculate potential entry/exit points in the attic

# calculate traffic in the attic