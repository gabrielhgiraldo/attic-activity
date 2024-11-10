from collections import deque
from functools import partial
import time

from utils.config import ROBOFLOW_API_KEY, VIDEO_INPUT
from utils.audio import trigger_fox_sounds
from utils.traps import create_sliding_zones, get_trap_placements, get_trap_annotators

from inference import get_roboflow_model, InferencePipeline
from inference.core.interfaces.stream.sinks import (
    render_boxes, DEFAULT_BBOX_ANNOTATOR, DEFAULT_LABEL_ANNOTATOR, VideoFileSink, multi_sink
)
from inference.core.interfaces.camera.entities import VideoFrame
import numpy as np
import supervision as sv



class AtticSupervisor:

    detections = sv.Detections.empty()
    last_sound_trigger = 0
    trap_placements = []

    def __init__(self, video_reference, video_output):
        self.video_reference = video_reference
        self.activity_zones = create_sliding_zones()
        
        # self.video_info = sv.VideoInfo.from_video_path(self.video_reference)
        # self.video_sink = sv.VideoSink(video_output, self.video_info)

        self.fps_monitor = sv.FPSMonitor()
        self.annotator = [
            # DEFAULT_LABEL_ANNOTATOR,
            # DEFAULT_BBOX_ANNOTATOR,
            sv.DotAnnotator(),
            # sv.HeatMapAnnotator(),
        ]

        self.video_sink = VideoFileSink.init(
            video_file_name=video_output,
            annotator=self.annotator
        )
                
        self.pipeline = InferencePipeline.init(
            video_reference=[self.video_reference],
            api_key=ROBOFLOW_API_KEY,
            model_id='attic-activity/2',
            # on_prediction=partial(multi_sink, sinks=[self.video_sink.on_prediction, self._on_prediction])
            on_prediction=self._on_prediction
        )

    def _on_prediction(self, prediction: dict, video_frame:VideoFrame):
        detections:sv.Detections = sv.Detections.from_inference(prediction)
        self.detections = detections if self.detections.is_empty() else sv.Detections.merge([self.detections, detections])
        if len(detections) > 0 and (time.time() - self.last_sound_trigger) > 30:
            self.last_sound_trigger = time.time()
            # trigger_fox_sounds()
        if len(self.detections) > 10:
            self.trap_placements = get_trap_placements(self.detections, self.activity_zones)
        # polygon zone annotator are not inhertied from BaseAnnotator
        for annotator in get_trap_annotators(self.trap_placements):
            annotator.annotate(video_frame.image)
        # detections = detections[detections.area > 1000]
        render_boxes(
            predictions=prediction,
            video_frame=video_frame,
            annotator=self.annotator,
            fps_monitor=self.fps_monitor,
            display_statistics=True
        )

    def start(self):
        self.pipeline.start()
        self.pipeline.join()
        self.stop()
        
    def stop(self):
        self.video_sink.release()

if __name__ == "__main__":
    supervisor = AtticSupervisor(
        video_reference=VIDEO_INPUT,
        video_output='./data/output.mp4'
    )
    try:
       supervisor.start()
    finally:
        supervisor.stop()