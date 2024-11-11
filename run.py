from collections import deque
from functools import partial
import time
from typing import List

from utils.config import (
    ROBOFLOW_API_KEY, VIDEO_INPUT, NOISE_DURATION_S, ACCESSWAY_DELAY_S, TRAP_PLACEMENT_DELAY_S
)
from utils.audio import trigger_fox_sounds
from utils.traps import create_sliding_zones, trigger_activity_zones, get_trap_annotators

import cv2
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import (
    render_boxes, DEFAULT_BBOX_ANNOTATOR, DEFAULT_LABEL_ANNOTATOR, VideoFileSink, multi_sink
)
from inference.core.interfaces.camera.entities import VideoFrame
from inference_sdk import InferenceHTTPClient
import supervision as sv

# TODO: scale up to multiple simultaneous rodents
class AtticSupervisor:

    historical_detections = sv.Detections.empty()
    last_detections = sv.Detections.empty()
    last_sound_trigger = 0
    last_trap_placement = 0
    last_detection_time = 0
    last_save_time = 0
    trap_placements = []

    def __init__(self, video_reference, video_output, use_server=True):
        if not isinstance(video_reference, list):
            video_reference = [video_reference]
        self.video_reference = video_reference
        self.activity_zones = create_sliding_zones()
        self.time_since_last_detection = 0
        self.accessways = set()
        
        # self.video_info = sv.VideoInfo.from_video_path(self.video_reference)
        # self.video_sink = sv.VideoSink(video_output, self.video_info)

        self.fps_monitor = sv.FPSMonitor()
        self.annotator = [
            DEFAULT_LABEL_ANNOTATOR,
            DEFAULT_BBOX_ANNOTATOR,
            # sv.DotAnnotator(),
            # sv.HeatMapAnnotator(),
        ]
        if use_server:
            self.inference_client = InferenceHTTPClient(
                api_key=ROBOFLOW_API_KEY,
                api_url='http://localhost:9001'
                # api_url='https://detect.roboflow.com'
            )

        self.video_sink = VideoFileSink.init(
            video_file_name=video_output,
            annotator=self.annotator,
            fps_monitor=self.fps_monitor,
            display_statistics=True,
            output_fps=12
        )
        if use_server:
            self.pipeline = InferencePipeline.init_with_custom_logic(
                video_reference=self.video_reference,
                on_video_frame=self._handle_video_frames,
                # on_prediction=partial(multi_sink, sinks=[self.video_sink.on_prediction, self._on_prediction])
                on_prediction=self._on_prediction
            )
        else:
            self.pipeline = InferencePipeline.init(
                video_reference=self.video_reference,
                api_key=ROBOFLOW_API_KEY,
                model_id='attic-activity/2',
                on_prediction=partial(multi_sink, sinks=[self.video_sink.on_prediction, self._on_prediction])
            )
    
    def update_accessways(self, detections):
        try:
            entry_zone:sv.PolygonZone = trigger_activity_zones(detections, self.activity_zones)[0]
            self.accessways.add(entry_zone)
        except IndexError:
            print('failed to detect entry zone')

        if self.last_detection_time is not None:
            try:
                exit_zone = trigger_activity_zones(self.last_detections, self.activity_zones)[0]
                self.accessways.add(exit_zone)
            except:
                print('failed to detect exit zone')
    
    def update_trap_placements(self):
        self.last_trap_placement = time.time()
        self.trap_placements = trigger_activity_zones(self.historical_detections, self.activity_zones)

    def annotate_frame(self, video_frame:VideoFrame):
        # annotate potential accessways for traps/remediation
        for i, annotator in enumerate(get_trap_annotators(self.trap_placements, n_traps=20)):
            annotator.annotate(video_frame.image, label=f"trap {i}")
        for i, annotator in enumerate(get_trap_annotators(list(self.accessways), n_traps=5, color=sv.Color.RED)):
            annotator.annotate(video_frame.image, label=f'access {i}')

    def _handle_video_frames(self, video_frames:List[VideoFrame]):
        return [self.inference_client.infer(video_frame.image, model_id='attic-activity/2') for video_frame in video_frames]

    def _on_prediction(self, prediction: dict, video_frame:VideoFrame):
        detections:sv.Detections = sv.Detections.from_inference(prediction)
        self.historical_detections = detections if self.historical_detections.is_empty() else sv.Detections.merge([self.historical_detections, detections])
        
        if len(detections) > 0:
            # detection_time = video_frame.frame_timestamp.timestamp()
            detection_time = time.time()
            # check for potential entry/exit
            if time.time() - self.last_detection_time > ACCESSWAY_DELAY_S:
                self.update_accessways(detections)
            # trigger noise on detection
            if detection_time - self.last_sound_trigger > NOISE_DURATION_S:
                self.last_sound_trigger = detection_time
                trigger_fox_sounds()
                
            if len(self.historical_detections) > 10 and (time.time() - self.last_trap_placement) > TRAP_PLACEMENT_DELAY_S:
                self.update_trap_placements()
            self.last_detections = detections
            self.last_detection_time = detection_time

        
        self.annotate_frame(video_frame)
        # detections = detections[detections.area > 1000]
        if time.time() - self.last_save_time > 10:
            cv2.imwrite('data/attic_activity.jpg', video_frame.image)
            self.last_save_time = time.time()

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
        video_output='./data/output.mp4',
        use_server=True
    )
    try:
       supervisor.start()
    finally:
        supervisor.stop()