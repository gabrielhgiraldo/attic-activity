from collections import deque
from functools import partial

from utils.config import ROBOFLOW_API_KEY, VIDEO_INPUT

import cv2
from inference import get_roboflow_model, InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes, display_image, ImageWithSourceID
from inference.core.interfaces.camera.entities import VideoFrame
import supervision as sv

def get_box_center(x, y, w, h):
    print(x,y,w,h)
    return int(x-w//2), int(y-h//2)




class AtticSupervisor:
    def __init__(self, video_reference, video_output):
        self.video_reference = video_reference
        # TODO: allow input directly from camera
        self.video_info = sv.VideoInfo.from_video_path(self.video_reference)
        self.video_sink = sv.VideoSink(video_output, self.video_info)
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        self.detections = deque(maxlen=100)

        self.pipeline = InferencePipeline.init(
            video_reference=[self.video_reference],
            api_key=ROBOFLOW_API_KEY,
            model_id='attic-activity/2',
            on_prediction=self._on_prediction
        )

    def process_frame(self, image:ImageWithSourceID, detections:sv.Detections):
        centers = [get_box_center(*box) for box in detections.xyxy]
        print(centers)
        annotated_frame = self.annotate_frame(image[1], centers)
        cv2.imshow("Attic Activity", annotated_frame)
        cv2.waitKey(1)
        

    def annotate_frame(self, frame, centroids):
        # annotated_frame = self.box_annotator.annotate(frame.copy(), detections)
        annotated_frame = frame.copy()
        for centroid in centroids:
            print(centroid)
            cv2.circle(annotated_frame, centroid, 0, color=(0,0, 255), thickness=2)
        return annotated_frame

    def _on_prediction(self, prediction: dict, video_frame:VideoFrame):
        detections:sv.Detections = sv.Detections.from_inference(prediction)
        # detections = detections[detections.area > 1000]
        render_boxes(prediction, video_frame, on_frame_rendered=partial(self.process_frame, detections=detections))

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