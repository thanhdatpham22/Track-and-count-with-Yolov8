import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from numpy import random

from trt_loader import TrtModelNMS

# ---------------Object Tracking---------------
#import skimage
# from sort import *  # use sort track
# from ocsort import *  # use ocsort track
from byte_tracker import *  # use byte track


# from deep_ocsort import * # use deep ocsort track

# from deep_sort_pytorch.utils.parser import get_config #use deep sort track
# from deep_sort_pytorch.deep_sort import DeepSort
# from collections import deque

# ---------------------------export csv file---------

# field_names = ['Object', 'Res', 'Cap', 'Inductor', 'Diot', 'Led', 'Ic', 'Other']
# file_path = r"home/tuanpro/Desktop/Machine_learning/YOLOv8/detect_component"
# --------------------------------------------------------------
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


class YOLOR(object):
    def __init__(self,
                 model_weights='/home/tuanpro/Desktop/Machine_learning/YOLOv8/tensorrt/data1.4.trt',
                 max_size=640,
                 img_size=(640, 640),
                 fps=30,
                 names=['res', 'cap', 'ind', 'dio', 'led', 'ic', 'ot']):
        self.names = names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.imgsz = (max_size, max_size)
        # Load model
        self.model = TrtModelNMS(model_weights, max_size)
        self.sort_tracker = BYTETracker(track_thresh=0.4, match_thresh=0.8, track_buffer=25,
                                        frame_rate=12)  # use byte track


    def detect(self, bgr_img, nw, nh, scale):
        # ---------Prediction
        ## Padded resize
        # h, w, _ = bgr_img.shape
        # scale = min(self.imgsz[0]/w, self.imgsz[1]/h)
        inp = np.zeros((self.imgsz[1], self.imgsz[0], 3), dtype=np.float32)
        # nh = int(scale * h)
        # nw = int(scale * w)
        inp[: nh, :nw, :] = cv2.resize(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), (nw, nh))
        inp = inp.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
        inp = np.expand_dims(inp.transpose(2, 0, 1), 0)
        # print(inp.shape)

        ## Inference
        # t1 = time.time()
        num_detection, nmsed_bboxes, nmsed_scores, nmsed_classes = self.model.run(inp)

        ## Apply NMS
        num_detection = num_detection[0][0]
        nmsed_bboxes = nmsed_bboxes[0]
        nmsed_scores = nmsed_scores[0]
        nmsed_classes = nmsed_classes[0]
        # print(nmsed_classes)
        # print(time.time())
        if num_detection > 0:
            print('Detected {} object(s)'.format(num_detection))
        # Rescale boxes from img_size to im0 size
        nmsed_bboxes[:, 0] /= scale
        nmsed_bboxes[:, 1] /= scale
        nmsed_bboxes[:, 2] /= scale
        nmsed_bboxes[:, 3] /= scale
        visualize_img = bgr_img.copy()

        # --------TRACKING-------------------------------------
        # khoi tao data tracking
        dets_to_sort = np.empty((0, 6))  # use for deep_ocsort, ocsort, Byte_tracker

        ## TRACKING USE SORT-OCSORT-BYTR TRACK-----------------------------
        ## tao data cho tracking  use for sort, ocsort, byte track
        for ix in range(num_detection):  # x1, y1, x2, y2 in pixel format
            cls = int(nmsed_classes[ix])
            x1, y1, x2, y2 = nmsed_bboxes[ix]
            det_scores = nmsed_scores[ix]
            #    currentArray = np.array([x1, y1, x2, y2, det_scores]) # sort
            currentArray = np.array([x1, y1, x2, y2, det_scores, cls])  # ocsort, Byte_tracker
            dets_to_sort = np.vstack((dets_to_sort, currentArray))

        ## tracking use for sort, ocsort, byte track-----------------------------
        tracking_result = self.sort_tracker.update(dets_to_sort, '')  # use for ocsort, Byte_tracker

        boxes = []
        track_ids = []
        classtrack = []
        for ix, (x1, y1, x2, y2, id_sort, score, clas, ind) in enumerate(
                tracking_result):  # x1, y1, x2, y2 in pixel format

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            ##cls = int(nmsed_classes[ix])
            label = '%d.%s.%2d' % (id_sort, self.names[int(clas)], int(score * 100))  # tao label cho box
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # label, font, scale, thickless
            ##print(label_size[0][0], label_size[0][1])
            if y1 < 10:
                cv2.rectangle(visualize_img, (int(x1), int(y1)), (int(x2), int(y2)), self.colors[int(clas)],
                              2)  # VE KHUNG BBOX
                cv2.rectangle(visualize_img, (int(x1), int(y1)),
                              (int(x1 + label_size[0][0]), int(y1 + 1 + label_size[0][1])), self.colors[int(clas)],
                              cv2.FILLED)  # VE KHUNG TEXT trong bbox
                cv2.putText(visualize_img, label, (int(x1), int(y1 + label_size[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA)  # VE TEXT trong bbox
            else:
                cv2.rectangle(visualize_img, (int(x1), int(y1)), (int(x2), int(y2)), self.colors[int(clas)], 2)
                cv2.rectangle(visualize_img, (int(x1), int(y1)),
                              (int(x1 + label_size[0][0]), int(y1 - label_size[0][1])), self.colors[int(clas)],
                              cv2.FILLED)  # VE KHUNG TEXT ngoai bbox
                cv2.putText(visualize_img, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)  # VE TEXT ngoai bbox

            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            w = abs(x1 - x2)
            h = abs(y1 - y2)
            # box = [cx, cy , w, h]
            boxes.append([cx, cy, w, h])
            track_ids.append(id_sort)
            classtrack.append(clas)
        '''print('boxes')
        print(boxes)
        print('track_ids')
        print(track_ids)
        print('classtrack')
        print(classtrack)'''

        return visualize_img, boxes, track_ids, classtrack


# Point start and end for Line
START = (0, 500)
END = (1000, 500)

# Store the track history
track_history = defaultdict(lambda: [])

# Create a dictionary to keep track of objects that have crossed the line
crossed_objects = {}

count_res = 0
count_cap = 0
count_ind = 0
count_dio = 0
count_led = 0
count_ic = 0
count_ot = 0
count_all = 0

if __name__ == '__main__':

    video_path = '/home/tuanpro/Desktop/Machine_learning/YOLOv8/detect_component/lk_doc.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('/home/tuanpro/Desktop/Machine_learning/YOLOv8/tracking/video_detected/Bedroom.data1.4-bytetrack.avi',fourcc,int(fps/7),(width,height))
    print('original image:', height, width, ' FPS =', fps)

    scale = min(640 / width, 640 / height)  # lay ti le khung hinh
    nh = int(scale * height)
    nw = int(scale * width)
    print('image detect:', nh, nw)

    # model = YOLOR(model_weights = "/home/tuanpro/Desktop/Machine_learning/YOLOv8/detect_component/detect_component.trt",names = ['res','cap','ind','dio','led','ic','ot'],img_size = (height,width),fps = fps)
    model = YOLOR(model_weights="/home/tuanpro/Desktop/Machine_learning/YOLOv8/detect_component/best_yolov8s.trt",
                  names=['res', 'cap', 'ind', 'dio', 'led', 'ic', 'ot'], img_size=(height, width), fps=fps)
    current_frame = 0
    # time_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, boxes, track_ids, classtrack = model.detect(frame, nw, nh, scale)

        # ---------------------------------------------------------
        # Plot the tracks and count objects crossing the line
        for i, (box, track_id, classed) in enumerate(zip(boxes, track_ids, classtrack)):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track_ids) > 50:  # retain 30 tracks for 30 frames
                track.pop(0)

            # Check if the object crosses the line
            if START[0] < x < END[0] and abs(y - START[1]) < 20:
                if i < len(classtrack):  # Ensure index is within the bounds of classtrack
                    class_info = classtrack[i]  # Use index instead of track_id
                    if track_id not in crossed_objects:
                        crossed_objects[track_id] = {
                            'class': class_info}  # Add class_info to track_id in crossed_object dict
                # count
                count_res = sum(1 for value in crossed_objects.values() if value.get(
                    'class') == 0.0)  # loop for crossed_object.values and get values has 'class'==0.0
                count_cap = sum(1 for value in crossed_objects.values() if value.get('class') == 1.0)
                count_ind = sum(1 for value in crossed_objects.values() if value.get('class') == 2.0)
                count_dio = sum(1 for value in crossed_objects.values() if value.get('class') == 3.0)
                count_led = sum(1 for value in crossed_objects.values() if value.get('class') == 4.0)
                count_ic = sum(1 for value in crossed_objects.values() if value.get('class') == 5.0)
                count_ot = sum(1 for value in crossed_objects.values() if value.get('class') == 6.0)
                count_all = len(crossed_objects)
                # print('track_id:',track_id)
                # print('class_info',class_info)
                # print(crossed_objects)

                # Annotate the object as it crosses the line
                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                              (244, 0, 0), 2)

            # Draw the line on the frame
            cv2.line(annotated_frame, (START[0], START[1]), (END[0], END[1]), (255, 255, 255), 2)

            # Write the count of objects on each frame
            cv2.putText(annotated_frame, f"Objects crossed: {len(crossed_objects)}", (int(10), int(30)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Res: {count_res}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 150), 2)
            cv2.putText(annotated_frame, f"Cap: {count_cap}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 120), 2)
            cv2.putText(annotated_frame, f"Inductor: {count_ind}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 90),
                        2)
            cv2.putText(annotated_frame, f"Diot: {count_dio}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 60), 2)
            cv2.putText(annotated_frame, f"Led: {count_led}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 30), 2)
            cv2.putText(annotated_frame, f"IC: {count_ic}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(annotated_frame, f"Other: {count_ot}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 100), 2)

        # ---------------------------------------------------------

        time_label = '%d' % (int(current_frame / fps))
        current_frame += 1
        cv2.putText(annotated_frame, time_label, (int(20), int(30)), cv2.FONT_HERSHEY_COMPLEX, 1, (208, 50, 45), 2,
                    cv2.LINE_AA)
        print(f'Speed Farme: {fps}')
        cv2.imshow('image', annotated_frame)
        # out.write(img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # out.release()
    cap.release()
    cv2.destroyAllWindows()