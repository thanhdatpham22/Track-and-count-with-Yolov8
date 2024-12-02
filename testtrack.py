import cv2
import time
from collections import Counter
from collections import defaultdict
import supervision as sv
import numpy as np
from pyparsing import results
from ultralytics import YOLO
#Load model yolov model
model=YOLO('best_yolov8s.pt')
#Open the file video
#cap = cv2.VideoCapture(r"E:\NCKH Part 2\Video test\lk_doc.mp4")
cap = cv2.VideoCapture(0)
# Loop through the video farmes

totalid=set()
track_id=dict()

#Toa do line
START =   sv.Point(0,1600)
END = sv.Point(10000, 1600)
color = (0, 255, 0)  # Màu xanh lá cây
thickness = 10
count_k=globals()
# Store the track history
track_history = defaultdict(lambda: [])
crossed_objects={}
while cap.isOpened():
    #Read a frame from the video
    success, farme=cap.read()
    if success:
        count = dict()
        #og_frame=cv2(farme,0)
        #Run YOLOv8 tracking on the farme, persisting tracks between farmes
        #results =model.track(farme,save=True, persist=True)
        results = model.track(farme, persist=True)
        for result in results:
            if result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu().tolist()
                print(boxes)
                probs = result.probs  # Class probabilities for classification outputs
                annotated_frame = results[0].plot()
                id_box = results[0].boxes.id.int().cpu().tolist()
                classtrack = results[0].boxes.cls.cpu().tolist()  # Convert tensor to list
                toltalid = (len(id_box))

                print(classtrack)
                print(id_box)
                #print(toltalid)
                #cars=classtrack.count(2.0)
                #trucks=classtrack.count(7.0)
                #print(classtrack.count(2.0))
                #totalid.add(id_box)

                id1 = 0
                while(id1 < len(id_box)):
                    track_id[id_box[id1]] = classtrack[id1]
                    id1 += 1
                print(track_id)
                count_k = Counter(track_id.values())
                #print(classtrack[id])

                """id2 = 0
                for key in track_id:
                    track_id.setdefault(key, classtrack[id2])
                    if id2 < len(classtrack) - 1:
                        id2 += 1
                    else:
                        id2 = 0"""


                #points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                #cv2.polylines(farme, [points], isClosed=False, color=(230, 230, 230), thickness=50)
                #cv2.line(farme, (START.x, START.y), (END.x, END.y), (0, 255, 0), 5)
                    #Hien thi len man hinh cv2
                cv2.putText(farme, f"Total Vehices: {len(track_id)}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                        (255, 0, 255),
                                        10, cv2.LINE_AA)
                cv2.putText(farme, f"Car: {count_k[0.0]}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                       (100, 0, 200),
                                       10, cv2.LINE_AA)
                cv2.putText(farme, f"Truck: {count_k[1.0]}", (50,250), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                       (50, 0, 255),
                                       10, cv2.LINE_AA)
                

                # Vẽ đường đếm lên hình ảnh
                #cv2.line(farme,START,END,color,thickness)
                cv2.imshow('Video',annotated_frame)
                #cv2.imshow("Video", cv2.resize(results[0].plot(),(800,600)))
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
    else:
        # Break the loop if the end of the video is reached
        break
    # Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


