import cv2
import csv
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO
from pathlib import Path
# Load the YOLOv8 model
model = YOLO('best_yolov8s.pt')

# Set up video capture
cap = cv2.VideoCapture(r"E:\NCKH Part 2\Video test\lk_doc.mp4")
#Get point Mouse
'''def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)'''
# Define the line coordinates
'''START = sv.Point(182, 254)
END = sv.Point(462, 254)'''

#Point start and end for Line
START = sv.Point(0,500)
END = sv.Point(1000, 500)

# Store the track history
track_history = defaultdict(lambda: [])
# Create a dictionary to keep track of objects that have crossed the line
crossed_objects = {}

count_res=0
count_cap=0
count_ind=0
count_dio=0
count_led=0
count_ic=0
count_ot=0
count_all=0
#-----------------------------------------------------------------------------
#field_names = ['Object', 'Res', 'Cap', 'Inductor', 'Diot', 'Led', 'Ic', 'Other']
#file_path = r'E:/NCKH Part 2/report.csv'
#---------------------------------------------------------
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        #results = model.track(frame, classes=[2, 3, 5, 7], persist=True, save=True, tracker="bytetrack.yaml")
        results = model.track(frame,  persist=True, tracker="bytetrack.yaml")
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classtrack = results[0].boxes.cls.cpu().tolist()
        print(track_ids)
        print(classtrack)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        #detections = sv.Detections.from_yolov8(results[0])

        # Plot the tracks and count objects crossing the line
        for i, (box, track_id, class_info) in enumerate(zip(boxes, track_ids, classtrack)):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((round(float(x),1), round(float(y),1)))  # x, y center point
            print(track)
            if len(track_ids) > 10:  # retain 10 tracks for 10 frames
                track.pop(0)
            # Check if the object crosses the line
            if START.x < x < END.x and abs(y - START.y) < 20:
                if i < len(classtrack):  # Ensure index is within the bounds of classtrack
                    class_info = classtrack[i]  # Use index instead of track_id
                    if track_id not in crossed_objects:
                        crossed_objects[track_id] = {'class': class_info}# Add class_info to track_id in crossed_object dict
                #count
                count_res = sum(1 for value in crossed_objects.values() if value.get('class') == 0.0)#loop for crossed_object.values and get values has 'class'==0.0
                count_cap = sum(1 for value in crossed_objects.values() if value.get('class') == 1.0)
                count_ind = sum(1 for value in crossed_objects.values() if value.get('class') == 2.0)
                count_dio = sum(1 for value in crossed_objects.values() if value.get('class') == 3.0)
                count_led = sum(1 for value in crossed_objects.values() if value.get('class') == 4.0)
                count_ic  = sum(1 for value in crossed_objects.values() if value.get('class') == 5.0)
                count_ot  = sum(1 for value in crossed_objects.values() if value.get('class') == 6.0)
                count_all = len(crossed_objects)
                #count_all = f"Objects crossed: {len(crossed_objects)}"
                print('track_id:',track_id)
                print('class_info',class_info)
                print(crossed_objects)
                print(count_ot)
                # Annotate the object as it crosses the line
                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (244, 0, 0), 2)

            # Draw the line on the frame
            cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (255, 255, 255), 2)


            # Write the count of objects on each frame
            cv2.putText(annotated_frame, f"Objects crossed: {count_all}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Res: {count_res}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,150),2)
            cv2.putText(annotated_frame, f"Cap: {count_cap}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 120), 2)
            cv2.putText(annotated_frame, f"Inductor: {count_ind}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 90), 2)
            cv2.putText(annotated_frame, f"Diot: {count_dio}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 60), 2)
            cv2.putText(annotated_frame, f"Led: {count_led}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 30), 2)
            cv2.putText(annotated_frame, f"IC: {count_ic}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(annotated_frame, f"Other: {count_ot}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 100), 2)



        #cv2.imshow("RGB",cv2.resize(annotated_frame,(800,600)))
        cv2.imshow("RGB", annotated_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        # Write the frame with annotations to the output video
    else:
        break

# Write counts to a CSV report
'''def write_report(counts):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerow({field: counts.get(field, 0) for field in field_names})
# Release the video capture
counts = {
    'Object': count_all,
    'Res': count_res,
    'Cap': count_cap,
    'Inductor': count_ind,
    'Diot': count_dio,
    'Led': count_led,
    'Ic': count_ic,
    'Other': count_ot
}
write_report(counts)'''


cap.release()
cv2.destroyAllWindows()