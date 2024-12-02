import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from collections import defaultdict
from ultralytics import YOLO

# Global variables for OpenCV-related objects and flags
cap = None
is_camera_on = False
video_paused = False

# Define start and end points for the counting line
START = (0, 500)
END = (1020, 500)

# Initialize crossed objects dictionary and track history
crossed_objects = {}
track_history = defaultdict(lambda: [])

# Create a YOLO model instance
model = YOLO('best_yolov8s.pt')


# Function to read classes from coco.txt
def read_classes_from_file(file_path):
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file]
    return classes


# Function to start the webcam feed
def start_webcam():
    global cap, is_camera_on, video_paused
    if not is_camera_on:
        cap = cv2.VideoCapture(0)
        is_camera_on = True
        video_paused = False
        update_canvas()


# Function to stop the webcam feed
def stop_webcam():
    global cap, is_camera_on, video_paused
    if cap is not None:
        cap.release()
        cap = None
    is_camera_on = False
    video_paused = False


# Function to pause or resume the video
def pause_resume_video():
    global video_paused
    video_paused = not video_paused


# Function to start video playback from a file
def select_file():
    global cap, is_camera_on, video_paused
    if is_camera_on:
        stop_webcam()
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        is_camera_on = True
        video_paused = False
        update_canvas()


# Function to update the Canvas with the webcam or video frame
def update_canvas():
    global is_camera_on, video_paused, track_history, crossed_objects

    if is_camera_on and not video_paused:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (1020, 500))

            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            boxes = results[0].boxes.xywh.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            for i, (box, track_id, class_info) in enumerate(zip(boxes, track_ids, classes)):
                x, y, w, h = box
                x, y = round(float(x)), round(float(y))

                track = track_history[track_id]
                track.append((x, y))

                if len(track) > 10:  # Keep only the last 10 points
                    track.pop(0)

                # Check if the object crosses the line
                if START[0] < x < END[0] and abs(y - START[1]) < 20:
                    if track_id not in crossed_objects:
                        crossed_objects[track_id] = class_info

            # Count objects of each class
            count_res = sum(1 for value in crossed_objects.values() if value == 0.0)
            count_cap = sum(1 for value in crossed_objects.values() if value == 1.0)
            count_ind = sum(1 for value in crossed_objects.values() if value == 2.0)
            count_dio = sum(1 for value in crossed_objects.values() if value == 3.0)
            count_led = sum(1 for value in crossed_objects.values() if value == 4.0)
            count_ic = sum(1 for value in crossed_objects.values() if value == 5.0)
            count_ot = sum(1 for value in crossed_objects.values() if value == 6.0)
            count_all = len(crossed_objects)

            # Update StringVars with the counts
            count_res_var.set(str(count_res))
            count_cap_var.set(str(count_cap))
            count_ind_var.set(str(count_ind))
            count_dio_var.set(str(count_dio))
            count_led_var.set(str(count_led))
            count_ic_var.set(str(count_ic))
            count_ot_var.set(str(count_ot))
            count_all_var.set(str(count_all))

            # Draw the line
            cv2.line(annotated_frame, START, END, (255, 255, 255), 2)

            # Convert the image to Tkinter format
            photo = ImageTk.PhotoImage(image=Image.fromarray(annotated_frame))
            canvas.img = photo
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        # Schedule the next update
        canvas.after(10, update_canvas)


# Function to quit the application
def quit_app():
    stop_webcam()
    root.quit()
    root.destroy()


# Create the main Tkinter window
root = tk.Tk()
root.title("YOLO v8 My App")

# Create a Canvas widget to display the webcam feed or video
canvas_frame = tk.Frame(root)
canvas_frame.pack(fill='both', expand=True)
canvas = tk.Canvas(root, width=1020, height=500)
canvas.pack(fill='both', expand=True)

# Read class names from the coco.txt file
class_list = read_classes_from_file('coco.txt')

# Create dropdown menu for class selection
class_selection = tk.StringVar()
class_selection.set("All")
class_selection_label = tk.Label(root, text="Select Class:")
class_selection_label.pack(side='left')
class_selection_entry = tk.OptionMenu(root, class_selection, "All", *class_list)
class_selection_entry.pack(side='left')

# Create a frame to hold the buttons
button_frame = tk.Frame(root)
button_frame.pack(fill='x')

# Create buttons
play_button = tk.Button(button_frame, text="Play", command=start_webcam)
play_button.pack(side='left')

stop_button = tk.Button(button_frame, text="Stop", command=stop_webcam)
stop_button.pack(side='left')

file_button = tk.Button(button_frame, text="Select File", command=select_file)
file_button.pack(side='left')

pause_button = tk.Button(button_frame, text="Pause/Resume", command=pause_resume_video)
pause_button.pack(side='left')

quit_button = tk.Button(button_frame, text="Quit", command=quit_app)
quit_button.pack(side='left')

# Create StringVar variables to store the counts
count_res_var = tk.StringVar()
count_cap_var = tk.StringVar()
count_ind_var = tk.StringVar()
count_dio_var = tk.StringVar()
count_led_var = tk.StringVar()
count_ic_var = tk.StringVar()
count_ot_var = tk.StringVar()
count_all_var = tk.StringVar()

# Create a frame to hold the count display
# Create a frame to hold the count display below the button frame
# Create a frame to hold the count display below the button frame
count_frame = tk.Frame(root)
count_frame.pack(fill='x', padx=10, pady=(10, 0))

# Create Entry widgets to display the counts in a horizontal row
tk.Label(count_frame, text="Resistors").pack(side='left', padx=(0, 5))
entry_res = tk.Entry(count_frame, textvariable=count_res_var, state='readonly', width=10)
entry_res.pack(side='left', padx=(0, 20))

tk.Label(count_frame, text="Capacitors").pack(side='left', padx=(0, 5))
entry_cap = tk.Entry(count_frame, textvariable=count_cap_var, state='readonly', width=10)
entry_cap.pack(side='left', padx=(0, 20))

tk.Label(count_frame, text="Inductors").pack(side='left', padx=(0, 5))
entry_ind = tk.Entry(count_frame, textvariable=count_ind_var, state='readonly', width=10)
entry_ind.pack(side='left', padx=(0, 20))

tk.Label(count_frame, text="Diodes").pack(side='left', padx=(0, 5))
entry_dio = tk.Entry(count_frame, textvariable=count_dio_var, state='readonly', width=10)
entry_dio.pack(side='left', padx=(0, 20))

tk.Label(count_frame, text="LEDs").pack(side='left', padx=(0, 5))
entry_led = tk.Entry(count_frame, textvariable=count_led_var, state='readonly', width=10)
entry_led.pack(side='left', padx=(0, 20))

tk.Label(count_frame, text="ICs").pack(side='left', padx=(0, 5))
entry_ic = tk.Entry(count_frame, textvariable=count_ic_var, state='readonly', width=10)
entry_ic.pack(side='left', padx=(0, 20))

tk.Label(count_frame, text="Others").pack(side='left', padx=(0, 5))
entry_ot = tk.Entry(count_frame, textvariable=count_ot_var, state='readonly', width=10)
entry_ot.pack(side='left', padx=(0, 20))

tk.Label(count_frame, text="Total").pack(side='left', padx=(0, 5))
entry_all = tk.Entry(count_frame, textvariable=count_all_var, state='readonly', width=10)
entry_all.pack(side='left', padx=(0, 20))





# Display an initial image on the canvas
initial_image = Image.open('yolo.jpg')
initial_photo = ImageTk.PhotoImage(image=initial_image)
canvas.img = initial_photo
canvas.create_image(0, 0, anchor=tk.NW, image=initial_photo)

# Start the Tkinter main loop
root.mainloop()
