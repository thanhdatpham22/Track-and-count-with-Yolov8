import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread
import cv2

class YOLOThread(Thread):
    def __init__(self, parent):
        Thread.__init__(self)
        self.parent = parent
        self.cap = cv2.VideoCapture(0)  # VideoCapture object for webcam
        self.is_camera_on = False
        self.video_paused = False

    def run(self):
        while True:
            if self.is_camera_on and not self.video_paused:
                ret, frame = self.cap.read()
                if ret:
                    # Call your YOLO script here to process the frame
                    # For example:
                    # processed_frame = your_yolo_function(frame)
                    # Then update the canvas with processed_frame
                    self.parent.update_frame(frame)

    def start_camera(self):
        self.is_camera_on = True

    def stop_camera(self):
        self.is_camera_on = False

    def pause_resume_video(self):
        self.video_paused = not self.video_paused

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.yolo_thread = YOLOThread(self)
        self.yolo_thread.start()

        self.canvas = tk.Canvas(self.parent)
        self.canvas.pack()

        self.after(100, self.update_canvas)

    def update_canvas(self):
        if self.yolo_thread.is_camera_on and not self.yolo_thread.video_paused:
            # Get frame from YOLO thread
            frame = self.yolo_thread.cap.read()[1]
            # Process the frame using your YOLO script
            processed_frame = frame  # Replace this with your YOLO processing function
            # Convert processed frame to an ImageTk object
            image = Image.fromarray(processed_frame)
            self.photo = ImageTk.PhotoImage(image=image)

            # Update canvas with the new frame
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.after(10, self.update_canvas)  # Update every 10 milliseconds

    def quit_app(self):
        self.yolo_thread.stop_camera()
        self.parent.quit()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("YOLO with Tkinter")
    app = MainApplication(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()
