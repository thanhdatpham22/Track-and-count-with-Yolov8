import sys
from collections import defaultdict
import supervision as sv

import cv2
import numpy as np
from PyQt5.QtCore import Qt,QRect
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen

from ultralytics import YOLO

from full import Ui_MainWindow


class mylabel(QLabel):
    y0 = 0
    x0 = 0
    x1 = 0
    y1 = 0
    flag = False

    def mousePressEvent(self, event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()
        print("start cap", self.x0, self.y0)

    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()
            # print("move", self.x1, self.y1)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        rec = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        painter.drawRect(rec)
class MainWindow(QMainWindow):
    def __init__(self):

        self.qt_img = None
        self.take = False
        self.lb = None

        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)

        self.uic.Start.clicked.connect(self.start_capture_video)
        self.uic.stop.clicked.connect(self.closeEvent)
        self.uic.Button_take.clicked.connect(self.take_pics)
        self.thread = {}

    def mouseReleaseEvent(self, event):
        if self.take:
            data = [self.lb.y0, self.lb.y1, self.lb.x0, self.lb.x1]
            self.thread[1].take_pic(data)
    def take_pics(self):
        data = [165, 202, 90, 127]
        #data = [200, 202, 200, 200]
        self.thread[1].take_pic(data)
        # show cursor
        self.lb = mylabel(self)
        self.lb.setGeometry(QRect(20, 0, 331, 251))
        self.lb.setCursor(Qt.CrossCursor)
        self.lb.show()

    def updatevailue(self,count_res, count_cap, count_ind, count_dio, count_led, count_ic, count_ot, count_all):
        self.uic.line_res.setText(str(count_res))
        self.uic.line_cap.setText(str(count_cap))
        self.uic.line_ind.setText(str(count_ind))
        self.uic.line_dio.setText(str(count_dio))
        self.uic.line_led.setText(str(count_led))
        self.uic.line_ic.setText(str(count_ic))
        self.uic.line_ot.setText(str(count_ot))
        self.uic.line_all.setText(str(count_all))
        #self.uic.line_ot_2.setText(str(count_all))

    def closeEvent(self, event):
        self.stop_capture_video()

    def start_capture_video(self):
        self.thread[1] = live_stream(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)
        self.thread[1].counter_signal.connect(self.updatevailue)
        self.thread[1].signal_1.connect(self.show_pic)
        #self.thread[1].signal_rec.connect(self.show_crossed_object)

    def show_pic(self, pic):
        self.take = True
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(pic)
        #self.uic.label1.setPixmap(qt_img)
        #self.uic.label_view_5.setPixmap(qt_img)
    def show_wedcam(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.label_view.setPixmap(qt_img)

    '''def show_crossed_object(self, rect):
        painter = QPainter(self.uic.label1.pixmap())
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        painter.drawRect(rect)
        painter.end()
        self.uic.label1.update()'''

    def convert_cv_qt(self, cv_img):
        label_h = self.uic.label_view.height()
        lebel_w = self.uic.label_view.width()
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(lebel_w, label_h)  # , Qt.KeepAspectRatio
        return QPixmap.fromImage(p)

class live_stream(QThread):
    signal = pyqtSignal(np.ndarray)
    counter_signal = pyqtSignal(int, int, int, int, int, int, int, int)
    signal_1 = pyqtSignal(object)
    signal_rec = pyqtSignal(QRect)
    def __init__(self, index):
        self.data = None
        self.pic = False
        self.index = index
        print("start threading", self.index)
        super(live_stream, self).__init__()


    def run(self):
        model = YOLO("best_yolov8s.pt")  # load a pretrained model (recommended for training)
        results = model.track(r"E:\NCKH_Part_2\Video test\lk_doc.mp4", show=True, stream=True)  # List of Results objects
        START = sv.Point(0, 500)
        END = sv.Point(1000, 500)

        # Store the track history
        track_history = defaultdict(lambda: [])

        crossed_objects = {}

        count_res = 0
        count_cap = 0
        count_ind = 0
        count_dio = 0
        count_led = 0
        count_ic = 0
        count_ot = 0
        count_all = 0
        for result, annotated_frame in results:
            boxes = result[0].boxes.xywh.cpu().tolist()
            track_ids = result[0].boxes.id.int().cpu().tolist()
            classtrack = result[0].boxes.cls.cpu().tolist()
            annotated_frame = result[0].plot()
            self.signal.emit(annotated_frame)
            if self.pic:
                print(annotated_frame.shape)
                print(self.data)

                y0 = int(self.data[0] * 4)
                y1 = int(self.data[1] * 4)
                x0 = int(self.data[2] * 5.1)
                x1 = int(self.data[3] * 5.1)

                crop_img = annotated_frame[y0:y1, x0:x1]
                self.signal_1.emit(crop_img)
                self.pic = False

            for i, (box, track_id, class_info) in enumerate(zip(boxes, track_ids, classtrack)):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((round(float(x), 1), round(float(y), 1)))  # x, y center point

                if len(track_ids) > 10:  # retain 10 tracks for 10 frames
                    track.pop(0)
                    if START.x < x < END.x and abs(y - START.y) < 20:
                        if i < len(classtrack):  # Ensure index is within the bounds of classtrack
                            class_info = classtrack[i]  # Use index instead of track_id
                            if track_id not in crossed_objects:
                                crossed_objects[track_id] = {'class': class_info}

                        count_res = sum(1 for value in crossed_objects.values() if value.get('class') == 0.0)  # loop for crossed_object.values and get values has 'class'==0.0
                        count_cap = sum(1 for value in crossed_objects.values() if value.get('class') == 1.0)
                        count_ind = sum(1 for value in crossed_objects.values() if value.get('class') == 2.0)
                        count_dio = sum(1 for value in crossed_objects.values() if value.get('class') == 3.0)
                        count_led = sum(1 for value in crossed_objects.values() if value.get('class') == 4.0)
                        count_ic = sum(1 for value in crossed_objects.values() if value.get('class') == 5.0)
                        count_ot = sum(1 for value in crossed_objects.values() if value.get('class') == 6.0)
                        count_all = len(crossed_objects)
                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)),
                                      (int(x + w / 2), int(y + h / 2)),
                                      (244, 0, 0), 2)
                    cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (255, 255, 255), 2)


            self.counter_signal.emit(count_res, count_cap, count_ind, count_dio, count_led, count_ic, count_ot, count_all)

    def take_pic(self, data):
        self.pic = True
        self.data = data
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
