from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import PIL.Image, PIL.ImageTk
from time import sleep
from threading import Thread
import tkinter
import cv2
from ultralytics import YOLO
# import threading
# import subprocess
model = YOLO('best.pt')

import csv
#from data_csv import*

# from detect_trt_nms_track_compare_usb2 import*
#from detect_trt_nms_track_compare_rtsp2 import*


# khai bao thong tin giao thuc RTSP
# rtsp_username = "admin"
# rtsp_password = "123456789a"
# rtsp_IP = "192.168.99.2"
# rtsp_port = "544"
# stream_number = '02'  ## main stream number = 01, sub stream number = 02

# Camera_name = ['Cam1','Cam2', 'Cam3', 'Cam4', 'Cam5','Cam6', 'Cam7', 'Cam8']

# link_rtsp = [
# # "rtsp://" + rtsp_username + ":" + rtsp_password + "@" + rtsp_IP + ":554/Streaming/channels/" + "1" + stream_number,
# # "rtsp://" + rtsp_username + ":" + rtsp_password + "@" + rtsp_IP + ":554/Streaming/channels/" + "2" + stream_number,
# # "rtsp://" + rtsp_username + ":" + rtsp_password + "@" + rtsp_IP + ":554/Streaming/channels/" + "3" + stream_number,
# # "rtsp://" + rtsp_username + ":" + rtsp_password + "@" + rtsp_IP + ":554/Streaming/channels/" + "4" + stream_number,
# "rtsp://" + rtsp_username + ":" + rtsp_password + "@" + rtsp_IP + ":554/Streaming/channels/" + "5" + stream_number,
# # "rtsp://" + rtsp_username + ":" + rtsp_password + "@" + rtsp_IP + ":554/Streaming/channels/" + "6" + stream_number,
# # "rtsp://" + rtsp_username + ":" + rtsp_password + "@" + rtsp_IP + ":554/Streaming/channels/" + "7" + stream_number,
# # "rtsp://" + rtsp_username + ":" + rtsp_password + "@" + rtsp_IP + ":554/Streaming/channels/" + "8" + stream_number,
# ]

# rtsp://admin:123456789a@192.168.99.2:554/Streaming/channels/502

if __name__ == '__main__':


    window = Tk()
    window.title("Fire smoke detection")
    width= window.winfo_screenwidth()
    height= window.winfo_screenheight()
    window.geometry("%dx%d" % (width, height))
    # window.resizable(False, False)

    ## Doc cam_model va rtsp link from database
    '''cam_model_ls=[]
    rtsp_link_ls=[]
    cam_model_ls, rtsp_link_ls=read_csv_rowall('database.csv')

    ## Doc came mode va rtsp link de su dung
    cam_model_use= []
    rtsp_link_use = []
    user = []
    Pass = []
    cam_model_use, rtsp_link_use, user, Pass = read_csv_row0('database.csv')

    rtsp_link_use = [rtsp_link_use[:7] + user + ':' + Pass + rtsp_link_use[15:]]
    print('RTSP link:',cam_model_use[:],rtsp_link_use[:])

    ## Start detect------------------------------
    # yolo = YOLOR(model_weights = "/home/jetson/Desktop/Machine_learning/YOLOv8/YOLOv9_trt/qat_best_data1.5.1c-converted-end2end-cpu.fp16-int8-grap.trt",names = ['smoke', 'fire'])
    yolo = YOLOR(model_weights = "/home/jetson/Desktop/Machine_learning/YOLOv8/YOLOv9_trt/qat_best_data1.5.1c-converted-end2end-cpu.fp16-int8-grap.trt",names = ['smoke', 'fire'],link_rtsp=rtsp_link_use)
    
    ## Draw GUI ----------------------------------------------

    combo_cam_model = Combobox(window, text = 'Camera model',width=10)
    combo_cam_model['values'] = (" ")
    # combo_cam_model.current(0)
    combo_cam_model.grid(column=2, row=2, columnspan=1, rowspan=1,padx=0, pady=5,  ipadx=0, ipady=0,sticky=W+N)

    ## ghi danh sach cam_model vao combo_cam_model
    for i, cam_model in enumerate(cam_model_ls):
        combo_cam_model['values'] += (cam_model,)

    ## Update link rtsp tren text theo combo_cam_model
    def text_rtsp_update(event):
        global cam_model_ls, rtsp_link_ls 
        # text_rtsp.delete('1.0', END)
        # text_rtsp.insert(END, rtsp_link_ls[combo_cam_model.current()])
        entry_rtsp.delete(0, END)
        entry_rtsp.insert(END, rtsp_link_ls[combo_cam_model.current()])
    combo_cam_model.bind('<<ComboboxSelected>>', text_rtsp_update)

    ## Save khi nhan Button
    def handleBT():
        global cam_model_ls, rtsp_link_ls, cam_model_use, rtsp_link_use, user, Pass
        # string_user = text_user.get("1.0",'end-1c') 
        # string_pass = text_pass.get("1.0",'end-1c') 
        string_user = entry_user.get() 
        string_pass = entry_pass.get() 
        string_cobo = combo_cam_model.get()
        # string_text = text_rtsp.get("1.0",'end-1c') 
        string_text = entry_rtsp.get() 
        if string_cobo and string_text and string_user and string_pass:
            if (string_cobo not in combo_cam_model['values']):
                writer_csv_add_row('database.csv',[str(string_cobo), str(string_text),str(string_user),str(string_pass)])
                messagebox.showinfo("Notice", "Add new camera success")
            else:
                messagebox.showinfo("Notice", "Save camera selected success")
            writer_csv_row0('database.csv',str(string_cobo), str(string_text),str(string_user),str(string_pass))
        else: 
            messagebox.showinfo("Notice","Can not add camera, please check again!")

        cam_model_ls, rtsp_link_ls=read_csv_rowall('database.csv')
        cam_model_use, rtsp_link_use, user, Pass = read_csv_row0('database.csv')
        rtsp_link_use = rtsp_link_use[:7] + user + ':' + Pass + rtsp_link_use[15:]

        combo_cam_model['values'] = (" ")
        for i, cam_model in enumerate(cam_model_ls):
            combo_cam_model['values'] += (cam_model,)

        print('Cam model list:',cam_model_ls)
        print('RTSP link list',rtsp_link_ls)
        print('RTSP link use:',cam_model_use[:],rtsp_link_use[:])'''


    def handleBT():
        pass
    yolo =model.track(  persist=True, tracker="bytetrack.yaml")
    ### Khai bao text cho User,coloumn 0
    label_user= tkinter.Label(window, text="User", fg="black", font=("Arial", 12))
    label_user.grid(column=0, row=1, columnspan=1, rowspan=1,padx=5, pady=0,  ipadx=0, ipady=0,sticky=W+N)

    entry_user = Entry(window,width=10)
    entry_user.grid(column=0, row=2, columnspan=1, rowspan=1, padx=5, pady=5,  ipadx=0, ipady=0,sticky=W+N)

    # text_user = Text(window, height=1, width=10)
    # text_user.grid(column=0, row=2, columnspan=1, rowspan=1, padx=0, pady=5,  ipadx=0, ipady=0,sticky=W+N)

    ### Khai bao text cho Pass coloumn 1
    label_pass= tkinter.Label(window, text="Pass", fg="black", font=("Arial", 12))
    label_pass.grid(column=1, row=1, columnspan=1, rowspan=1,padx=0, pady=0,  ipadx=0, ipady=0,sticky=W+N)

    entry_pass= Entry(window, show = '*',width=10)
    entry_pass.grid(column=1, row=2, columnspan=1, rowspan=1, padx=0, pady=5,  ipadx=0, ipady=0,sticky=W+N)

    # text_pass = Text(window, height=1, width=15)
    # text_pass.grid(column=1, row=2, columnspan=1, rowspan=1, padx=0, pady=5,  ipadx=0, ipady=0,sticky=W+N)

    ### Khai bao text cho RTSP coloumn 3
    label_text= tkinter.Label(window, text="Link rtsp", fg="black", font=("Arial", 12))
    label_text.grid(column=3, row=1, columnspan=1, rowspan=1,padx=0, pady=0,  ipadx=0, ipady=0,sticky=W+N)

    entry_rtsp = Entry(window,width=80)
    entry_rtsp.grid(column=3, row=2, columnspan=1, rowspan=1, padx=0, pady=5,  ipadx=0, ipady=0,sticky=W+N)

    # text_rtsp = Text(window, height=1, width=80)
    # text_rtsp.grid(column=3, row=2, columnspan=1, rowspan=1, padx=0, pady=5,  ipadx=0, ipady=0,sticky=W+N)

    ### Khai bao button coloumn 4
    button = Button(window,text = "Save", command=handleBT)
    button.grid(column=4, row=2, columnspan=1, rowspan=1, padx=5, pady=5,  ipadx=0, ipady=0,sticky=W+N)

    ## Khai bao combobox column 2
    label_combo= tkinter.Label(window, text="Camera model", fg="black", font=("Arial", 12))
    label_combo.grid(column=2, row=1, columnspan=1, rowspan=1,padx=0, pady=0,  ipadx=0, ipady=0,sticky=W+N)



    ## Khai bao Canvas de hien thi hinh anh
    # canvas = Canvas(window, width = canvas_w, height= canvas_h , bg= "red")
    canvas_w = 640
    canvas_h = 480
    canvas = Canvas(window, width = canvas_w, height= canvas_h)
    canvas.grid(column=0, row=0, columnspan=5, rowspan=1, padx=0, pady=0,  ipadx=0, ipady=0,sticky=N+S+E+W)


    def update_frame():
        global canvas, photo

        frame = yolo.detect_out()
        if frame is not None:

            frame = cv2.resize(frame, (canvas.winfo_width(),canvas.winfo_height()))
             # Chuyen he mau
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert hanh image TK
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                # Show
            canvas.create_image(0,0, image = photo, anchor=tkinter.NW)

        window.after(15, update_frame)

    update_frame()

    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=1)
    window.columnconfigure(2, weight=1)
    window.rowconfigure(0, weight=1) # not needed, this is the default behavior
    window.rowconfigure(1, weight=0)
    window.rowconfigure(2, weight=0)

    window.mainloop()

# from tkinter import *
# import os

# root = Tk()
# termf = Frame(root, height=400, width=500)

# termf.pack(fill=BOTH, expand=YES)
# wid = termf.winfo_id()

# os.system('xterm -into %d -geometry 120x60 -sb &' % wid)

# root.mainloop()


# import csv
# import pandas as pd


# csv_file = pd.read_csv("database.csv")  # Load CSV file
# csv_file.cam_model[0] = 'cam_defaul'
# csv_file.rtsp_link[0]= 'rtsp_link_defaul' # Write the value 10 to column A, row 5 (zero-indexed)
# print(csv_file)
# csv_file.to_csv("database.csv",index=False)  # Save the file
# csv_file.to_csv("database.csv",mode='a',index=False,header=False)  # Save the file

# data = ['Imou','http://admin:admin@192.168.1.20:80/cgi-bin/snapshot.cgi?chn=0&u=admin&p=admin']
# writer_csv('database.csv',data)

# cam_model, rtsp_link = read_csv_row1('database.csv')
# print(cam_model)
# print(rtsp_link)

# read_csv = read_csv_rowall('database.csv')
# print(read_csv)
# for i, row in enumerate(read_csv):
#     print(row)


# with open('database.csv', 'a', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerow(['asd','adsda'])
#     csvfile.close()

# with open("database.csv","r") as csvfile:
#     reader = csv.DictReader(csvfile)
#     # print(reader)
#     for i, row in enumerate(reader):
#         # if i ==0:
#         print(row)

#     csvfile.close()

# with open("database.csv","r",) as csvfile:
#     for row_number, row in enumerate(csvfile.readlines()):
#         if row_number == 0:
#             continue
#         data = row.split(',')
#         print(data[0])
#         print(data[1])
#     csvfile.close()
'''window = Tk()
window.title("Tkinter OpenCV")

video = cv2.VideoCapture(0)
canvas_w = video.get(cv2.CAP_PROP_FRAME_WIDTH) // 2
canvas_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2

canvas = Canvas(window, width = canvas_w, height= canvas_h , bg= "red")
canvas.pack()

bw = 0

def handleBW():
    global bw
    bw = 1 - bw

button = Button(window,text = "Black & White", command=handleBW)
button.pack()

photo = None
count = 0

def send_to_server():
    global button
    sleep(10)
    button.configure(text="ThangNC")
    return
zalo
def update_frame():
    global canvas, photo, bw, count
    # Doc tu camera
    ret, frame = video.read()
    # Ressize
    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    # Chuyen he mau
    if bw==0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert hanh image TK
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
    # Show
    canvas.create_image(0,0, image = photo, anchor=tkinter.NW)

    count = count +1
    if count%10==0:
        #send_to_server()
        thread = Thread(target=send_to_server)
        thread.start()

    window.after(15, update_frame)
update_frame()
window.mainloop()'''