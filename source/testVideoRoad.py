# coding=utf-8
# 2021.4.2 


from cv2 import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import glob
from lanetracker.camera import CameraCalibration
from lanetracker.gradients import get_edges
from lanetracker.perspective import flatten_perspective
from lanetracker.tracker import LaneTracker
from moviepy.editor import VideoFileClip

# create camera object, use compute's camera or vedio file
# vedio = cv2.VideoCapture('path of vedio file')
vedio = cv2.VideoCapture(0)
# time.sleep(10)
calibrate = CameraCalibration(glob.glob('F:\\detecting-road-features\\data\\camera_cal\\calibration*.jpg'), retain_calibration_images=True)
face_cascade = cv2.CascadeClassifier("F:\\ForPython\\OpenCV\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml")
while True:
    # get frame
    check, frame = vedio.read()
    # print(frame)

    # handle frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    calibrated = calibrate(frame)
    lane_tracker = LaneTracker(calibrated)
    overlay_frame = lane_tracker.process(calibrated, draw_lane=True, draw_statistics=True)
    # mpimg.imsave(frame.replace('test_images', 'output_images'), overlay_frame)
    # plt.imshow(overlay_frame)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    resized = cv2.resize(overlay_frame, (int(gray.shape[1]),int(gray.shape[0]))) 
    # show frame
    cv2.imshow('kechen', resized)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    
vedio.release()
cv2.destroyAllWindows()