# coding: utf-8

# # Lane Finding


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import glob
import cv2
from lanetracker.camera import CameraCalibration
from lanetracker.gradients import get_edges
from lanetracker.perspective import flatten_perspective
from lanetracker.tracker import LaneTracker
from moviepy.editor import VideoFileClip

calibrate = CameraCalibration(glob.glob('F:\\detecting-road-features\\data\\camera_cal\\calibration*.jpg'), retain_calibration_images=True)

print('Correction images (successfully detected corners):')
plt.figure(figsize = (11.5, 9))
gridspec.GridSpec(5, 4)
# Step through the list and search for chessboard corners
for i, image in enumerate(calibrate.calibration_images_success):
    plt.subplot2grid((5, 4), (i // 4, i % 4), colspan=1, rowspan=1)
    plt.imshow(image)
    plt.axis('off')


# print('\nTest images (failed to detect corners):')
# for i, image in enumerate(calibrate.calibration_images_error):
#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
#     ax1.axis('off')
#     ax1.imshow(image)
#     ax1.set_title('Original', fontsize=10)
#     ax2.axis('off')
#     ax2.imshow(calibrate(image))
#     ax2.set_title('Calibrated', fontsize=10)



image = mpimg.imread('F:\\detecting-road-features\\data\\test_images\\test6.jpg')
result = get_edges(image, separate_channels=True)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
ax1.axis('off')
ax1.imshow(image)
ax1.set_title('Original', fontsize=18)
ax2.axis('off')
ax2.imshow(result)
ax2.set_title('Edges', fontsize=18)





image = mpimg.imread('F:\\detecting-road-features\\data\\test_images\\test6.jpg')
result, _ = flatten_perspective(image)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
ax1.axis('off')
ax1.imshow(image)
ax1.set_title('Original', fontsize=18)
ax2.axis('off')
ax2.imshow(result)
ax2.set_title('Bird\'s eye view', fontsize=18)




for image_name in glob.glob('F:\\detecting-road-features\\data\\test_images\\*.jpg'):
    calibrated = calibrate(mpimg.imread(image_name))
    lane_tracker = LaneTracker(calibrated)
    overlay_frame = lane_tracker.process(calibrated, draw_lane=True, draw_statistics=True)
    mpimg.imsave(image_name.replace('test_images', 'output_images'), overlay_frame)
    plt.imshow(overlay_frame)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.show()