import time
import pickle
import numpy as np
import cv2
from scripts.classes.CCurves import Curves
from scripts.classes.CLaneDetector import LaneDetector
import scripts.utils.common_utils as common_utils


# CONSTANTS
# 1 - Image dimensions
image_dimensions = (1200, 1920, 3)
# 2 - ROI Crop
roi_top = 350
roi_bottom = 1200 - 127
roi_left = 0
roi_right = 1920 - 0

# 3 - BEV Points
src_points_crop = np.array([(585, 61), (342, 243), (936, 243), (645, 61)], np.float32)
dest_points_crop = np.array([(342, 0), (342, 243), (936, 243), (936, 0)], np.float32)
src_points = np.array([(577, 416), (317, 599), (966, 604), (661, 423)], np.float32)
dest_points = np.array([(317, 0), (317, 720), (966, 720), (966, 0)], np.float32)
# CONSTANTS


video_file = common_utils.get_single_file()
carlibration_file = common_utils.get_single_file(file_type='pkl')
calibration_data = common_utils.load_pickle(carlibration_file)
# video_file = './../../data/vid/video20_001.avi'
capture = cv2.VideoCapture(video_file)
logger = common_utils.create_logger()

# Lane Detector
lane_detector = LaneDetector(logger=logger)
lane_detector.initialize_detector(camera_matrix=calibration_data['camera_matrix'],
                                  dist_coeffs=calibration_data['distortion_coefficient'], dimensions=image_dimensions,
                                  roi_top=roi_top, roi_bottom=roi_bottom, roi_left=roi_left, roi_right=roi_right,
                                  bev_src_points=src_points, bev_dest_points=dest_points)


count = 0
while 1:
    ret, image = capture.read()
    if image is None:
        break
    start_time = time.time()
    lane_detector.process(image_array=image)

    # my_frame.crop(top_margin=615, bottom_margin=140, left_margin=0, right_margin=0)
    # my_frame.show_processed()
    # my_frame.gray()
    # my_frame.nieto(tau=50)
    # my_frame.canny()
    # my_frame.BEV(src_points=src_points, dest_points=dest_points)
    # my_frame.nieto(tau=50)
    ###########
    # p = {'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 205,
    #      'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20}
    # curves = Curves(number_of_windows=9, margin=100, minimum_pixels=50,
    #                 ym_per_pix=30 / 720, xm_per_pix=3.7 / 700)
    # result = curves.fit(my_frame.processed)
    ###########
    print("--- %s seconds ---" % (time.time() - start_time))
    save_images = False
    if save_images:
        im_name = r'./img_out/img_{0}.jpg'.format(count)
        count += 1
        cv2.imwrite(im_name, lane_detector.original)
    # my_frame.show_original()
    lane_detector.show_processed(wait=0)
    # lane_detector.cleanup_data()
    # my_frame.threshold()
    # my_frame.canny()

    # my_frame.show_processed()
    # cv2.imshow('fit', result['image'])
    # cv2.waitKey(0)















