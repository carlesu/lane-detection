import time
import pickle
import numpy as np
import cv2
from scripts.classes.CFrame import Frame
from scripts.classes.CCurves import Curves
import scripts.utils.common_utils as common_utils

video_file = common_utils.get_single_file()
carlibration_file = common_utils.get_single_file(file_type='pkl')
calibration_data = common_utils.load_pickle(carlibration_file)
# video_file = './../../data/vid/video20_001.avi'
capture = cv2.VideoCapture(video_file)
src_points_crop = np.array([(585, 61), (342, 243), (936, 243), (645, 61)], np.float32)
dest_points_crop = np.array([(342, 0), (342, 243), (936, 243), (936, 0)], np.float32)
src_points = np.array([(577, 416), (317, 599), (966, 604), (661, 423)], np.float32)
dest_points = np.array([(317, 0), (317, 720), (966, 720), (966, 0)], np.float32)

count = 0
while 1:
    ret, image = capture.read()
    if image is None:
        break
    start_time = time.time()
    my_frame = Frame()
    my_frame.load_image(image_array=image)
    my_frame.load_calibration(cameraMatrix=calibration_data['camera_matrix'],
                              distCoeffs=calibration_data['distortion_coefficient'])
    my_frame.undistort()
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
        cv2.imwrite(im_name, my_frame.original)


    # my_frame.show_original()
    my_frame.show_processed(wait=0)

    # my_frame.threshold()
    # my_frame.canny()

    # my_frame.show_processed()
    # cv2.imshow('fit', result['image'])
    # cv2.waitKey(0)















